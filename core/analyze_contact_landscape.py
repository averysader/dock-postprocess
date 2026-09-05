#!/usr/bin/env python3

from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, ChemicalFeatures, RDConfig

from Bio.PDB import PDBParser

from sklearn.cluster import AgglomerativeClustering


# ============================================================
# COMMAND LINE
# ============================================================

parser_cli = argparse.ArgumentParser(
    description=(
        "Analyze minimized docking poses as a chemically typed "
        "protein-ligand interaction landscape."
    )
)

parser_cli.add_argument(
    "top_n",
    nargs="?",
    type=int,
    default=10,
    help="Number of ranked ligands to analyze (default: 10)",
)

args = parser_cli.parse_args()

TOP_N = args.top_n


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    f"top{TOP_N}_openmm_minimized"
)

ENRICHED_CONTACT_FILE = (
    BASE_DIR
    / "post_minimization_contacts_enriched.csv"
)

QC_FILE = (
    BASE_DIR
    / "post_minimization_QC.csv"
)

OUT_DIR = (
    BASE_DIR
    / "contact_landscape_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

CONTACT_CUTOFF = 4.0

HBOND_DA_MAX = 3.5
HBOND_HA_MAX = 2.6
HBOND_ANGLE_MIN = 120.0

SALT_BRIDGE_MAX = 4.0

AROMATIC_CONTACT_MAX = 4.5
HYDROPHOBIC_CONTACT_MAX = 4.2

# Residues contacted by at least this fraction of ligands
# are treated as recurrent anchor residues.
ANCHOR_FREQUENCY = 0.50

# Pair must share this many recurrent anchors.
MIN_SHARED_ANCHORS = 2

# Pair must add at least this many residues beyond the
# better-covered single ligand.
MIN_COVERAGE_GAIN = 2

# Spatial pharmacophore clustering.
SPATIAL_CLUSTER_MAX_DIAMETER_A = 1.75

# Reject clusters that are still too broad.
MAX_HOTSPOT_RMS_RADIUS_A = 1.25
MAX_HOTSPOT_RADIUS_A = 2.00

MIN_HOTSPOT_LIGANDS = 2

# Plot only interactions observed in this many ligands.
MIN_LIGANDS_FOR_PLOT = 2


# ============================================================
# FEATURE FACTORY
# ============================================================

FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    str(
        Path(RDConfig.RDDataDir)
        / "BaseFeatures.fdef"
    )
)


FEATURE_NORMALIZATION = {
    "Hydrophobe": "Hydrophobe",
    "LumpedHydrophobe": "Hydrophobe",
    "Aromatic": "Aromatic",
    "Acceptor": "Acceptor",
    "Donor": "Donor",
    "PosIonizable": "PosIonizable",
    "NegIonizable": "NegIonizable",
    "ZnBinder": "ZnBinder",
}


# ============================================================
# PROTEIN INTERACTION DEFINITIONS
# ============================================================

ACIDIC_ATOMS = {
    ("ASP", "OD1"),
    ("ASP", "OD2"),
    ("GLU", "OE1"),
    ("GLU", "OE2"),
}


BASIC_ATOMS = {
    ("ARG", "NE"),
    ("ARG", "NH1"),
    ("ARG", "NH2"),
    ("LYS", "NZ"),
}


PROTEIN_ACCEPTORS = {
    ("ASP", "OD1"),
    ("ASP", "OD2"),
    ("GLU", "OE1"),
    ("GLU", "OE2"),

    ("ASN", "OD1"),
    ("GLN", "OE1"),

    ("SER", "OG"),
    ("THR", "OG1"),
    ("TYR", "OH"),

    ("HIS", "ND1"),
    ("HIS", "NE2"),
    ("HID", "NE2"),
    ("HIE", "ND1"),

    ("MET", "SD"),

    # Backbone O handled separately below.
}


PROTEIN_DONORS = {
    ("ARG", "NE"),
    ("ARG", "NH1"),
    ("ARG", "NH2"),

    ("LYS", "NZ"),

    ("ASN", "ND2"),
    ("GLN", "NE2"),

    ("SER", "OG"),
    ("THR", "OG1"),
    ("TYR", "OH"),

    ("HIS", "ND1"),
    ("HIS", "NE2"),
    ("HID", "ND1"),
    ("HIE", "NE2"),

    ("TRP", "NE1"),
}


AROMATIC_RESIDUES = {
    "PHE",
    "TYR",
    "TRP",
    "HIS",
    "HID",
    "HIE",
    "HIP",
}


HYDROPHOBIC_RESIDUES = {
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "MET",
    "PHE",
    "TYR",
    "TRP",
    "PRO",
    "CYS",
    "CYM",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

pdb_parser = PDBParser(
    QUIET=True
)


def residue_label(
    chain,
    resname,
    resid,
):

    return (
        f"{chain}:"
        f"{resname}"
        f"{int(resid)}"
    )


def angle_degrees(
    a,
    b,
    c,
):
    """
    Angle A-B-C in degrees.
    """

    v1 = (
        np.asarray(a)
        - np.asarray(b)
    )

    v2 = (
        np.asarray(c)
        - np.asarray(b)
    )

    n1 = np.linalg.norm(
        v1
    )

    n2 = np.linalg.norm(
        v2
    )

    if (
        n1 == 0
        or n2 == 0
    ):
        return np.nan

    cosang = (
        np.dot(
            v1,
            v2,
        )
        / (
            n1
            * n2
        )
    )

    cosang = np.clip(
        cosang,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cosang
            )
        )
    )


def load_minimized_mol(
    rank,
):

    hit_dir = (
        BASE_DIR
        / f"{rank:03d}"
    )

    matches = list(
        hit_dir.glob(
            "*_ligand_minimized.sdf"
        )
    )

    if len(matches) != 1:

        raise RuntimeError(
            f"Rank {rank:03d}: expected one minimized "
            f"ligand SDF, found {len(matches)}"
        )

    supplier = Chem.SDMolSupplier(
        str(matches[0]),
        removeHs=False,
        sanitize=True,
    )

    mol = next(
        (
            m
            for m in supplier
            if m is not None
        ),
        None,
    )

    if mol is None:

        raise RuntimeError(
            f"Could not read {matches[0]}"
        )

    return mol


def load_complex_pdb(
    rank,
):

    hit_dir = (
        BASE_DIR
        / f"{rank:03d}"
    )

    matches = list(
        hit_dir.glob(
            "*_complex_minimized.pdb"
        )
    )

    if len(matches) != 1:

        raise RuntimeError(
            f"Rank {rank:03d}: expected one minimized "
            f"complex PDB, found {len(matches)}"
        )

    return matches[0]


# ============================================================
# LIGAND FEATURE MAPPING
# ============================================================

def ligand_feature_map(
    mol,
):

    fmap = {
        atom.GetIdx(): set()
        for atom in mol.GetAtoms()
    }

    for feat in FEATURE_FACTORY.GetFeaturesForMol(
        mol
    ):

        family = feat.GetFamily()

        if (
            family
            not in FEATURE_NORMALIZATION
        ):
            continue

        normalized = (
            FEATURE_NORMALIZATION[
                family
            ]
        )

        for atom_idx in feat.GetAtomIds():

            fmap[
                atom_idx
            ].add(
                normalized
            )

    return fmap


# ============================================================
# LIGAND COORDINATES / HYDROGENS
# ============================================================

def ligand_coordinate_map(
    mol,
):

    conf = mol.GetConformer()

    result = {}

    for atom in mol.GetAtoms():

        idx = atom.GetIdx()

        p = conf.GetAtomPosition(
            idx
        )

        result[
            idx
        ] = np.array(
            [
                p.x,
                p.y,
                p.z,
            ],
            dtype=float,
        )

    return result


def attached_hydrogens(
    mol,
    heavy_idx,
):

    atom = mol.GetAtomWithIdx(
        heavy_idx
    )

    return [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetAtomicNum() == 1
    ]


# ============================================================
# PROTEIN ATOM/HYDROGEN COORDINATES
# ============================================================

def load_protein_atoms(
    pdb_file,
):

    structure = pdb_parser.get_structure(
        "complex",
        str(pdb_file),
    )

    heavy = []
    hydrogens = []


    for atom in structure.get_atoms():

        residue = atom.get_parent()
        chain = residue.get_parent()

        element = (
            atom.element
            .strip()
            .upper()
        )

        record = {
            "chain":
                chain.id,

            "resid":
                residue.id[1],

            "resname":
                residue.resname,

            "atom":
                atom.name,

            "element":
                element,

            "coord":
                atom.coord.astype(float),
        }

        if element == "H":

            hydrogens.append(
                record
            )

        else:

            heavy.append(
                record
            )


    return heavy, hydrogens


def protein_hydrogens_near(
    donor_atom,
    protein_hydrogens,
    max_distance=1.35,
):

    donor_xyz = (
        donor_atom[
            "coord"
        ]
    )

    hits = []

    for h in protein_hydrogens:

        # Restrict to the same residue.
        if (
            h["chain"]
            != donor_atom["chain"]
            or
            h["resid"]
            != donor_atom["resid"]
        ):
            continue

        d = np.linalg.norm(
            donor_xyz
            - h["coord"]
        )

        if d <= max_distance:

            hits.append(
                h
            )

    return hits


# ============================================================
# PROTEIN ROLE ASSIGNMENT
# ============================================================

def protein_is_acceptor(
    resname,
    atom_name,
):

    if atom_name == "O":
        return True

    return (
        (
            resname,
            atom_name,
        )
        in PROTEIN_ACCEPTORS
    )


def protein_is_donor(
    resname,
    atom_name,
):

    # Backbone amide nitrogen.
    if atom_name == "N":

        if resname != "PRO":
            return True

    return (
        (
            resname,
            atom_name,
        )
        in PROTEIN_DONORS
    )


def protein_is_acidic(
    resname,
    atom_name,
):

    return (
        (
            resname,
            atom_name,
        )
        in ACIDIC_ATOMS
    )


def protein_is_basic(
    resname,
    atom_name,
):

    return (
        (
            resname,
            atom_name,
        )
        in BASIC_ATOMS
    )


# ============================================================
# H-BOND GEOMETRY
# ============================================================

def ligand_donor_hbond(
    mol,
    ligand_xyz,
    ligand_atom_idx,
    acceptor_xyz,
):

    hydrogens = attached_hydrogens(
        mol,
        ligand_atom_idx,
    )

    if not hydrogens:

        return None


    donor_xyz = (
        ligand_xyz[
            ligand_atom_idx
        ]
    )

    best = None


    for h_idx in hydrogens:

        h_xyz = (
            ligand_xyz[
                h_idx
            ]
        )

        da = np.linalg.norm(
            donor_xyz
            - acceptor_xyz
        )

        ha = np.linalg.norm(
            h_xyz
            - acceptor_xyz
        )

        angle = angle_degrees(
            donor_xyz,
            h_xyz,
            acceptor_xyz,
        )


        if (
            da <= HBOND_DA_MAX
            and
            ha <= HBOND_HA_MAX
            and
            angle >= HBOND_ANGLE_MIN
        ):

            candidate = {
                "DA_distance_A":
                    float(da),

                "HA_distance_A":
                    float(ha),

                "DHA_angle_deg":
                    float(angle),

                "hydrogen_atom_index":
                    h_idx,
            }


            if (
                best is None
                or
                ha
                < best[
                    "HA_distance_A"
                ]
            ):

                best = candidate


    return best


def protein_donor_hbond(
    donor_atom,
    protein_hydrogens,
    ligand_acceptor_xyz,
):

    hydrogens = protein_hydrogens_near(
        donor_atom,
        protein_hydrogens,
    )

    if not hydrogens:

        return None


    donor_xyz = (
        donor_atom[
            "coord"
        ]
    )

    best = None


    for h in hydrogens:

        h_xyz = (
            h[
                "coord"
            ]
        )

        da = np.linalg.norm(
            donor_xyz
            - ligand_acceptor_xyz
        )

        ha = np.linalg.norm(
            h_xyz
            - ligand_acceptor_xyz
        )

        angle = angle_degrees(
            donor_xyz,
            h_xyz,
            ligand_acceptor_xyz,
        )


        if (
            da <= HBOND_DA_MAX
            and
            ha <= HBOND_HA_MAX
            and
            angle >= HBOND_ANGLE_MIN
        ):

            candidate = {
                "DA_distance_A":
                    float(da),

                "HA_distance_A":
                    float(ha),

                "DHA_angle_deg":
                    float(angle),

                "protein_H_atom":
                    h[
                        "atom"
                    ],
            }


            if (
                best is None
                or
                ha
                < best[
                    "HA_distance_A"
                ]
            ):

                best = candidate


    return best


# ============================================================
# PROTEIN ATOM LOOKUP
# ============================================================

def protein_atom_lookup(
    protein_atoms,
):

    lookup = {}

    for atom in protein_atoms:

        key = (
            atom["chain"],
            int(atom["resid"]),
            atom["resname"],
            atom["atom"],
        )

        lookup[
            key
        ] = atom

    return lookup


# ============================================================
# CLASSIFY INTERACTIONS FOR ONE LIGAND
# ============================================================

def classify_rank(
    rank,
    rank_contacts,
):

    mol = load_minimized_mol(
        rank
    )

    lig_features = ligand_feature_map(
        mol
    )

    ligand_xyz = ligand_coordinate_map(
        mol
    )

    pdb_file = load_complex_pdb(
        rank
    )

    (
        protein_atoms,
        protein_hydrogens,
    ) = load_protein_atoms(
        pdb_file
    )

    protein_lookup = protein_atom_lookup(
        protein_atoms
    )


    interaction_rows = []


    for _, row in rank_contacts.iterrows():

        lig_idx = int(
            row[
                "rdkit_atom_index"
            ]
        )

        distance = float(
            row[
                "distance_A"
            ]
        )

        chain = str(
            row[
                "protein_chain"
            ]
        )

        resid = int(
            row[
                "protein_resid"
            ]
        )

        resname = str(
            row[
                "protein_resname"
            ]
        )

        protein_atom_name = str(
            row[
                "protein_atom"
            ]
        )


        key = (
            chain,
            resid,
            resname,
            protein_atom_name,
        )


        if key not in protein_lookup:
            continue


        protein_atom = (
            protein_lookup[
                key
            ]
        )


        features = (
            lig_features
            .get(
                lig_idx,
                set(),
            )
        )


        ligand_atom = (
            mol.GetAtomWithIdx(
                lig_idx
            )
        )


        ligand_coord = (
            ligand_xyz[
                lig_idx
            ]
        )


        base = {
            "rank":
                rank,

            "residue":
                residue_label(
                    chain,
                    resname,
                    resid,
                ),

            "protein_chain":
                chain,

            "protein_resid":
                resid,

            "protein_resname":
                resname,

            "protein_atom":
                protein_atom_name,

            "ligand_atom_index":
                lig_idx,

            "ligand_element":
                ligand_atom.GetSymbol(),

            "ligand_formal_charge":
                ligand_atom.GetFormalCharge(),

            "ligand_features":
                ";".join(
                    sorted(
                        features
                    )
                ),

            "distance_A":
                distance,

            "x":
                ligand_coord[0],

            "y":
                ligand_coord[1],

            "z":
                ligand_coord[2],
        }


        typed = []


        # ----------------------------------------------------
        # SALT BRIDGE
        # ----------------------------------------------------

        if (
            "PosIonizable"
            in features
            and
            protein_is_acidic(
                resname,
                protein_atom_name,
            )
            and
            distance
            <= SALT_BRIDGE_MAX
        ):

            typed.append(
                (
                    "SaltBridge_PosLigand",
                    "PosIonizable",
                    {},
                )
            )


        if (
            "NegIonizable"
            in features
            and
            protein_is_basic(
                resname,
                protein_atom_name,
            )
            and
            distance
            <= SALT_BRIDGE_MAX
        ):

            typed.append(
                (
                    "SaltBridge_NegLigand",
                    "NegIonizable",
                    {},
                )
            )


        # ----------------------------------------------------
        # LIGAND DONOR -> PROTEIN ACCEPTOR
        # ----------------------------------------------------

        if (
            "Donor"
            in features
            and
            protein_is_acceptor(
                resname,
                protein_atom_name,
            )
        ):

            geom = ligand_donor_hbond(
                mol,
                ligand_xyz,
                lig_idx,
                protein_atom[
                    "coord"
                ],
            )

            if geom is not None:

                typed.append(
                    (
                        "HBond_LigandDonor",
                        "Donor",
                        geom,
                    )
                )


        # ----------------------------------------------------
        # PROTEIN DONOR -> LIGAND ACCEPTOR
        # ----------------------------------------------------

        if (
            "Acceptor"
            in features
            and
            protein_is_donor(
                resname,
                protein_atom_name,
            )
        ):

            geom = protein_donor_hbond(
                protein_atom,
                protein_hydrogens,
                ligand_coord,
            )

            if geom is not None:

                typed.append(
                    (
                        "HBond_LigandAcceptor",
                        "Acceptor",
                        geom,
                    )
                )


        # ----------------------------------------------------
        # AROMATIC CONTACT
        # ----------------------------------------------------

        if (
            "Aromatic"
            in features
            and
            resname
            in AROMATIC_RESIDUES
            and
            distance
            <= AROMATIC_CONTACT_MAX
        ):

            typed.append(
                (
                    "AromaticContact",
                    "Aromatic",
                    {},
                )
            )


        # ----------------------------------------------------
        # CATION-AROMATIC PROXIMITY
        #
        # This is deliberately named "proximity", not
        # "cation-pi", because proper cation-pi geometry
        # requires ring centroid/normal analysis.
        # ----------------------------------------------------

        if (
            "Aromatic"
            in features
            and
            protein_is_basic(
                resname,
                protein_atom_name,
            )
            and
            distance
            <= 4.5
        ):

            typed.append(
                (
                    "CationAromatic_ProteinCation",
                    "Aromatic",
                    {},
                )
            )


        if (
            "PosIonizable"
            in features
            and
            resname
            in AROMATIC_RESIDUES
            and
            distance
            <= 4.5
        ):

            typed.append(
                (
                    "CationAromatic_LigandCation",
                    "PosIonizable",
                    {},
                )
            )


        # ----------------------------------------------------
        # HYDROPHOBIC CONTACT
        # ----------------------------------------------------

        if (
            (
                "Hydrophobe"
                in features
                or
                "Aromatic"
                in features
            )
            and
            resname
            in HYDROPHOBIC_RESIDUES
            and
            distance
            <= HYDROPHOBIC_CONTACT_MAX
        ):

            typed.append(
                (
                    "HydrophobicContact",
                    (
                        "Aromatic"
                        if "Aromatic"
                        in features
                        else "Hydrophobe"
                    ),
                    {},
                )
            )


        # ----------------------------------------------------
        # GENERIC CLOSE CONTACT
        # ----------------------------------------------------

        if not typed:

            typed.append(
                (
                    "CloseContact",
                    "Other",
                    {},
                )
            )


        for (
            interaction_type,
            spatial_feature,
            extra,
        ) in typed:

            out = (
                base.copy()
            )

            out[
                "interaction_type"
            ] = interaction_type

            out[
                "spatial_feature"
            ] = spatial_feature

            out.update(
                extra
            )

            interaction_rows.append(
                out
            )


    return (
        interaction_rows,
        mol,
    )


# ============================================================
# LOAD ENRICHED CONTACTS
# ============================================================

if not ENRICHED_CONTACT_FILE.exists():

    raise RuntimeError(
        "\nMissing:\n"
        f"  {ENRICHED_CONTACT_FILE}\n\n"
        "Run:\n"
        f"  dock-enrich {TOP_N}\n"
        "first."
    )


contacts = pd.read_csv(
    ENRICHED_CONTACT_FILE
)


contacts = contacts[
    contacts[
        "distance_A"
    ]
    <= CONTACT_CUTOFF
].copy()


contacts[
    "rank"
] = (
    contacts[
        "rank"
    ]
    .astype(int)
)


print()
print("=" * 80)
print("DOCK-LANDSCAPE V2")
print("=" * 80)

print(
    f"Top N:              {TOP_N}"
)

print(
    f"Raw contact rows:   {len(contacts):,}"
)

print(
    f"Output directory:   {OUT_DIR}"
)


# ============================================================
# CLASSIFY ALL INTERACTIONS
# ============================================================

all_interactions = []

molecule_cache = {}


for rank in range(
    1,
    TOP_N + 1,
):

    rank_contacts = (
        contacts[
            contacts[
                "rank"
            ]
            == rank
        ]
    )


    rows, mol = classify_rank(
        rank,
        rank_contacts,
    )


    all_interactions.extend(
        rows
    )

    molecule_cache[
        rank
    ] = mol


interactions = pd.DataFrame(
    all_interactions
)


interactions.to_csv(
    OUT_DIR
    / "true_interactions.csv",
    index=False,
)


print(
    f"Typed interaction rows: "
    f"{len(interactions):,}"
)


# ============================================================
# RESIDUE CONTACT FREQUENCY
# ============================================================

residue_frequency = (
    interactions
    .groupby(
        "residue"
    )[
        "rank"
    ]
    .nunique()
    .sort_values(
        ascending=False
    )
)


residue_frequency_df = (
    residue_frequency
    .rename(
        "n_ligands"
    )
    .reset_index()
)


residue_frequency_df[
    "fraction_of_ligands"
] = (
    residue_frequency_df[
        "n_ligands"
    ]
    / TOP_N
)


residue_frequency_df.to_csv(
    OUT_DIR
    / "residue_contact_frequency.csv",
    index=False,
)


# ============================================================
# AUTOMATIC ANCHOR RESIDUES
# ============================================================

anchor_residues = set(
    residue_frequency_df.loc[
        residue_frequency_df[
            "fraction_of_ligands"
        ]
        >= ANCHOR_FREQUENCY,
        "residue",
    ]
)


anchor_table = (
    residue_frequency_df[
        residue_frequency_df[
            "residue"
        ]
        .isin(
            anchor_residues
        )
    ]
    .copy()
)


anchor_table.to_csv(
    OUT_DIR
    / "anchor_residues.csv",
    index=False,
)


# ============================================================
# RESIDUE × INTERACTION TYPE MATRIX
# ============================================================

residue_interaction_matrix = (
    interactions
    .groupby(
        [
            "residue",
            "interaction_type",
        ]
    )[
        "rank"
    ]
    .nunique()
    .unstack(
        fill_value=0
    )
)


residue_interaction_matrix.to_csv(
    OUT_DIR
    / "residue_interaction_type_matrix.csv"
)


# ============================================================
# LIGAND × INTERACTION FINGERPRINT
# ============================================================

interactions[
    "interaction_key"
] = (
    interactions[
        "residue"
    ]
    + "|"
    + interactions[
        "interaction_type"
    ]
)


ligand_interaction_fp = (
    interactions
    .groupby(
        [
            "rank",
            "interaction_key",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


ligand_interaction_binary = (
    ligand_interaction_fp > 0
).astype(int)


ligand_interaction_binary = (
    ligand_interaction_binary
    .reindex(
        range(
            1,
            TOP_N + 1
        ),
        fill_value=0,
    )
)


ligand_interaction_binary.to_csv(
    OUT_DIR
    / "ligand_interaction_fingerprint.csv"
)


# ============================================================
# MORGAN CHEMICAL SIMILARITY
# ============================================================

fps = {}


for rank, mol in molecule_cache.items():

    heavy = Chem.RemoveHs(
        mol
    )

    fps[
        rank
    ] = (
        AllChem.GetMorganGenerator(
            radius=2,
            fpSize=2048,
        )
        .GetFingerprint(
            heavy
        )
    )


# ============================================================
# LIGAND PAIR COMPLEMENTARITY V2
# ============================================================

ligand_residue_sets = {}

ligand_interaction_sets = {}


for rank in range(
    1,
    TOP_N + 1,
):

    subset = (
        interactions[
            interactions[
                "rank"
            ]
            == rank
        ]
    )


    ligand_residue_sets[
        rank
    ] = set(
        subset[
            "residue"
        ]
    )


    favorable = subset[
        subset[
            "interaction_type"
        ]
        != "CloseContact"
    ]


    ligand_interaction_sets[
        rank
    ] = set(
        favorable[
            "interaction_key"
        ]
    )


pair_rows = []


complementarity_score_matrix = np.zeros(
    (
        TOP_N,
        TOP_N,
    )
)


for A in range(
    1,
    TOP_N + 1,
):

    for B in range(
        A + 1,
        TOP_N + 1,
    ):

        RA = (
            ligand_residue_sets[
                A
            ]
        )

        RB = (
            ligand_residue_sets[
                B
            ]
        )


        IA = (
            ligand_interaction_sets[
                A
            ]
        )

        IB = (
            ligand_interaction_sets[
                B
            ]
        )


        shared_residues = (
            RA & RB
        )


        shared_anchors = (
            shared_residues
            & anchor_residues
        )


        union_residues = (
            RA | RB
        )


        unique_A = (
            RA - RB
        )

        unique_B = (
            RB - RA
        )


        coverage_gain = (
            len(
                union_residues
            )
            - max(
                len(RA),
                len(RB),
            )
        )


        shared_interactions = (
            IA & IB
        )


        unique_interactions_A = (
            IA - IB
        )

        unique_interactions_B = (
            IB - IA
        )


        jaccard = (
            len(
                shared_residues
            )
            / len(
                union_residues
            )
            if union_residues
            else 0.0
        )


        anchor_overlap_weight = sum(
            float(
                residue_frequency_df.loc[
                    residue_frequency_df[
                        "residue"
                    ]
                    == residue,
                    "fraction_of_ligands",
                ].iloc[0]
            )
            for residue in shared_anchors
        )


        chemical_similarity = (
            DataStructs.TanimotoSimilarity(
                fps[A],
                fps[B],
            )
        )


        # Main complementarity score:
        #
        # rewards:
        #   shared recurrent anchors
        #   additional residue coverage
        #   additional typed interactions
        #
        # does not force chemical similarity.
        score = (
            2.0
            * anchor_overlap_weight
            +
            1.5
            * coverage_gain
            +
            0.25
            * (
                len(
                    unique_interactions_A
                )
                +
                len(
                    unique_interactions_B
                )
            )
        )


        candidate = (
            len(
                shared_anchors
            )
            >= MIN_SHARED_ANCHORS
            and
            coverage_gain
            >= MIN_COVERAGE_GAIN
        )


        if chemical_similarity >= 0.50:

            strategy = (
                "Analog/hybrid candidate"
            )

        elif chemical_similarity >= 0.30:

            strategy = (
                "Moderate scaffold relationship"
            )

        else:

            strategy = (
                "Vector/fragment hypothesis; "
                "scaffolds may be unrelated"
            )


        pair_rows.append({
            "ligand_A":
                A,

            "ligand_B":
                B,

            "shared_residue_count":
                len(
                    shared_residues
                ),

            "shared_anchor_count":
                len(
                    shared_anchors
                ),

            "anchor_overlap_weight":
                anchor_overlap_weight,

            "coverage_gain":
                coverage_gain,

            "shared_typed_interactions":
                len(
                    shared_interactions
                ),

            "unique_typed_interactions_A":
                len(
                    unique_interactions_A
                ),

            "unique_typed_interactions_B":
                len(
                    unique_interactions_B
                ),

            "residue_jaccard":
                jaccard,

            "morgan_tanimoto":
                chemical_similarity,

            "complementarity_score":
                score,

            "candidate_merge_pair":
                candidate,

            "strategy_hint":
                strategy,

            "shared_anchors":
                ";".join(
                    sorted(
                        shared_anchors
                    )
                ),

            "shared_residues":
                ";".join(
                    sorted(
                        shared_residues
                    )
                ),

            "unique_residues_A":
                ";".join(
                    sorted(
                        unique_A
                    )
                ),

            "unique_residues_B":
                ";".join(
                    sorted(
                        unique_B
                    )
                ),

            "unique_interactions_A":
                ";".join(
                    sorted(
                        unique_interactions_A
                    )
                ),

            "unique_interactions_B":
                ";".join(
                    sorted(
                        unique_interactions_B
                    )
                ),
        })


        complementarity_score_matrix[
            A - 1,
            B - 1
        ] = score

        complementarity_score_matrix[
            B - 1,
            A - 1
        ] = score


pair_df = pd.DataFrame(
    pair_rows
)


pair_df = (
    pair_df
    .sort_values(
        [
            "candidate_merge_pair",
            "complementarity_score",
        ],
        ascending=[
            False,
            False,
        ],
    )
)


pair_df.to_csv(
    OUT_DIR
    / "ligand_pair_anchor_compatibility.csv",
    index=False,
)


candidate_pairs = (
    pair_df[
        pair_df[
            "candidate_merge_pair"
        ]
    ]
)


candidate_pairs.to_csv(
    OUT_DIR
    / "candidate_merge_pairs.csv",
    index=False,
)


# ============================================================
# SPATIAL INTERACTION HOTSPOTS
# ============================================================

# Ignore generic contacts.
spatial_source = (
    interactions[
        interactions[
            "interaction_type"
        ]
        != "CloseContact"
    ]
    .copy()
)


# One ligand atom only once per:
#
#   ligand
#   interaction type
#   partner residue
#   spatial feature
#
spatial_source = (
    spatial_source
    .sort_values(
        "distance_A"
    )
    .drop_duplicates(
        subset=[
            "rank",
            "ligand_atom_index",
            "interaction_type",
            "residue",
            "spatial_feature",
        ],
        keep="first",
    )
)


hotspot_rows = []


group_columns = [
    "interaction_type",
    "residue",
    "spatial_feature",
]


for group_key, group in spatial_source.groupby(
    group_columns
):

    (
        interaction_type,
        residue,
        spatial_feature,
    ) = group_key


    if len(
        group
    ) < MIN_HOTSPOT_LIGANDS:
        continue


    xyz = (
        group[
            [
                "x",
                "y",
                "z",
            ]
        ]
        .values
    )


    if len(xyz) == 1:
        continue


    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=(
            SPATIAL_CLUSTER_MAX_DIAMETER_A
        ),
        linkage="complete",
        metric="euclidean",
    )


    labels_cluster = (
        clustering.fit_predict(
            xyz
        )
    )


    group = (
        group.copy()
    )

    group[
        "cluster"
    ] = labels_cluster


    for cluster_id, cluster in group.groupby(
        "cluster"
    ):

        n_ligands = (
            cluster[
                "rank"
            ]
            .nunique()
        )


        if (
            n_ligands
            < MIN_HOTSPOT_LIGANDS
        ):
            continue


        coords = (
            cluster[
                [
                    "x",
                    "y",
                    "z",
                ]
            ]
            .values
        )


        center = (
            coords.mean(
                axis=0
            )
        )


        radial = np.linalg.norm(
            coords
            - center,
            axis=1,
        )


        rms_radius = float(
            np.sqrt(
                np.mean(
                    radial**2
                )
            )
        )


        max_radius = float(
            np.max(
                radial
            )
        )


        if (
            rms_radius
            > MAX_HOTSPOT_RMS_RADIUS_A
            or
            max_radius
            > MAX_HOTSPOT_RADIUS_A
        ):
            continue


        supporting = sorted(
            set(
                int(x)
                for x in cluster[
                    "rank"
                ]
            )
        )


        hotspot_rows.append({
            "interaction_type":
                interaction_type,

            "residue":
                residue,

            "feature":
                spatial_feature,

            "x":
                center[0],

            "y":
                center[1],

            "z":
                center[2],

            "n_points":
                len(
                    cluster
                ),

            "n_ligands":
                n_ligands,

            "fraction_of_ligands":
                n_ligands
                / TOP_N,

            "rms_radius_A":
                rms_radius,

            "max_radius_A":
                max_radius,

            "supporting_ligands":
                ";".join(
                    f"{x:03d}"
                    for x in supporting
                ),
        })


hotspots = pd.DataFrame(
    hotspot_rows
)


if len(
    hotspots
) > 0:

    hotspots = (
        hotspots
        .sort_values(
            [
                "n_ligands",
                "rms_radius_A",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    hotspots[
        "hotspot_id"
    ] = [
        f"H{i:03d}"
        for i in range(
            1,
            len(
                hotspots
            )
            + 1
        )
    ]


    columns = [
        "hotspot_id",
        "interaction_type",
        "residue",
        "feature",
        "x",
        "y",
        "z",
        "n_points",
        "n_ligands",
        "fraction_of_ligands",
        "rms_radius_A",
        "max_radius_A",
        "supporting_ligands",
    ]


    hotspots = (
        hotspots[
            columns
        ]
    )


hotspots.to_csv(
    OUT_DIR
    / "interaction_hotspots_3D.csv",
    index=False,
)


# ============================================================
# WRITE HOTSPOTS AS PDB PSEUDOATOMS
# ============================================================

HOTSPOT_CODES = {
    "HBond_LigandDonor":
        "DON",

    "HBond_LigandAcceptor":
        "ACC",

    "SaltBridge_PosLigand":
        "POS",

    "SaltBridge_NegLigand":
        "NEG",

    "AromaticContact":
        "ARO",

    "CationAromatic_ProteinCation":
        "CAP",

    "CationAromatic_LigandCation":
        "CAL",

    "HydrophobicContact":
        "HYD",
}


hotspot_pdb = (
    OUT_DIR
    / "interaction_hotspots_3D.pdb"
)


with open(
    hotspot_pdb,
    "w",
) as handle:

    handle.write(
        "REMARK dock-landscape v2\n"
    )

    handle.write(
        "REMARK Pseudoatoms represent consensus "
        "interaction positions, not physical atoms.\n"
    )


    if len(
        hotspots
    ) > 0:

        for serial, row in enumerate(
            hotspots.itertuples(),
            start=1,
        ):

            resname = (
                HOTSPOT_CODES
                .get(
                    row.interaction_type,
                    "HOT",
                )
            )


            atom_name = (
                f"H{serial:03d}"
            )[-4:]


            occupancy = min(
                1.0,
                float(
                    row.fraction_of_ligands
                ),
            )


            bfactor = float(
                row.n_ligands
            )


            handle.write(
                f"HETATM"
                f"{serial:5d} "
                f"{atom_name:>4s} "
                f"{resname:>3s} "
                f"H"
                f"{serial:4d}    "
                f"{row.x:8.3f}"
                f"{row.y:8.3f}"
                f"{row.z:8.3f}"
                f"{occupancy:6.2f}"
                f"{bfactor:6.2f}"
                f"          C \n"
            )


    handle.write(
        "END\n"
    )


# ============================================================
# PLOT 1 — RESIDUE INTERACTION TYPE HEATMAP
# ============================================================

plot_residues = (
    residue_frequency_df.loc[
        residue_frequency_df[
            "n_ligands"
        ]
        >= MIN_LIGANDS_FOR_PLOT,
        "residue",
    ]
    .tolist()
)


interaction_types = [
    c
    for c in [
        "HBond_LigandDonor",
        "HBond_LigandAcceptor",
        "SaltBridge_PosLigand",
        "SaltBridge_NegLigand",
        "AromaticContact",
        "CationAromatic_ProteinCation",
        "CationAromatic_LigandCation",
        "HydrophobicContact",
    ]
    if c
    in residue_interaction_matrix.columns
]


if (
    plot_residues
    and
    interaction_types
):

    heat = (
        residue_interaction_matrix
        .reindex(
            plot_residues
        )[
            interaction_types
        ]
        .fillna(
            0
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            max(
                10,
                1.1
                * len(
                    interaction_types
                ),
            ),
            max(
                6,
                0.35
                * len(
                    heat
                ),
            ),
        )
    )


    im = ax.imshow(
        heat.values,
        aspect="auto",
    )


    ax.set_xticks(
        np.arange(
            len(
                interaction_types
            )
        )
    )

    ax.set_xticklabels(
        interaction_types,
        rotation=45,
        ha="right",
    )


    ax.set_yticks(
        np.arange(
            len(
                heat.index
            )
        )
    )

    ax.set_yticklabels(
        heat.index
    )


    ax.set_xlabel(
        "Interaction type"
    )

    ax.set_ylabel(
        "Receptor residue"
    )

    ax.set_title(
        "Residue × chemically typed interaction landscape"
    )


    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Number of ligands"
    )


    fig.tight_layout()


    fig.savefig(
        OUT_DIR
        / "residue_interaction_type_heatmap.png",
        dpi=300,
    )


    plt.close(
        fig
    )


# ============================================================
# PLOT 2 — LIGAND INTERACTION FINGERPRINT HEATMAP
# ============================================================

interaction_support = (
    ligand_interaction_binary
    .sum(
        axis=0
    )
)


top_keys = (
    interaction_support[
        interaction_support
        >= MIN_LIGANDS_FOR_PLOT
    ]
    .sort_values(
        ascending=False
    )
    .index
    .tolist()
)


if top_keys:

    heat = (
        ligand_interaction_binary[
            top_keys
        ]
    )


    fig, ax = plt.subplots(
        figsize=(
            max(
                12,
                0.40
                * len(
                    top_keys
                ),
            ),
            max(
                7,
                0.32
                * TOP_N,
            ),
        )
    )


    im = ax.imshow(
        heat.values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )


    ax.set_xticks(
        np.arange(
            len(
                top_keys
            )
        )
    )

    ax.set_xticklabels(
        top_keys,
        rotation=90,
    )


    ax.set_yticks(
        np.arange(
            TOP_N
        )
    )

    ax.set_yticklabels(
        [
            f"{i:03d}"
            for i in range(
                1,
                TOP_N + 1
            )
        ]
    )


    ax.set_xlabel(
        "Residue | interaction"
    )

    ax.set_ylabel(
        "Ligand rank"
    )

    ax.set_title(
        "Ligand interaction fingerprints"
    )


    fig.tight_layout()


    fig.savefig(
        OUT_DIR
        / "ligand_interaction_fingerprint_heatmap.png",
        dpi=300,
    )


    plt.close(
        fig
    )


# ============================================================
# PLOT 3 — ANCHOR-PRESERVING COMPLEMENTARITY
# ============================================================

labels = [
    f"{i:03d}"
    for i in range(
        1,
        TOP_N + 1
    )
]


fig, ax = plt.subplots(
    figsize=(
        max(
            8,
            0.34
            * TOP_N,
        ),
        max(
            8,
            0.34
            * TOP_N,
        ),
    )
)


im = ax.imshow(
    complementarity_score_matrix,
)


ax.set_xticks(
    np.arange(
        TOP_N
    )
)

ax.set_xticklabels(
    labels,
    rotation=90,
)


ax.set_yticks(
    np.arange(
        TOP_N
    )
)

ax.set_yticklabels(
    labels
)


ax.set_xlabel(
    "Ligand rank"
)

ax.set_ylabel(
    "Ligand rank"
)

ax.set_title(
    "Anchor-preserving complementarity score"
)


cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Complementarity score"
)


fig.tight_layout()


fig.savefig(
    OUT_DIR
    / "anchor_preserving_complementarity_heatmap.png",
    dpi=300,
)


plt.close(
    fig
)


# ============================================================
# PLOT 4 — HOTSPOT SUPPORT
# ============================================================

if len(
    hotspots
) > 0:

    plot_hotspots = (
        hotspots
        .sort_values(
            "n_ligands",
            ascending=True,
        )
        .copy()
    )


    plot_labels = (
        plot_hotspots[
            "hotspot_id"
        ]
        + " "
        + plot_hotspots[
            "interaction_type"
        ]
        + " "
        + plot_hotspots[
            "residue"
        ]
    )


    fig, ax = plt.subplots(
        figsize=(
            11,
            max(
                6,
                0.35
                * len(
                    plot_hotspots
                ),
            ),
        )
    )


    ax.barh(
        plot_labels,
        plot_hotspots[
            "n_ligands"
        ],
    )


    ax.set_xlabel(
        f"Supporting ligands out of {TOP_N}"
    )

    ax.set_ylabel(
        "Consensus interaction hotspot"
    )

    ax.set_title(
        "Spatial interaction hotspot support"
    )


    fig.tight_layout()


    fig.savefig(
        OUT_DIR
        / "interaction_hotspot_support.png",
        dpi=300,
    )


    plt.close(
        fig
    )


# ============================================================
# CONSOLE REPORT
# ============================================================

print()
print("=" * 80)
print("AUTOMATIC ANCHOR RESIDUES")
print("=" * 80)

if len(
    anchor_table
) == 0:

    print(
        "No residues met the anchor-frequency threshold."
    )

else:

    print(
        anchor_table
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


print()
print("=" * 80)
print("TOP TYPED INTERACTIONS")
print("=" * 80)


interaction_summary = (
    interactions[
        interactions[
            "interaction_type"
        ]
        != "CloseContact"
    ]
    .groupby(
        [
            "residue",
            "interaction_type",
        ]
    )[
        "rank"
    ]
    .nunique()
    .rename(
        "n_ligands"
    )
    .reset_index()
    .sort_values(
        "n_ligands",
        ascending=False,
    )
)


print(
    interaction_summary
    .head(
        30
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("TOP ANCHOR-PRESERVING COMPLEMENTARY PAIRS")
print("=" * 80)


if len(
    candidate_pairs
) == 0:

    print(
        "No ligand pairs passed the current criteria."
    )

else:

    print(
        candidate_pairs[
            [
                "ligand_A",
                "ligand_B",
                "shared_anchor_count",
                "coverage_gain",
                "morgan_tanimoto",
                "complementarity_score",
                "strategy_hint",
                "shared_anchors",
            ]
        ]
        .head(
            25
        )
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


print()
print("=" * 80)
print("TOP SPATIAL INTERACTION HOTSPOTS")
print("=" * 80)


if len(
    hotspots
) == 0:

    print(
        "No spatial interaction hotspots passed "
        "the strict clustering criteria."
    )

else:

    print(
        hotspots[
            [
                "hotspot_id",
                "interaction_type",
                "residue",
                "feature",
                "n_ligands",
                "fraction_of_ligands",
                "rms_radius_A",
                "max_radius_A",
            ]
        ]
        .head(
            30
        )
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print()
print(
    f"Results:\n"
    f"  {OUT_DIR}"
)

print()
print(
    "ChimeraX pharmacophore pseudoatoms:\n"
    f"  {hotspot_pdb}"
)

