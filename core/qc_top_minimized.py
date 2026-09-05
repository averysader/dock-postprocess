#!/usr/bin/env python3

from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd

from rdkit import Chem
from Bio.PDB import PDBParser


# ============================================================
# COMMAND-LINE SETTINGS
# ============================================================

TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 10


# ============================================================
# PATHS
# ============================================================

MIN_DIR = Path(
    f"top{TOP_N}_openmm_minimized"
)

SUMMARY_OUT = (
    MIN_DIR
    / "post_minimization_QC.csv"
)

CONTACTS_OUT = (
    MIN_DIR
    / "post_minimization_contacts.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CONTACT_CUTOFF = 4.0
CLOSE_CONTACT_CUTOFF = 3.5

CLASH_OVERLAP = 0.40
SEVERE_CLASH_OVERLAP = 0.80

Y220_CHAIN = "B"
Y220_RESID = 220

INSPECT_RMSD = 1.50
INSPECT_CENTROID_SHIFT = 2.00


# ------------------------------------------------------------
# OPTIONAL docking box
#
# Example:
#
# BOX_CENTER = (10.0, -20.0, 5.0)
# BOX_SIZE   = (20.0, 20.0, 20.0)
#
# Leave as None to skip box analysis.
# ------------------------------------------------------------

BOX_CENTER = None
BOX_SIZE = None


# ============================================================
# VDW RADII
# ============================================================

VDW = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "ZN": 1.39,
}


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYM",
    "GLN", "GLU", "GLY", "HIS", "HID", "HIE",
    "HIP", "ILE", "LEU", "LYS", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


parser = PDBParser(QUIET=True)


# ============================================================
# HELPERS
# ============================================================

def vdw_radius(element):
    element = element.upper()

    if element not in VDW:
        return 1.70

    return VDW[element]


def load_first_sdf(filename):

    supplier = Chem.SDMolSupplier(
        str(filename),
        removeHs=False,
        sanitize=True,
    )

    for mol in supplier:

        if mol is not None:
            return mol

    raise RuntimeError(
        f"Could not read {filename}"
    )


def rdkit_heavy_coordinates(mol):

    conf = mol.GetConformer()

    indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 1
    ]

    xyz = np.array([
        [
            conf.GetAtomPosition(i).x,
            conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z,
        ]
        for i in indices
    ])

    elements = [
        mol.GetAtomWithIdx(i)
        .GetSymbol()
        .upper()
        for i in indices
    ]

    return indices, xyz, elements


def get_original_sdf(rank):

    pattern = re.compile(
        rf"^{rank:03d}_.*\.sdf$"
    )

    matches = [
        p
        for p in Path(".").glob("*.sdf")
        if pattern.match(p.name)
    ]

    if len(matches) != 1:

        raise RuntimeError(
            f"Rank {rank:03d}: "
            f"expected one original SDF, "
            f"found {len(matches)}"
        )

    return matches[0]


def get_minimized_files(rank):

    hit_dir = (
        MIN_DIR
        / f"{rank:03d}"
    )

    if not hit_dir.exists():

        raise RuntimeError(
            f"Missing {hit_dir}"
        )


    pdbs = list(
        hit_dir.glob(
            "*_complex_minimized.pdb"
        )
    )

    sdfs = list(
        hit_dir.glob(
            "*_ligand_minimized.sdf"
        )
    )


    if len(pdbs) != 1:

        raise RuntimeError(
            f"{hit_dir}: expected one minimized "
            f"complex PDB, found {len(pdbs)}"
        )


    if len(sdfs) != 1:

        raise RuntimeError(
            f"{hit_dir}: expected one minimized "
            f"ligand SDF, found {len(sdfs)}"
        )


    return pdbs[0], sdfs[0]


def get_protein_heavy_atoms(pdb_file):

    structure = parser.get_structure(
        "complex",
        str(pdb_file),
    )

    atoms = []

    for atom in structure.get_atoms():

        residue = atom.get_parent()
        chain = residue.get_parent()

        if residue.resname not in STANDARD_AA:
            continue

        element = (
            atom.element
            .strip()
            .upper()
        )

        if element == "H":
            continue

        atoms.append({
            "chain": chain.id,
            "resid": residue.id[1],
            "resname": residue.resname,
            "atom": atom.name,
            "element": element,
            "coord": atom.coord.astype(float),
        })

    return atoms


def get_y220_atoms(protein_atoms):

    return [
        a
        for a in protein_atoms
        if (
            a["chain"] == Y220_CHAIN
            and a["resid"] == Y220_RESID
        )
    ]


def inside_box(coords):

    if (
        BOX_CENTER is None
        or BOX_SIZE is None
    ):
        return None


    center = np.array(
        BOX_CENTER,
        dtype=float,
    )

    half = (
        np.array(
            BOX_SIZE,
            dtype=float,
        )
        / 2.0
    )


    lower = center - half
    upper = center + half


    inside = np.all(
        (coords >= lower)
        & (coords <= upper),
        axis=1,
    )


    return {
        "all_inside": bool(
            np.all(inside)
        ),

        "fraction_inside": float(
            np.mean(inside)
        ),
    }


# ============================================================
# POSE DISPLACEMENT
# ============================================================

def pose_displacement(
    original_mol,
    minimized_mol,
):

    (
        orig_idx,
        orig_xyz,
        orig_elements,
    ) = rdkit_heavy_coordinates(
        original_mol
    )


    (
        min_idx,
        min_xyz,
        min_elements,
    ) = rdkit_heavy_coordinates(
        minimized_mol
    )


    if len(orig_xyz) != len(min_xyz):

        raise RuntimeError(
            "Original/minimized heavy-atom counts differ."
        )


    if orig_elements != min_elements:

        raise RuntimeError(
            "Original/minimized heavy-atom element order differs."
        )


    disp = np.linalg.norm(
        min_xyz - orig_xyz,
        axis=1,
    )


    rms = np.sqrt(
        np.mean(
            disp**2
        )
    )


    centroid_original = np.mean(
        orig_xyz,
        axis=0,
    )


    centroid_minimized = np.mean(
        min_xyz,
        axis=0,
    )


    centroid_shift = np.linalg.norm(
        centroid_minimized
        - centroid_original
    )


    return {
        "heavy_atom_RMS_displacement_A":
            float(rms),

        "max_heavy_atom_displacement_A":
            float(
                np.max(disp)
            ),

        "centroid_shift_A":
            float(
                centroid_shift
            ),

        "minimized_xyz":
            min_xyz,

        "elements":
            min_elements,
    }


# ============================================================
# PROTEIN-LIGAND CONTACT ANALYSIS
# ============================================================

def contact_analysis(
    protein_atoms,
    ligand_xyz,
    ligand_elements,
):

    all_pairs = []
    residue_contacts = {}


    for lig_idx, (
        lig_coord,
        lig_elem,
    ) in enumerate(
        zip(
            ligand_xyz,
            ligand_elements,
        )
    ):

        for pa in protein_atoms:

            d = np.linalg.norm(
                lig_coord
                - pa["coord"]
            )


            overlap = (
                vdw_radius(lig_elem)
                + vdw_radius(
                    pa["element"]
                )
                - d
            )


            record = {
                "ligand_atom_index":
                    lig_idx,

                "ligand_element":
                    lig_elem,

                "protein_chain":
                    pa["chain"],

                "protein_resid":
                    pa["resid"],

                "protein_resname":
                    pa["resname"],

                "protein_atom":
                    pa["atom"],

                "protein_element":
                    pa["element"],

                "distance_A":
                    float(d),

                "vdw_overlap_A":
                    float(overlap),
            }


            all_pairs.append(
                record
            )


            if d <= CONTACT_CUTOFF:

                key = (
                    pa["chain"],
                    pa["resid"],
                    pa["resname"],
                )

                residue_contacts.setdefault(
                    key,
                    []
                ).append(
                    record
                )


    nearest = min(
        all_pairs,
        key=lambda x:
            x["distance_A"],
    )


    contacts_40 = [
        x
        for x in all_pairs
        if (
            x["distance_A"]
            <= CONTACT_CUTOFF
        )
    ]


    contacts_35 = [
        x
        for x in all_pairs
        if (
            x["distance_A"]
            <= CLOSE_CONTACT_CUTOFF
        )
    ]


    clashes = [
        x
        for x in all_pairs
        if (
            x["vdw_overlap_A"]
            >= CLASH_OVERLAP
        )
    ]


    severe = [
        x
        for x in all_pairs
        if (
            x["vdw_overlap_A"]
            >= SEVERE_CLASH_OVERLAP
        )
    ]


    return {
        "nearest":
            nearest,

        "contacts_4A":
            len(
                contacts_40
            ),

        "contacts_3.5A":
            len(
                contacts_35
            ),

        "clashes":
            clashes,

        "severe_clashes":
            severe,

        "residue_contacts":
            residue_contacts,

        "all_pairs":
            all_pairs,
    }


# ============================================================
# Y220 ANALYSIS
# ============================================================

def y220_analysis(
    y220_atoms,
    ligand_xyz,
    ligand_elements,
):

    if not y220_atoms:

        return {
            "contacts": 0,
            "nearest_distance": np.nan,
            "nearest_y220_atom": "",
            "nearest_ligand_atom": "",
        }


    pairs = []


    for lig_idx, (
        lig_coord,
        lig_elem,
    ) in enumerate(
        zip(
            ligand_xyz,
            ligand_elements,
        )
    ):

        for ya in y220_atoms:

            d = np.linalg.norm(
                lig_coord
                - ya["coord"]
            )


            pairs.append({
                "lig_idx":
                    lig_idx,

                "lig_elem":
                    lig_elem,

                "y_atom":
                    ya["atom"],

                "distance":
                    float(d),
            })


    nearest = min(
        pairs,
        key=lambda x:
            x["distance"],
    )


    contacts = sum(
        p["distance"]
        <= CONTACT_CUTOFF
        for p in pairs
    )


    return {
        "contacts":
            contacts,

        "nearest_distance":
            nearest["distance"],

        "nearest_y220_atom":
            nearest["y_atom"],

        "nearest_ligand_atom":
            (
                f"{nearest['lig_elem']}"
                f"{nearest['lig_idx']}"
            ),
    }


# ============================================================
# ANALYZE ONE RANK
# ============================================================

def analyze_rank(rank):

    original_sdf = get_original_sdf(
        rank
    )


    (
        complex_pdb,
        minimized_sdf,
    ) = get_minimized_files(
        rank
    )


    print()
    print("=" * 72)
    print(
        f"{rank:03d}"
    )
    print("=" * 72)

    print(
        "Original:",
        original_sdf.name,
    )

    print(
        "Minimized:",
        minimized_sdf.name,
    )


    original_mol = load_first_sdf(
        original_sdf
    )

    minimized_mol = load_first_sdf(
        minimized_sdf
    )


    displacement = pose_displacement(
        original_mol,
        minimized_mol,
    )


    protein_atoms = (
        get_protein_heavy_atoms(
            complex_pdb
        )
    )


    contacts = contact_analysis(
        protein_atoms,
        displacement[
            "minimized_xyz"
        ],
        displacement[
            "elements"
        ],
    )


    y220 = y220_analysis(
        get_y220_atoms(
            protein_atoms
        ),
        displacement[
            "minimized_xyz"
        ],
        displacement[
            "elements"
        ],
    )


    box = inside_box(
        displacement[
            "minimized_xyz"
        ]
    )


    nearest = contacts[
        "nearest"
    ]


    # --------------------------------------------------------
    # Residue contact list
    # --------------------------------------------------------

    contacting_residues = sorted(
        contacts[
            "residue_contacts"
        ].keys(),
        key=lambda x: (
            x[0],
            x[1],
        ),
    )


    residue_string = ";".join(
        f"{chain}:{resname}{resid}"
        for (
            chain,
            resid,
            resname,
        ) in contacting_residues
    )


    # --------------------------------------------------------
    # QC flag
    # --------------------------------------------------------

    reasons = []


    if contacts[
        "severe_clashes"
    ]:

        reasons.append(
            "severe_clash"
        )


    if (
        displacement[
            "heavy_atom_RMS_displacement_A"
        ]
        >= INSPECT_RMSD
    ):

        reasons.append(
            "large_RMS_displacement"
        )


    if (
        displacement[
            "centroid_shift_A"
        ]
        >= INSPECT_CENTROID_SHIFT
    ):

        reasons.append(
            "large_centroid_shift"
        )


    if box is not None:

        if not box[
            "all_inside"
        ]:

            reasons.append(
                "outside_docking_box"
            )


    qc_status = (
        "PASS"
        if not reasons
        else "INSPECT"
    )


    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        f"Heavy-atom RMS displacement: "
        f"{displacement['heavy_atom_RMS_displacement_A']:.3f} A"
    )

    print(
        f"Centroid shift: "
        f"{displacement['centroid_shift_A']:.3f} A"
    )

    print(
        f"Minimum protein-ligand distance: "
        f"{nearest['distance_A']:.3f} A"
    )

    print(
        f"Contacts <=4.0 A: "
        f"{contacts['contacts_4A']}"
    )

    print(
        f"Clashes >=0.4 A overlap: "
        f"{len(contacts['clashes'])}"
    )

    print(
        f"Severe clashes >=0.8 A overlap: "
        f"{len(contacts['severe_clashes'])}"
    )

    print(
        f"Y220 contacts <=4.0 A: "
        f"{y220['contacts']}"
    )

    print(
        "QC:",
        qc_status,
        ",".join(
            reasons
        ),
    )


    # --------------------------------------------------------
    # Per-contact output
    # --------------------------------------------------------

    contact_rows = []


    for pair in contacts[
        "all_pairs"
    ]:

        if (
            pair[
                "distance_A"
            ]
            <= CONTACT_CUTOFF
            or
            pair[
                "vdw_overlap_A"
            ]
            >= CLASH_OVERLAP
        ):

            row = {
                "rank":
                    rank,

                **pair,
            }

            contact_rows.append(
                row
            )


    # --------------------------------------------------------
    # Summary row
    # --------------------------------------------------------

    row = {
        "rank":
            rank,

        "original_sdf":
            original_sdf.name,

        "minimized_sdf":
            minimized_sdf.name,

        "complex_pdb":
            complex_pdb.name,

        "ligand_heavy_atoms":
            minimized_mol
            .GetNumHeavyAtoms(),

        "heavy_atom_RMS_displacement_A":
            displacement[
                "heavy_atom_RMS_displacement_A"
            ],

        "max_heavy_atom_displacement_A":
            displacement[
                "max_heavy_atom_displacement_A"
            ],

        "centroid_shift_A":
            displacement[
                "centroid_shift_A"
            ],

        "minimum_protein_ligand_distance_A":
            nearest[
                "distance_A"
            ],

        "closest_protein_chain":
            nearest[
                "protein_chain"
            ],

        "closest_protein_resname":
            nearest[
                "protein_resname"
            ],

        "closest_protein_resid":
            nearest[
                "protein_resid"
            ],

        "closest_protein_atom":
            nearest[
                "protein_atom"
            ],

        "closest_ligand_atom_index":
            nearest[
                "ligand_atom_index"
            ],

        "protein_ligand_contacts_le_4A":
            contacts[
                "contacts_4A"
            ],

        "protein_ligand_contacts_le_3.5A":
            contacts[
                "contacts_3.5A"
            ],

        "vdw_clashes_ge_0.4A":
            len(
                contacts[
                    "clashes"
                ]
            ),

        "severe_clashes_ge_0.8A":
            len(
                contacts[
                    "severe_clashes"
                ]
            ),

        "maximum_vdw_overlap_A":
            max(
                (
                    x[
                        "vdw_overlap_A"
                    ]
                    for x in contacts[
                        "all_pairs"
                    ]
                ),
                default=np.nan,
            ),

        "Y220_contacts_le_4A":
            y220[
                "contacts"
            ],

        "Y220_nearest_distance_A":
            y220[
                "nearest_distance"
            ],

        "Y220_nearest_atom":
            y220[
                "nearest_y220_atom"
            ],

        "Y220_nearest_ligand_atom":
            y220[
                "nearest_ligand_atom"
            ],

        "contacting_residue_count":
            len(
                contacting_residues
            ),

        "contacting_residues":
            residue_string,

        "QC_status":
            qc_status,

        "QC_reasons":
            ";".join(
                reasons
            ),
    }


    if box is not None:

        row[
            "all_heavy_atoms_inside_box"
        ] = box[
            "all_inside"
        ]

        row[
            "fraction_heavy_atoms_inside_box"
        ] = box[
            "fraction_inside"
        ]


    return (
        row,
        contact_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        f"QC top N: {TOP_N}"
    )

    print(
        f"Reading minimized structures from: "
        f"{MIN_DIR}"
    )

    print()


    if not MIN_DIR.exists():

        raise RuntimeError(
            f"Results directory does not exist: "
            f"{MIN_DIR}"
        )


    summary_rows = []
    all_contact_rows = []


    for rank in range(
        1,
        TOP_N + 1,
    ):

        try:

            (
                summary_row,
                contact_rows,
            ) = analyze_rank(
                rank
            )


            summary_rows.append(
                summary_row
            )


            all_contact_rows.extend(
                contact_rows
            )


        except Exception as exc:

            print()
            print(
                f"ERROR rank "
                f"{rank:03d}: "
                f"{exc}"
            )


            summary_rows.append({
                "rank":
                    rank,

                "QC_status":
                    "FAIL",

                "QC_reasons":
                    str(exc),
            })


    # ========================================================
    # WRITE OUTPUTS
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )


    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )


    contacts_df = pd.DataFrame(
        all_contact_rows
    )


    contacts_df.to_csv(
        CONTACTS_OUT,
        index=False,
    )


    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("POST-MINIMIZATION QC")
    print("=" * 100)


    display_cols = [
        "rank",
        "heavy_atom_RMS_displacement_A",
        "centroid_shift_A",
        "minimum_protein_ligand_distance_A",
        "protein_ligand_contacts_le_4A",
        "vdw_clashes_ge_0.4A",
        "severe_clashes_ge_0.8A",
        "Y220_contacts_le_4A",
        "Y220_nearest_distance_A",
        "QC_status",
        "QC_reasons",
    ]


    display_cols = [
        c
        for c in display_cols
        if c in summary.columns
    ]


    print(
        summary[
            display_cols
        ]
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


    print()

    print(
        f"Wrote: "
        f"{SUMMARY_OUT}"
    )

    print(
        f"Wrote: "
        f"{CONTACTS_OUT}"
    )


if __name__ == "__main__":
    main()

