#!/usr/bin/env python3

from pathlib import Path
import sys

import pandas as pd

from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures


# ============================================================
# COMMAND LINE
# ============================================================

TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

BASE_DIR = Path(
    f"top{TOP_N}_openmm_minimized"
)

CONTACT_FILE = (
    BASE_DIR
    / "post_minimization_contacts.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "post_minimization_contacts_enriched.csv"
)

PREFERENCE_FILE = (
    BASE_DIR
    / "residue_feature_preferences.csv"
)


# ============================================================
# RDKit PHARMACOPHORE FEATURE FACTORY
# ============================================================

FEATURE_DEF = (
    Path(RDConfig.RDDataDir)
    / "BaseFeatures.fdef"
)

FEATURE_FACTORY = (
    ChemicalFeatures.BuildFeatureFactory(
        str(FEATURE_DEF)
    )
)


# ============================================================
# LOAD SDF
# ============================================================

def load_minimized_ligand(rank):

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
            f"Rank {rank:03d}: expected one minimized SDF, "
            f"found {len(matches)}"
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


# ============================================================
# FEATURE MAP
# ============================================================

def build_feature_map(mol):

    """
    Returns a mapping:

        RDKit atom index -> set of feature families

    Typical RDKit families:
        Donor
        Acceptor
        Aromatic
        Hydrophobe
        LumpedHydrophobe
        PosIonizable
        NegIonizable
        ZnBinder
    """

    fmap = {
        atom.GetIdx(): set()
        for atom in mol.GetAtoms()
    }

    for feature in FEATURE_FACTORY.GetFeaturesForMol(
        mol
    ):

        family = feature.GetFamily()

        for atom_idx in feature.GetAtomIds():

            fmap[
                atom_idx
            ].add(
                family
            )

    return fmap


# ============================================================
# SIMPLE ATOM DESCRIPTION
# ============================================================

def atom_description(atom):

    symbol = atom.GetSymbol()

    hyb = str(
        atom.GetHybridization()
    )

    aromatic = (
        "aromatic"
        if atom.GetIsAromatic()
        else "aliphatic"
    )

    charge = atom.GetFormalCharge()

    return (
        f"{symbol};"
        f"{hyb};"
        f"{aromatic};"
        f"charge={charge:+d}"
    )


# ============================================================
# MAIN
# ============================================================

contacts = pd.read_csv(
    CONTACT_FILE
)

print(
    f"Loaded {len(contacts):,} contact rows"
)

enriched_rows = []


for rank in sorted(
    contacts["rank"].unique()
):

    rank = int(rank)

    mol = load_minimized_ligand(
        rank
    )

    feature_map = build_feature_map(
        mol
    )

    # The QC script stores ligand_atom_index as the ordinal
    # within the HEAVY-ATOM list, not necessarily the raw
    # RDKit atom index.
    heavy_atom_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 1
    ]

    rank_contacts = contacts[
        contacts["rank"] == rank
    ]

    for _, row in rank_contacts.iterrows():

        heavy_idx = int(
            row["ligand_atom_index"]
        )

        if heavy_idx >= len(
            heavy_atom_indices
        ):

            raise RuntimeError(
                f"Rank {rank:03d}: heavy atom index "
                f"{heavy_idx} out of range"
            )

        rdkit_idx = (
            heavy_atom_indices[
                heavy_idx
            ]
        )

        atom = mol.GetAtomWithIdx(
            rdkit_idx
        )

        features = sorted(
            feature_map[
                rdkit_idx
            ]
        )

        if not features:
            features = [
                "Other"
            ]

        new_row = row.to_dict()

        new_row.update({
            "rdkit_atom_index":
                rdkit_idx,

            "ligand_atom_symbol":
                atom.GetSymbol(),

            "ligand_formal_charge":
                atom.GetFormalCharge(),

            "ligand_aromatic":
                atom.GetIsAromatic(),

            "ligand_hybridization":
                str(
                    atom.GetHybridization()
                ),

            "ligand_total_degree":
                atom.GetTotalDegree(),

            "ligand_total_H":
                atom.GetTotalNumHs(),

            "ligand_feature_families":
                ";".join(
                    features
                ),

            "ligand_atom_description":
                atom_description(
                    atom
                ),
        })

        enriched_rows.append(
            new_row
        )


enriched = pd.DataFrame(
    enriched_rows
)

enriched.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# RESIDUE FEATURE PREFERENCES
# ============================================================

feature_rows = []


for _, row in enriched.iterrows():

    features = str(
        row[
            "ligand_feature_families"
        ]
    ).split(";")

    residue = (
        f"{row['protein_chain']}:"
        f"{row['protein_resname']}"
        f"{int(row['protein_resid'])}"
    )

    for feature in features:

        feature_rows.append({
            "rank":
                int(
                    row["rank"]
                ),

            "residue":
                residue,

            "protein_atom":
                row[
                    "protein_atom"
                ],

            "feature":
                feature,

            "ligand_element":
                row[
                    "ligand_atom_symbol"
                ],

            "distance_A":
                row[
                    "distance_A"
                ],

            "vdw_overlap_A":
                row[
                    "vdw_overlap_A"
                ],
        })


feature_df = pd.DataFrame(
    feature_rows
)


preferences = (
    feature_df
    .groupby(
        [
            "residue",
            "feature",
        ]
    )
    .agg(
        ligands_with_feature=(
            "rank",
            "nunique"
        ),

        total_contacts=(
            "rank",
            "size"
        ),

        minimum_distance_A=(
            "distance_A",
            "min"
        ),

        mean_distance_A=(
            "distance_A",
            "mean"
        ),
    )
    .reset_index()
)


preferences[
    "fraction_of_ligands"
] = (
    preferences[
        "ligands_with_feature"
    ]
    / TOP_N
)


preferences = (
    preferences
    .sort_values(
        [
            "residue",
            "ligands_with_feature",
        ],
        ascending=[
            True,
            False,
        ],
    )
)


preferences.to_csv(
    PREFERENCE_FILE,
    index=False,
)


print()
print(
    f"Wrote enriched contacts:"
)
print(
    OUTPUT_FILE
)

print()
print(
    f"Wrote residue feature preferences:"
)
print(
    PREFERENCE_FILE
)

