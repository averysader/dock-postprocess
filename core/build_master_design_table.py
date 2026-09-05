#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# COMMAND LINE
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Merge docking postprocessing, QC, interaction-landscape, "
        "and ligand-strain metrics into a master design table."
    )
)

parser.add_argument(
    "top_n",
    nargs="?",
    type=int,
    default=10,
)

args = parser.parse_args()

TOP_N = args.top_n


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    f"top{TOP_N}_openmm_minimized"
)


QC_FILE = (
    BASE_DIR
    / "post_minimization_QC.csv"
)


MIN_FILE = (
    BASE_DIR
    / "minimization_summary.csv"
)


STRAIN_FILE = (
    BASE_DIR
    / "ligand_strain"
    / "strain_summary_enriched.csv"
)


LANDSCAPE_DIR = (
    BASE_DIR
    / "contact_landscape_v2"
)


INTERACTIONS_FILE = (
    LANDSCAPE_DIR
    / "true_interactions.csv"
)


ANCHORS_FILE = (
    LANDSCAPE_DIR
    / "anchor_residues.csv"
)


OUTPUT_FILE = (
    BASE_DIR
    / "master_design_table.csv"
)


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

for filename in [
    QC_FILE,
    STRAIN_FILE,
    INTERACTIONS_FILE,
    ANCHORS_FILE,
]:

    if not filename.exists():

        raise RuntimeError(
            f"Missing required input:\n  {filename}"
        )


# ============================================================
# LOAD BASE TABLE
# ============================================================

qc = pd.read_csv(
    QC_FILE
)


master = qc.copy()


# ============================================================
# MINIMIZATION SUMMARY
# ============================================================

if MIN_FILE.exists():

    minimization = pd.read_csv(
        MIN_FILE
    )

    keep = [
        c
        for c in [
            "rank",
            "initial_energy_kcal_mol",
            "final_energy_kcal_mol",
            "delta_energy_kcal_mol",
            "ligand_RMS_displacement_A",
            "max_ligand_displacement_A",
        ]
        if c in minimization.columns
    ]


    if "rank" in keep:

        master = master.merge(
            minimization[
                keep
            ],
            on="rank",
            how="left",
            suffixes=(
                "",
                "_min",
            ),
        )


# ============================================================
# STRAIN SUMMARY
# ============================================================

strain = pd.read_csv(
    STRAIN_FILE
)


strain_keep = [
    c
    for c in [
        "rank",
        "strain_bound_fixed_kcal_mol",
        "receptor_held_distortion_kcal_mol",
        "strain_bound_relaxed_kcal_mol",
        "bound_relax_direct_RMSD_A",
        "closest_solution_RMSD_A",
        "closest_solution_delta_E_kcal_mol",
        "conformational_strain_class",
        "bound_distortion_class",
        "strain_interpretation",
        "preorganization_score",
    ]
    if c in strain.columns
]


master = master.merge(
    strain[
        strain_keep
    ],
    on="rank",
    how="left",
)


# ============================================================
# INTERACTION SUMMARY
# ============================================================

interactions = pd.read_csv(
    INTERACTIONS_FILE
)


anchors = pd.read_csv(
    ANCHORS_FILE
)


anchor_set = set(
    anchors[
        "residue"
    ]
)


favorable = interactions[
    interactions[
        "interaction_type"
    ]
    != "CloseContact"
].copy()


# ============================================================
# PER-LIGAND INTERACTION COUNTS
# ============================================================

summary_rows = []


for rank in range(
    1,
    TOP_N + 1,
):

    subset = interactions[
        interactions[
            "rank"
        ]
        == rank
    ]


    fav = favorable[
        favorable[
            "rank"
        ]
        == rank
    ]


    residues = set(
        subset[
            "residue"
        ]
    )


    favorable_residues = set(
        fav[
            "residue"
        ]
    )


    contacted_anchors = (
        residues
        & anchor_set
    )


    typed_keys = set(
        fav[
            "residue"
        ].astype(str)
        + "|"
        + fav[
            "interaction_type"
        ].astype(str)
    )


    type_counts = (
        fav[
            "interaction_type"
        ]
        .value_counts()
        .to_dict()
    )


    summary_rows.append({
        "rank":
            rank,

        "contacted_residue_count":
            len(
                residues
            ),

        "favorable_residue_count":
            len(
                favorable_residues
            ),

        "anchor_residue_count":
            len(
                contacted_anchors
            ),

        "typed_interaction_count":
            len(
                typed_keys
            ),

        "anchor_residues_contacted":
            ";".join(
                sorted(
                    contacted_anchors
                )
            ),

        "favorable_residues":
            ";".join(
                sorted(
                    favorable_residues
                )
            ),

        "n_hbond_ligand_donor":
            type_counts.get(
                "HBond_LigandDonor",
                0,
            ),

        "n_hbond_ligand_acceptor":
            type_counts.get(
                "HBond_LigandAcceptor",
                0,
            ),

        "n_saltbridge_posligand":
            type_counts.get(
                "SaltBridge_PosLigand",
                0,
            ),

        "n_saltbridge_negligand":
            type_counts.get(
                "SaltBridge_NegLigand",
                0,
            ),

        "n_aromatic_contacts":
            type_counts.get(
                "AromaticContact",
                0,
            ),

        "n_hydrophobic_contacts":
            type_counts.get(
                "HydrophobicContact",
                0,
            ),
    })


interaction_summary = pd.DataFrame(
    summary_rows
)


master = master.merge(
    interaction_summary,
    on="rank",
    how="left",
)


# ============================================================
# DESIGN FLAGS
# ============================================================

def design_flag(
    row,
):

    flags = []


    if (
        row.get(
            "QC_status",
            ""
        )
        != "PASS"
    ):

        flags.append(
            "QC_INSPECT"
        )


    strain_class = row.get(
        "strain_interpretation",
        ""
    )


    if strain_class == "PREORGANIZED":

        flags.append(
            "PREORGANIZED"
        )


    if strain_class == "LOCALLY_DISTORTED":

        flags.append(
            "LOCAL_DISTORTION"
        )


    if strain_class in {
        "CONFORMATIONALLY_STRAINED",
        "HIGH_INTRINSIC_STRAIN",
    }:

        flags.append(
            "STRAIN_WARNING"
        )


    if (
        row.get(
            "anchor_residue_count",
            0
        )
        >= 3
    ):

        flags.append(
            "STRONG_ANCHOR_COVERAGE"
        )


    if (
        row.get(
            "typed_interaction_count",
            0
        )
        >= 4
    ):

        flags.append(
            "RICH_INTERACTION_NETWORK"
        )


    return ";".join(
        flags
    )


master[
    "design_flags"
] = master.apply(
    design_flag,
    axis=1,
)


# ============================================================
# SIMPLE DESIGN SCORE
#
# Descriptive heuristic only.
# This is NOT a binding free energy.
# ============================================================

preorg = master[
    "preorganization_score"
].fillna(
    0
)


anchors_norm = (
    master[
        "anchor_residue_count"
    ]
    .fillna(
        0
    )
    / max(
        1,
        len(
            anchor_set
        )
    )
)


typed = (
    master[
        "typed_interaction_count"
    ]
    .fillna(
        0
    )
)


typed_norm = (
    typed
    / max(
        1,
        typed.max()
    )
)


master[
    "design_priority_score"
] = (
    0.40
    * preorg
    +
    0.35
    * anchors_norm
    +
    0.25
    * typed_norm
)


# ============================================================
# SORT
# ============================================================

master = master.sort_values(
    [
        "design_priority_score",
        "rank",
    ],
    ascending=[
        False,
        True,
    ],
)


# ============================================================
# WRITE
# ============================================================

master.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 120)
print("MASTER DESIGN TABLE")
print("=" * 120)


display_cols = [
    "rank",
    "QC_status",
    "strain_bound_fixed_kcal_mol",
    "strain_bound_relaxed_kcal_mol",
    "strain_interpretation",
    "anchor_residue_count",
    "typed_interaction_count",
    "n_hbond_ligand_donor",
    "n_hbond_ligand_acceptor",
    "n_saltbridge_posligand",
    "n_aromatic_contacts",
    "design_priority_score",
    "design_flags",
]


display_cols = [
    c
    for c in display_cols
    if c in master.columns
]


print(
    master[
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
    f"Wrote:\n  {OUTPUT_FILE}"
)

print()
print(
    "IMPORTANT: design_priority_score is a descriptive "
    "multi-objective heuristic, not an affinity prediction."
)

