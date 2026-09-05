#!/usr/bin/env python3

"""
dock-strain-post

Postprocess dock-strain results.

Adds:
    - relaxed-bound conformational strain
    - conformational strain class
    - bound distortion class
    - overall strain interpretation
    - descriptive preorganization score
    - ensemble-level diagnostic plots

Input:
    topN_openmm_minimized/
        ligand_strain/
            strain_summary.csv

Output:
    topN_openmm_minimized/
        ligand_strain/
            strain_summary_enriched.csv
            distortion_vs_intrinsic_strain.png
            preorganization_landscape.png
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# COMMAND LINE
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Postprocess dock-strain results and generate "
        "derived strain metrics and plots."
    )
)

parser.add_argument(
    "top_n",
    nargs="?",
    type=int,
    default=10,
    help=(
        "Number used in the results directory name. "
        "For example, 25 reads top25_openmm_minimized. "
        "Default: 10"
    ),
)

args = parser.parse_args()

TOP_N = args.top_n


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    f"top{TOP_N}_openmm_minimized"
)

STRAIN_DIR = (
    BASE_DIR
    / "ligand_strain"
)

SUMMARY_FILE = (
    STRAIN_DIR
    / "strain_summary.csv"
)

OUT_FILE = (
    STRAIN_DIR
    / "strain_summary_enriched.csv"
)


# ============================================================
# INPUT CHECK
# ============================================================

if not SUMMARY_FILE.exists():

    raise RuntimeError(
        "\nMissing dock-strain summary:\n"
        f"  {SUMMARY_FILE}\n\n"
        "Run dock-strain first."
    )


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    SUMMARY_FILE
)


required = {
    "rank",
    "status",
    "strain_bound_fixed_kcal_mol",
    "receptor_held_distortion_kcal_mol",
    "bound_relax_direct_RMSD_A",
    "closest_solution_RMSD_A",
}


missing = (
    required
    - set(df.columns)
)


if missing:

    raise RuntimeError(
        "strain_summary.csv is missing required columns:\n"
        + "\n".join(
            sorted(missing)
        )
    )


# Work directly on a copy of the original table.
result = df.copy()


# ============================================================
# RELAXED-BOUND CONFORMATIONAL STRAIN
# ============================================================

# dock-strain may already have written this field.
# Recalculate it consistently here either way.

result[
    "strain_bound_relaxed_kcal_mol"
] = (
    result[
        "strain_bound_fixed_kcal_mol"
    ]
    -
    result[
        "receptor_held_distortion_kcal_mol"
    ]
)


# ============================================================
# STRAIN CLASSIFICATION
#
# These are descriptive heuristic bins, not universal
# thermodynamic cutoffs.
# ============================================================

def conformational_class(
    value,
):

    if pd.isna(
        value
    ):
        return "UNKNOWN"

    if value < 2.0:
        return "FAVORABLE"

    if value < 5.0:
        return "MODEST"

    if value < 10.0:
        return "SUBSTANTIAL"

    return "HIGH"


def distortion_class(
    value,
):

    if pd.isna(
        value
    ):
        return "UNKNOWN"

    if value < 3.0:
        return "LOW"

    if value < 8.0:
        return "MODERATE"

    return "HIGH"


result[
    "conformational_strain_class"
] = (
    result[
        "strain_bound_relaxed_kcal_mol"
    ]
    .apply(
        conformational_class
    )
)


result[
    "bound_distortion_class"
] = (
    result[
        "receptor_held_distortion_kcal_mol"
    ]
    .apply(
        distortion_class
    )
)


# ============================================================
# OVERALL STRAIN INTERPRETATION
# ============================================================

def strain_interpretation(
    row,
):

    if (
        row.get(
            "status",
            ""
        )
        != "PASS"
    ):
        return "FAILED"


    conf = row[
        "conformational_strain_class"
    ]

    distortion = row[
        "bound_distortion_class"
    ]

    rms = row[
        "bound_relax_direct_RMSD_A"
    ]


    # Particularly attractive case:
    # low intrinsic strain, little local distortion,
    # and bound geometry barely moves when released.

    if (
        conf == "FAVORABLE"
        and
        distortion == "LOW"
        and
        not pd.isna(
            rms
        )
        and
        rms < 0.75
    ):
        return "PREORGANIZED"


    # Favorable conformational basin underneath
    # substantial local bound-pose distortion.

    if (
        conf == "FAVORABLE"
        and
        distortion == "HIGH"
    ):
        return "LOCALLY_DISTORTED"


    if conf == "HIGH":

        return (
            "HIGH_INTRINSIC_STRAIN"
        )


    if conf == "SUBSTANTIAL":

        return (
            "CONFORMATIONALLY_STRAINED"
        )


    if distortion == "HIGH":

        return "LOCALLY_DISTORTED"


    if conf in {
        "FAVORABLE",
        "MODEST",
    }:

        return "LOW_STRAIN"


    return "INTERMEDIATE"


result[
    "strain_interpretation"
] = result.apply(
    strain_interpretation,
    axis=1,
)


# ============================================================
# PREORGANIZATION SCORE
#
# Descriptive heuristic only.
#
# Larger = more favorable.
#
# Rewards:
#   low intrinsic conformational strain
#   low RMSD from sampled solution conformers to bound pose
#
# This is NOT a binding free energy or probability.
# ============================================================

valid = (
    result[
        "status"
    ]
    == "PASS"
)


result[
    "preorganization_score"
] = np.nan


result.loc[
    valid,
    "preorganization_score"
] = (
    1.0
    / (
        1.0
        +
        result.loc[
            valid,
            "strain_bound_relaxed_kcal_mol"
        ]
        .clip(
            lower=0.0
        )
    )
) * (
    1.0
    / (
        1.0
        +
        result.loc[
            valid,
            "closest_solution_RMSD_A"
        ]
        .clip(
            lower=0.0
        )
    )
)


# ============================================================
# WRITE ENRICHED TABLE
# ============================================================

result.to_csv(
    OUT_FILE,
    index=False,
)


# ============================================================
# PASSED LIGANDS FOR PLOTS
# ============================================================

passed = result[
    result[
        "status"
    ]
    == "PASS"
].copy()


# ============================================================
# PLOT 1
#
# x = receptor-held local distortion
# y = relaxed-bound intrinsic conformational strain
# ============================================================

if len(
    passed
) > 0:

    fig, ax = plt.subplots(
        figsize=(
            8,
            7,
        )
    )


    ax.scatter(
        passed[
            "receptor_held_distortion_kcal_mol"
        ],
        passed[
            "strain_bound_relaxed_kcal_mol"
        ],
        s=55,
    )


    for _, row in passed.iterrows():

        ax.annotate(
            f"{int(row['rank']):03d}",
            (
                row[
                    "receptor_held_distortion_kcal_mol"
                ],
                row[
                    "strain_bound_relaxed_kcal_mol"
                ],
            ),
            xytext=(
                4,
                4,
            ),
            textcoords="offset points",
            fontsize=8,
        )


    # Heuristic strain reference levels.

    for y in [
        2.0,
        5.0,
        10.0,
    ]:

        ax.axhline(
            y,
            linestyle="--",
            linewidth=1,
        )


    ax.set_xlabel(
        "Receptor-held distortion (kcal/mol)"
    )

    ax.set_ylabel(
        "Relaxed-bound conformational strain (kcal/mol)"
    )

    ax.set_title(
        "Local distortion vs intrinsic conformational strain"
    )


    fig.tight_layout()


    fig.savefig(
        STRAIN_DIR
        / "distortion_vs_intrinsic_strain.png",
        dpi=300,
    )


    plt.close(
        fig
    )


# ============================================================
# PLOT 2
#
# x = closest sampled solution conformer RMSD
# y = intrinsic conformational strain
# ============================================================

if len(
    passed
) > 0:

    fig, ax = plt.subplots(
        figsize=(
            8,
            7,
        )
    )


    ax.scatter(
        passed[
            "closest_solution_RMSD_A"
        ],
        passed[
            "strain_bound_relaxed_kcal_mol"
        ],
        s=55,
    )


    for _, row in passed.iterrows():

        ax.annotate(
            f"{int(row['rank']):03d}",
            (
                row[
                    "closest_solution_RMSD_A"
                ],
                row[
                    "strain_bound_relaxed_kcal_mol"
                ],
            ),
            xytext=(
                4,
                4,
            ),
            textcoords="offset points",
            fontsize=8,
        )


    ax.axhline(
        2.0,
        linestyle="--",
        linewidth=1,
    )


    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1,
    )


    ax.set_xlabel(
        "Closest solution conformer RMSD to bound pose (A)"
    )

    ax.set_ylabel(
        "Relaxed-bound conformational strain (kcal/mol)"
    )

    ax.set_title(
        "Energetic and geometric preorganization"
    )


    fig.tight_layout()


    fig.savefig(
        STRAIN_DIR
        / "preorganization_landscape.png",
        dpi=300,
    )


    plt.close(
        fig
    )


# ============================================================
# PLOT 3
#
# Ranked intrinsic strain
# ============================================================

if len(
    passed
) > 0:

    strain_ranked = (
        passed
        .sort_values(
            "strain_bound_relaxed_kcal_mol",
            ascending=True,
        )
    )


    labels = [
        f"{int(x):03d}"
        for x in strain_ranked[
            "rank"
        ]
    ]


    fig, ax = plt.subplots(
        figsize=(
            9,
            max(
                6,
                0.32
                * len(
                    strain_ranked
                ),
            ),
        )
    )


    ax.barh(
        labels,
        strain_ranked[
            "strain_bound_relaxed_kcal_mol"
        ],
    )


    ax.set_xlabel(
        "Relaxed-bound intrinsic strain (kcal/mol)"
    )

    ax.set_ylabel(
        "Ligand rank"
    )

    ax.set_title(
        "Intrinsic conformational strain ranking"
    )


    fig.tight_layout()


    fig.savefig(
        STRAIN_DIR
        / "intrinsic_strain_ranking.png",
        dpi=300,
    )


    plt.close(
        fig
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 112)
print("ENRICHED STRAIN SUMMARY")
print("=" * 112)


display_columns = [
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
    "status",
]


display_columns = [
    col
    for col in display_columns
    if col in result.columns
]


print(
    result[
        display_columns
    ]
    .round(
        3
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 112)

print(
    f"Wrote:\n"
    f"  {OUT_FILE}"
)

print(
    f"  {STRAIN_DIR / 'distortion_vs_intrinsic_strain.png'}"
)

print(
    f"  {STRAIN_DIR / 'preorganization_landscape.png'}"
)

print(
    f"  {STRAIN_DIR / 'intrinsic_strain_ranking.png'}"
)

print()
print(
    "NOTE: strain classifications and preorganization_score "
    "are descriptive heuristics, not thermodynamic observables."
)

