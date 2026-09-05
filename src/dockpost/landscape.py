#!/usr/bin/env python3

"""
Generalized interaction-landscape adapter for dock-postprocess.

Public output
-------------
results/contact_landscape/

The validated historical scientific engine currently writes an internal
compatibility directory named contact_landscape_v2. That legacy name is
confined to temporary staging and is not exposed in the generalized
results tree.

This adapter:
    1. creates the historical compatibility layout temporarily,
    2. runs the validated scientific engine,
    3. exports all landscape outputs into results/contact_landscape/,
    4. converts ligand ordinals into stable L#### identities,
    5. converts ligand-pair ordinal columns into stable ligand IDs.

Compatibility ranks and historical version labels remain implementation
details only.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import subprocess
import sys

import pandas as pd


THIS_FILE = Path(__file__).resolve()

PACKAGE_ROOT = (
    THIS_FILE
    .parents[2]
)

CORE_DIR = (
    PACKAGE_ROOT
    / "core"
)

LEGACY_LANDSCAPE = (
    CORE_DIR
    / "analyze_contact_landscape.py"
)


sys.path.insert(
    0,
    str(
        THIS_FILE.parent.parent
    ),
)


from dockpost.workspace import Workspace  # noqa: E402


# ============================================================
# DIRECTORY NAMES
# ============================================================

# Clean generalized public name.
PUBLIC_LANDSCAPE_DIRNAME = (
    "contact_landscape"
)

# Historical directory name expected from the frozen engine.
# This remains internal only.
LEGACY_LANDSCAPE_DIRNAME = (
    "contact_landscape_v2"
)

# Remove this from public results if left by an earlier test.
OLD_PUBLIC_LANDSCAPE_DIRNAME = (
    "contact_landscape_v2"
)


# ============================================================
# IDENTITY MAPPING
# ============================================================

def build_mapping(
    workspace: Workspace,
) -> pd.DataFrame:

    rows = []

    for ligand in workspace.ligand_records():

        rows.append({
            "rank":
                ligand.internal_ordinal,

            "ligand_id":
                ligand.ligand_id,

            "internal_ordinal":
                ligand.internal_ordinal,

            "compound_id":
                ligand.compound_id,

            "pose_index":
                ligand.pose_index,

            "title":
                ligand.title,

            "source_filename":
                ligand.source_filename,

            "source_rank":
                ligand.source_rank,
        })

    return pd.DataFrame(
        rows
    )


def mapping_dict(
    workspace: Workspace,
) -> dict[int, dict]:

    result = {}

    for ligand in workspace.ligand_records():

        result[
            int(
                ligand.internal_ordinal
            )
        ] = {
            "ligand_id":
                ligand.ligand_id,

            "internal_ordinal":
                ligand.internal_ordinal,

            "compound_id":
                ligand.compound_id,

            "pose_index":
                ligand.pose_index,

            "title":
                ligand.title,

            "source_filename":
                ligand.source_filename,

            "source_rank":
                ligand.source_rank,
        }

    return result


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    workspace: Workspace,
):

    required_public_files = [
        workspace.results_dir
        / "post_minimization_QC.csv",

        workspace.results_dir
        / "post_minimization_contacts.csv",

        workspace.results_dir
        / "post_minimization_contacts_enriched.csv",

        workspace.results_dir
        / "residue_feature_preferences.csv",
    ]


    missing = [
        str(path)
        for path
        in required_public_files
        if not path.exists()
    ]


    for ligand in workspace.ligand_records():

        paths = [
            ligand.canonical_sdf,

            workspace.minimized_ligand_sdf(
                ligand.ligand_id
            ),

            workspace.minimized_complex_pdb(
                ligand.ligand_id
            ),
        ]


        for path in paths:

            if not path.exists():

                missing.append(
                    str(
                        path
                    )
                )


    if missing:

        raise RuntimeError(
            "\nInteraction-landscape analysis requires completed "
            "QC and contact enrichment.\n\n"
            "Missing files:\n  "
            + "\n  ".join(
                missing
            )
        )


# ============================================================
# GENERALIZED TABLE -> COMPATIBILITY TABLE
# ============================================================

def make_compatibility_table(
    source_file: Path,
    destination_file: Path,
):

    try:

        df = pd.read_csv(
            source_file
        )

    except pd.errors.EmptyDataError:

        shutil.copy2(
            source_file,
            destination_file,
        )

        return


    compat = (
        df.copy()
    )


    if (
        "internal_ordinal"
        in compat.columns
    ):

        compat[
            "rank"
        ] = (
            compat[
                "internal_ordinal"
            ]
        )


        generalized_columns = [
            "ligand_id",
            "internal_ordinal",
            "compound_id",
            "pose_index",
            "title",
            "source_filename",
            "source_rank",
        ]


        compat = compat.drop(
            columns=[
                column
                for column
                in generalized_columns
                if column
                in compat.columns
            ],
            errors="ignore",
        )


        columns = list(
            compat.columns
        )


        if "rank" in columns:

            columns.remove(
                "rank"
            )

            compat = compat[
                [
                    "rank"
                ]
                + columns
            ]


    compat.to_csv(
        destination_file,
        index=False,
    )


# ============================================================
# STAGE COMPATIBILITY WORKSPACE
# ============================================================

def stage_results(
    workspace: Workspace,
    staging_root: Path,
):

    records = (
        workspace.ligand_records()
    )

    n_ligands = len(
        records
    )


    compat_results = (
        staging_root
        / f"top{n_ligands}_openmm_minimized"
    )

    compat_results.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # RECEPTOR
    # --------------------------------------------------------

    shutil.copy2(
        workspace.receptor,
        staging_root
        / "receptor.pdb",
    )


    # --------------------------------------------------------
    # ORIGINAL + MINIMIZED LIGANDS
    # --------------------------------------------------------

    for ligand in records:

        rank = (
            ligand.internal_ordinal
        )


        compat_original = (
            staging_root
            / (
                f"{rank:03d}__"
                f"{ligand.ligand_id}__"
                "original.sdf"
            )
        )


        shutil.copy2(
            ligand.canonical_sdf,
            compat_original,
        )


        compat_dir = (
            compat_results
            / f"{rank:03d}"
        )

        compat_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        compat_sdf = (
            compat_dir
            / (
                f"{rank:03d}_"
                f"{ligand.ligand_id}_"
                "ligand_minimized.sdf"
            )
        )


        compat_pdb = (
            compat_dir
            / (
                f"{rank:03d}_"
                f"{ligand.ligand_id}_"
                "complex_minimized.pdb"
            )
        )


        shutil.copy2(
            workspace.minimized_ligand_sdf(
                ligand.ligand_id
            ),
            compat_sdf,
        )


        shutil.copy2(
            workspace.minimized_complex_pdb(
                ligand.ligand_id
            ),
            compat_pdb,
        )


    # --------------------------------------------------------
    # ANALYSIS TABLES
    # --------------------------------------------------------

    table_names = [
        "post_minimization_QC.csv",
        "post_minimization_contacts.csv",
        "post_minimization_contacts_enriched.csv",
    ]


    for filename in table_names:

        make_compatibility_table(
            workspace.results_dir
            / filename,

            compat_results
            / filename,
        )


    shutil.copy2(
        workspace.results_dir
        / "residue_feature_preferences.csv",

        compat_results
        / "residue_feature_preferences.csv",
    )


    minimization_summary = (
        workspace.minimization_summary()
    )


    if minimization_summary.exists():

        make_compatibility_table(
            minimization_summary,

            compat_results
            / "minimization_summary.csv",
        )


    return (
        compat_results,
        n_ligands,
    )


# ============================================================
# ORDINAL DETECTION
# ============================================================

SINGLE_ORDINAL_COLUMNS = {
    "rank",
    "ligand_rank",
}


PAIR_ORDINAL_COLUMN_PAIRS = [
    (
        "ligand_A",
        "ligand_B",
    ),
    (
        "rank_A",
        "rank_B",
    ),
    (
        "rank_a",
        "rank_b",
    ),
    (
        "rank_1",
        "rank_2",
    ),
    (
        "rank1",
        "rank2",
    ),
    (
        "ligand1_rank",
        "ligand2_rank",
    ),
    (
        "ligand_1_rank",
        "ligand_2_rank",
    ),
    (
        "ligand_a_rank",
        "ligand_b_rank",
    ),
]


def values_are_valid_ordinals(
    series: pd.Series,
    valid_ordinals: set[int],
) -> bool:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()


    if len(values) == 0:

        return False


    for value in values:

        if not float(
            value
        ).is_integer():

            return False


        if int(
            value
        ) not in valid_ordinals:

            return False


    return True


# ============================================================
# SINGLE-LIGAND TABLE NORMALIZATION
# ============================================================

def normalize_single_ordinal_table(
    df: pd.DataFrame,
    ordinal_column: str,
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    mapping_for_merge = (
        mapping.drop(
            columns=[
                "internal_ordinal"
            ],
            errors="ignore",
        )
        .rename(
            columns={
                "rank":
                    ordinal_column,
            }
        )
    )


    merged = df.merge(
        mapping_for_merge,
        on=ordinal_column,
        how="left",
        validate="many_to_one",
    )


    merged = merged.rename(
        columns={
            ordinal_column:
                "internal_ordinal",
        }
    )


    front = [
        column
        for column
        in [
            "ligand_id",
            "internal_ordinal",
            "compound_id",
            "pose_index",
            "title",
            "source_filename",
            "source_rank",
        ]
        if column
        in merged.columns
    ]


    remainder = [
        column
        for column
        in merged.columns
        if column
        not in front
    ]


    return merged[
        front
        + remainder
    ]


# ============================================================
# PAIR-TABLE NORMALIZATION
# ============================================================

def add_pair_identity(
    df: pd.DataFrame,
    ordinal_a_column: str,
    ordinal_b_column: str,
    identity_map: dict[int, dict],
) -> pd.DataFrame:

    out = (
        df.copy()
    )


    ordinal_a = pd.to_numeric(
        out[
            ordinal_a_column
        ],
        errors="coerce",
    )


    ordinal_b = pd.to_numeric(
        out[
            ordinal_b_column
        ],
        errors="coerce",
    )


    def get_ligand_id(
        value,
    ):

        if pd.isna(
            value
        ):

            return None


        item = identity_map.get(
            int(
                value
            )
        )


        if item is None:

            return None


        return item[
            "ligand_id"
        ]


    ligand_ids_a = ordinal_a.map(
        get_ligand_id
    )

    ligand_ids_b = ordinal_b.map(
        get_ligand_id
    )


    out = out.rename(
        columns={
            ordinal_a_column:
                "internal_ordinal_a",

            ordinal_b_column:
                "internal_ordinal_b",
        }
    )


    out.insert(
        0,
        "ligand_id_a",
        ligand_ids_a,
    )


    out.insert(
        1,
        "ligand_id_b",
        ligand_ids_b,
    )


    # Put the corresponding ordinals immediately after IDs.
    preferred_front = [
        "ligand_id_a",
        "ligand_id_b",
        "internal_ordinal_a",
        "internal_ordinal_b",
    ]


    remainder = [
        column
        for column
        in out.columns
        if column
        not in preferred_front
    ]


    return out[
        preferred_front
        + remainder
    ]


# ============================================================
# NORMALIZE ONE CSV
# ============================================================

def normalize_csv(
    source_file: Path,
    destination_file: Path,
    mapping: pd.DataFrame,
    identity_map: dict[int, dict],
    valid_ordinals: set[int],
):

    try:

        df = pd.read_csv(
            source_file
        )

    except pd.errors.EmptyDataError:

        shutil.copy2(
            source_file,
            destination_file,
        )

        return


    if len(
        df.columns
    ) == 0:

        shutil.copy2(
            source_file,
            destination_file,
        )

        return


    # --------------------------------------------------------
    # PAIR TABLES
    # --------------------------------------------------------

    for (
        ordinal_a,
        ordinal_b,
    ) in PAIR_ORDINAL_COLUMN_PAIRS:

        if (
            ordinal_a in df.columns
            and
            ordinal_b in df.columns
            and
            values_are_valid_ordinals(
                df[
                    ordinal_a
                ],
                valid_ordinals,
            )
            and
            values_are_valid_ordinals(
                df[
                    ordinal_b
                ],
                valid_ordinals,
            )
        ):

            normalized = (
                add_pair_identity(
                    df,
                    ordinal_a,
                    ordinal_b,
                    identity_map,
                )
            )


            normalized.to_csv(
                destination_file,
                index=False,
            )

            return


    # --------------------------------------------------------
    # SINGLE-LIGAND TABLES
    # --------------------------------------------------------

    for ordinal_column in SINGLE_ORDINAL_COLUMNS:

        if (
            ordinal_column in df.columns
            and
            values_are_valid_ordinals(
                df[
                    ordinal_column
                ],
                valid_ordinals,
            )
        ):

            normalized = (
                normalize_single_ordinal_table(
                    df,
                    ordinal_column,
                    mapping,
                )
            )


            normalized.to_csv(
                destination_file,
                index=False,
            )

            return


    # --------------------------------------------------------
    # AGGREGATED / NON-LIGAND TABLE
    # --------------------------------------------------------

    df.to_csv(
        destination_file,
        index=False,
    )


# ============================================================
# EXPORT LANDSCAPE DIRECTORY
# ============================================================

def export_landscape(
    source_dir: Path,
    destination_dir: Path,
    workspace: Workspace,
):

    if destination_dir.exists():

        shutil.rmtree(
            destination_dir
        )


    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    mapping = build_mapping(
        workspace
    )

    identity_map = mapping_dict(
        workspace
    )

    valid_ordinals = {
        int(
            ligand.internal_ordinal
        )
        for ligand
        in workspace.ligand_records()
    }


    for source_path in sorted(
        source_dir.rglob(
            "*"
        )
    ):

        relative = (
            source_path.relative_to(
                source_dir
            )
        )

        destination_path = (
            destination_dir
            / relative
        )


        if source_path.is_dir():

            destination_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            continue


        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        if (
            source_path.suffix.lower()
            == ".csv"
        ):

            normalize_csv(
                source_path,
                destination_path,
                mapping,
                identity_map,
                valid_ordinals,
            )


        else:

            shutil.copy2(
                source_path,
                destination_path,
            )


# ============================================================
# REMOVE OBSOLETE PUBLIC VERSIONED DIRECTORY
# ============================================================

def remove_old_public_directory(
    workspace: Workspace,
):

    old_directory = (
        workspace.results_dir
        / OLD_PUBLIC_LANDSCAPE_DIRNAME
    )


    new_directory = (
        workspace.results_dir
        / PUBLIC_LANDSCAPE_DIRNAME
    )


    if (
        old_directory.exists()
        and
        old_directory != new_directory
    ):

        shutil.rmtree(
            old_directory
        )


# ============================================================
# SUMMARIZE OUTPUT
# ============================================================

def summarize_landscape(
    public_dir: Path,
):

    print()
    print("=" * 100)
    print("GENERALIZED INTERACTION LANDSCAPE")
    print("=" * 100)


    files = sorted(
        path
        for path
        in public_dir.rglob(
            "*"
        )
        if path.is_file()
    )


    print(
        f"Output files: {len(files)}"
    )


    for path in files:

        print(
            "  "
            + str(
                path.relative_to(
                    public_dir
                )
            )
        )


    fingerprint = (
        public_dir
        / "ligand_interaction_fingerprint.csv"
    )


    if fingerprint.exists():

        try:

            df = pd.read_csv(
                fingerprint
            )


            if (
                "ligand_id"
                in df.columns
            ):

                print()
                print(
                    "Ligands represented in interaction fingerprint:"
                )

                print(
                    "  "
                    + ", ".join(
                        df[
                            "ligand_id"
                        ]
                        .dropna()
                        .astype(str)
                        .drop_duplicates()
                        .tolist()
                    )
                )


        except pd.errors.EmptyDataError:

            pass


# ============================================================
# RUN LANDSCAPE ANALYSIS
# ============================================================

def run_landscape(
    workspace: Workspace,
    dry_run: bool = False,
    keep_staging: bool = False,
):

    workspace.ensure_results_dir()


    if not LEGACY_LANDSCAPE.exists():

        raise RuntimeError(
            "\nValidated interaction-landscape engine "
            "is missing:\n"
            f"  {LEGACY_LANDSCAPE}"
        )


    validate_inputs(
        workspace
    )


    staging_root = (
        workspace.root
        / ".dockpost"
        / "native_landscape_staging"
    )


    if staging_root.exists():

        shutil.rmtree(
            staging_root
        )


    staging_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    try:

        (
            compat_results,
            n_ligands,
        ) = stage_results(
            workspace,
            staging_root,
        )


        command = [
            sys.executable,
            str(
                LEGACY_LANDSCAPE
            ),
            str(
                n_ligands
            ),
        ]


        print()
        print("=" * 80)
        print("DOCK-POSTPROCESS NATIVE INTERACTION LANDSCAPE")
        print("=" * 80)

        print(
            f"Workspace: {workspace.root}"
        )

        print(
            f"Ligands:   {n_ligands}"
        )

        print(
            f"Results:   {workspace.results_dir}"
        )

        print()
        print(
            "Scientific engine:"
        )

        print(
            f"  {LEGACY_LANDSCAPE}"
        )


        if dry_run:

            print()
            print(
                "Staged ligand mapping:"
            )


            for ligand in workspace.ligand_records():

                print(
                    f"  "
                    f"{ligand.internal_ordinal:03d} "
                    f"-> {ligand.ligand_id}"
                )


            print()
            print(
                "Public output:"
            )

            print(
                "  "
                + str(
                    workspace.results_dir
                    / PUBLIC_LANDSCAPE_DIRNAME
                )
            )


            print()
            print(
                "DRY RUN:"
            )

            print(
                f"  cwd = {staging_root}"
            )

            print(
                "  "
                + " ".join(
                    command
                )
            )

            return


        # ====================================================
        # RUN VALIDATED ENGINE
        # ====================================================

        subprocess.run(
            command,
            cwd=str(
                staging_root
            ),
            check=True,
        )


        legacy_output = (
            compat_results
            / LEGACY_LANDSCAPE_DIRNAME
        )


        if not legacy_output.exists():

            raise RuntimeError(
                "\nInteraction-landscape engine completed but "
                "the expected internal output directory "
                "was not produced:\n"
                f"  {legacy_output}"
            )


        public_output = (
            workspace.results_dir
            / PUBLIC_LANDSCAPE_DIRNAME
        )


        export_landscape(
            legacy_output,
            public_output,
            workspace,
        )


        # Delete any earlier public troubleshooting directory.
        remove_old_public_directory(
            workspace
        )


        summarize_landscape(
            public_output
        )


        print()
        print("=" * 80)
        print("INTERACTION LANDSCAPE COMPLETE")
        print("=" * 80)

        print(
            "Wrote:"
        )

        print(
            f"  {public_output}"
        )


    finally:

        if (
            not keep_staging
            and
            staging_root.exists()
        ):

            shutil.rmtree(
                staging_root
            )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generalized chemistry-aware interaction-landscape analysis."
        )
    )


    parser.add_argument(
        "--results",
        required=True,
        help=(
            "Generalized analysis workspace "
            "or its results directory."
        ),
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show staging and execution plan without "
            "running landscape analysis."
        ),
    )


    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help=(
            "Keep temporary compatibility staging files."
        ),
    )


    args = parser.parse_args()


    workspace = Workspace.from_path(
        Path(
            args.results
        )
    )


    run_landscape(
        workspace,
        dry_run=args.dry_run,
        keep_staging=args.keep_staging,
    )


if __name__ == "__main__":
    main()
