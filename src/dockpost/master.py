#!/usr/bin/env python3

"""
Generalized master design-table adapter for dock-postprocess.

Public inputs
-------------
results/
    minimization_summary.csv
    post_minimization_QC.csv
    post_minimization_contacts.csv
    post_minimization_contacts_enriched.csv

    contact_landscape/
        ligand_interaction_fingerprint.csv
        ...

    ligand_strain/
        strain_summary.csv
        strain_summary_enriched.csv
        ...

Public output
-------------
results/
    master_design_table.csv

The validated historical master-table engine still expects:
    topN_openmm_minimized/
    contact_landscape_v2/
    rank-numbered ligand identities

Those assumptions are recreated only inside temporary staging.

The generalized public master table uses:
    ligand_id
    internal_ordinal
    source_rank

and does not expose the legacy rank identity.
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

LEGACY_MASTER = (
    CORE_DIR
    / "build_master_design_table.py"
)


sys.path.insert(
    0,
    str(
        THIS_FILE.parent.parent
    ),
)


from dockpost.workspace import Workspace  # noqa: E402


PUBLIC_LANDSCAPE_DIRNAME = (
    "contact_landscape"
)

LEGACY_LANDSCAPE_DIRNAME = (
    "contact_landscape_v2"
)

STRAIN_DIRNAME = (
    "ligand_strain"
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

            "source_file":
                str(
                    ligand.source_file
                ),

            "record_index":
                ligand.pose_index,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    workspace: Workspace,
):

    required = [
        workspace.results_dir
        / "post_minimization_QC.csv",

        workspace.results_dir
        / STRAIN_DIRNAME
        / "strain_summary_enriched.csv",

        workspace.results_dir
        / PUBLIC_LANDSCAPE_DIRNAME
        / "ligand_interaction_fingerprint.csv",
    ]


    missing = [
        str(path)
        for path
        in required
        if not path.exists()
    ]


    if missing:

        raise RuntimeError(
            "\nMaster design-table construction requires "
            "completed QC, strain, and interaction-landscape "
            "analysis.\n\n"
            "Missing files:\n  "
            + "\n  ".join(
                missing
            )
        )


# ============================================================
# GENERALIZED CSV -> LEGACY COMPATIBILITY CSV
# ============================================================

def make_compatibility_csv(
    source_file: Path,
    destination_file: Path,
):

    destination_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


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


    # --------------------------------------------------------
    # SINGLE-LIGAND GENERALIZED IDENTITY
    # --------------------------------------------------------

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


        generalized_identity_columns = [
            "ligand_id",
            "internal_ordinal",
            "compound_id",
            "pose_index",
            "title",
            "source_filename",
            "source_rank",
            "source_file",
            "record_index",
        ]


        compat = compat.drop(
            columns=[
                column
                for column
                in generalized_identity_columns
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


    # --------------------------------------------------------
    # PAIR-TABLE GENERALIZED IDENTITY
    #
    # These tables are not currently required by the master
    # engine, but convert them back cleanly so the staged
    # landscape remains internally consistent.
    # --------------------------------------------------------

    elif (
        "internal_ordinal_a"
        in compat.columns
        and
        "internal_ordinal_b"
        in compat.columns
    ):

        compat = compat.rename(
            columns={
                "internal_ordinal_a":
                    "ligand_A",

                "internal_ordinal_b":
                    "ligand_B",
            }
        )


        compat = compat.drop(
            columns=[
                "ligand_id_a",
                "ligand_id_b",
            ],
            errors="ignore",
        )


    compat.to_csv(
        destination_file,
        index=False,
    )


# ============================================================
# COPY DIRECTORY INTO COMPATIBILITY FORM
# ============================================================

def copy_directory_compatibly(
    source_dir: Path,
    destination_dir: Path,
):

    if destination_dir.exists():

        shutil.rmtree(
            destination_dir
        )


    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


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

            make_compatibility_csv(
                source_path,
                destination_path,
            )


        else:

            shutil.copy2(
                source_path,
                destination_path,
            )


# ============================================================
# STAGE MASTER-TABLE INPUTS
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
    # TOP-LEVEL TABLES
    # --------------------------------------------------------

    top_level_tables = [
        "minimization_summary.csv",
        "post_minimization_QC.csv",
        "post_minimization_contacts.csv",
        "post_minimization_contacts_enriched.csv",
        "residue_feature_preferences.csv",
    ]


    for filename in top_level_tables:

        source = (
            workspace.results_dir
            / filename
        )


        if source.exists():

            make_compatibility_csv(
                source,
                compat_results
                / filename,
            )


    # --------------------------------------------------------
    # STRAIN RESULTS
    # --------------------------------------------------------

    public_strain = (
        workspace.results_dir
        / STRAIN_DIRNAME
    )


    compat_strain = (
        compat_results
        / STRAIN_DIRNAME
    )


    copy_directory_compatibly(
        public_strain,
        compat_strain,
    )


    # The historical strain directory contains L#### folders
    # publicly. Rename them back to numeric directories only
    # inside compatibility staging.
    for ligand in records:

        public_name = (
            compat_strain
            / ligand.ligand_id
        )


        legacy_name = (
            compat_strain
            / f"{ligand.internal_ordinal:03d}"
        )


        if public_name.exists():

            public_name.rename(
                legacy_name
            )


    # --------------------------------------------------------
    # LANDSCAPE RESULTS
    #
    # Public:
    #     contact_landscape/
    #
    # Legacy engine expects:
    #     contact_landscape_v2/
    # --------------------------------------------------------

    public_landscape = (
        workspace.results_dir
        / PUBLIC_LANDSCAPE_DIRNAME
    )


    compat_landscape = (
        compat_results
        / LEGACY_LANDSCAPE_DIRNAME
    )


    copy_directory_compatibly(
        public_landscape,
        compat_landscape,
    )


    return (
        compat_results,
        n_ligands,
    )


# ============================================================
# NORMALIZE MASTER TABLE
# ============================================================

def normalize_master_table(
    legacy_master: Path,
    public_master: Path,
    workspace: Workspace,
):

    master = pd.read_csv(
        legacy_master
    )


    if (
        "rank"
        not in master.columns
    ):

        raise RuntimeError(
            "\nLegacy master table is missing rank column:\n"
            f"  {legacy_master}"
        )


    mapping = build_mapping(
        workspace
    )


    mapping_for_merge = (
        mapping.drop(
            columns=[
                "internal_ordinal"
            ],
            errors="ignore",
        )
    )


    enriched = master.merge(
        mapping_for_merge,
        on="rank",
        how="left",
        validate="one_to_one",
    )


    enriched = enriched.rename(
        columns={
            "rank":
                "internal_ordinal",
        }
    )


    # --------------------------------------------------------
    # REMOVE COMPATIBILITY PATH FIELDS WHEN THEY LEAK OUT
    # --------------------------------------------------------

    compatibility_path_columns = [
        "original_sdf",
        "ligand_sdf",
    ]


    enriched = enriched.drop(
        columns=[
            column
            for column
            in compatibility_path_columns
            if column
            in enriched.columns
        ],
        errors="ignore",
    )


    # --------------------------------------------------------
    # REPLACE PATHS WITH GENERALIZED PUBLIC PATHS
    # --------------------------------------------------------

    minimized_sdf_values = []
    complex_pdb_values = []


    for _, row in enriched.iterrows():

        ligand_id = (
            row[
                "ligand_id"
            ]
        )


        minimized_sdf_values.append(
            str(
                workspace.minimized_ligand_sdf(
                    ligand_id
                )
            )
        )


        complex_pdb_values.append(
            str(
                workspace.minimized_complex_pdb(
                    ligand_id
                )
            )
        )


    enriched[
        "minimized_sdf"
    ] = minimized_sdf_values


    enriched[
        "complex_pdb"
    ] = complex_pdb_values


    # --------------------------------------------------------
    # PUBLIC COLUMN ORDER
    # --------------------------------------------------------

    preferred_front = [
        "ligand_id",
        "compound_id",
        "pose_index",
        "title",
        "source_filename",
        "source_rank",
        "internal_ordinal",
    ]


    front = [
        column
        for column
        in preferred_front
        if column
        in enriched.columns
    ]


    remainder = [
        column
        for column
        in enriched.columns
        if column
        not in front
    ]


    enriched = enriched[
        front
        + remainder
    ]


    public_master.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    enriched.to_csv(
        public_master,
        index=False,
    )


# ============================================================
# PRINT GENERALIZED MASTER SUMMARY
# ============================================================

def print_master_summary(
    public_master: Path,
):

    df = pd.read_csv(
        public_master
    )


    print()
    print("=" * 120)
    print("GENERALIZED MASTER DESIGN TABLE")
    print("=" * 120)


    preferred = [
        "ligand_id",
        "QC_status",
        "strain_bound_fixed_kcal_mol",
        "strain_bound_relaxed_kcal_mol",
        "strain_interpretation",
        "anchor_residue_count",
        "typed_interaction_count",
        "design_priority_score",
        "design_flags",
    ]


    columns = [
        column
        for column
        in preferred
        if column
        in df.columns
    ]


    if columns:

        display = (
            df[
                columns
            ]
            .copy()
        )


        numeric_columns = (
            display.select_dtypes(
                include="number"
            ).columns
        )


        display[
            numeric_columns
        ] = (
            display[
                numeric_columns
            ]
            .round(
                3
            )
        )


        print(
            display.to_string(
                index=False
            )
        )


    else:

        print(
            f"Rows: {len(df)}"
        )


# ============================================================
# RUN MASTER TABLE
# ============================================================

def run_master(
    workspace: Workspace,
    dry_run: bool = False,
    keep_staging: bool = False,
):

    workspace.ensure_results_dir()


    if not LEGACY_MASTER.exists():

        raise RuntimeError(
            "\nValidated master-table engine is missing:\n"
            f"  {LEGACY_MASTER}"
        )


    validate_inputs(
        workspace
    )


    staging_root = (
        workspace.root
        / ".dockpost"
        / "native_master_staging"
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
                LEGACY_MASTER
            ),
            str(
                n_ligands
            ),
        ]


        print()
        print("=" * 80)
        print("DOCK-POSTPROCESS NATIVE MASTER TABLE")
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
            f"  {LEGACY_MASTER}"
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
                    / "master_design_table.csv"
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
        # RUN VALIDATED MASTER ENGINE
        # ====================================================

        subprocess.run(
            command,
            cwd=str(
                staging_root
            ),
            check=True,
        )


        legacy_master = (
            compat_results
            / "master_design_table.csv"
        )


        if not legacy_master.exists():

            raise RuntimeError(
                "\nMaster-table engine completed but expected "
                "output was not produced:\n"
                f"  {legacy_master}"
            )


        public_master = (
            workspace.results_dir
            / "master_design_table.csv"
        )


        normalize_master_table(
            legacy_master,
            public_master,
            workspace,
        )


        print_master_summary(
            public_master
        )


        print()
        print("=" * 80)
        print("MASTER DESIGN TABLE COMPLETE")
        print("=" * 80)

        print(
            "Wrote:"
        )

        print(
            f"  {public_master}"
        )

        print()
        print(
            "IMPORTANT: design_priority_score is a descriptive "
            "multi-objective heuristic, not an affinity prediction."
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
            "Build the generalized dock-postprocess master design table."
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
            "building the master table."
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


    run_master(
        workspace,
        dry_run=args.dry_run,
        keep_staging=args.keep_staging,
    )


if __name__ == "__main__":
    main()
