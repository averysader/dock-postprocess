#!/usr/bin/env python3

"""
Generalized contact-feature enrichment adapter for dock-postprocess.

Public inputs
-------------
analysis/
    ligand_manifest.csv
    receptor.pdb

    results/
        L0001/
            complex_minimized.pdb
            ligand_minimized.sdf
        L0002/
            ...
        post_minimization_QC.csv
        post_minimization_contacts.csv

The validated enrichment engine still expects the historical compatibility
layout:

    topN_openmm_minimized/
        001/
        002/
        ...
        post_minimization_contacts.csv

This module constructs that layout temporarily, runs the validated
scientific engine, then converts ligand-indexed results back to the
generalized public identity model:

    ligand_id
    internal_ordinal

Aggregated residue-level tables that do not contain ligand ranks are copied
without modification.
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

LEGACY_ENRICH = (
    CORE_DIR
    / "enrich_contact_features.py"
)


sys.path.insert(
    0,
    str(
        THIS_FILE.parent.parent
    ),
)


from dockpost.workspace import Workspace  # noqa: E402


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


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    workspace: Workspace,
):

    missing = []


    contacts_file = (
        workspace.results_dir
        / "post_minimization_contacts.csv"
    )

    qc_file = (
        workspace.results_dir
        / "post_minimization_QC.csv"
    )


    if not contacts_file.exists():

        missing.append(
            str(
                contacts_file
            )
        )


    if not qc_file.exists():

        missing.append(
            str(
                qc_file
            )
        )


    for ligand in workspace.ligand_records():

        minimized_sdf = (
            workspace.minimized_ligand_sdf(
                ligand.ligand_id
            )
        )

        complex_pdb = (
            workspace.minimized_complex_pdb(
                ligand.ligand_id
            )
        )


        if not minimized_sdf.exists():

            missing.append(
                str(
                    minimized_sdf
                )
            )


        if not complex_pdb.exists():

            missing.append(
                str(
                    complex_pdb
                )
            )


    if missing:

        raise RuntimeError(
            "\nContact enrichment requires completed native QC "
            "and minimized ligand results.\n\n"
            "Missing files:\n  "
            + "\n  ".join(
                missing
            )
        )


# ============================================================
# CONVERT PUBLIC TABLE TO COMPATIBILITY FORM
# ============================================================

def make_compatibility_table(
    source_file: Path,
    destination_file: Path,
):

    df = pd.read_csv(
        source_file
    )


    if (
        "internal_ordinal"
        not in df.columns
    ):

        raise RuntimeError(
            "\nGeneralized table is missing internal_ordinal:\n"
            f"  {source_file}"
        )


    compat = (
        df.copy()
    )


    compat[
        "rank"
    ] = (
        compat[
            "internal_ordinal"
        ]
    )


    # Legacy tools do not need generalized metadata.
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
            if column in compat.columns
        ],
        errors="ignore",
    )


    # Put rank first for readability and maximum compatibility.
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
    # STAGE MINIMIZED STRUCTURES
    # --------------------------------------------------------

    for ligand in records:

        rank = (
            ligand.internal_ordinal
        )


        compat_dir = (
            compat_results
            / f"{rank:03d}"
        )

        compat_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        source_sdf = (
            workspace.minimized_ligand_sdf(
                ligand.ligand_id
            )
        )

        source_pdb = (
            workspace.minimized_complex_pdb(
                ligand.ligand_id
            )
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
            source_sdf,
            compat_sdf,
        )


        shutil.copy2(
            source_pdb,
            compat_pdb,
        )


    # --------------------------------------------------------
    # STAGE QC CONTACT TABLE
    # --------------------------------------------------------

    public_contacts = (
        workspace.results_dir
        / "post_minimization_contacts.csv"
    )

    compat_contacts = (
        compat_results
        / "post_minimization_contacts.csv"
    )


    make_compatibility_table(
        public_contacts,
        compat_contacts,
    )


    # --------------------------------------------------------
    # STAGE QC SUMMARY AS WELL
    #
    # It may not be required by the current enrichment engine,
    # but retaining it makes the compatibility workspace
    # self-consistent.
    # --------------------------------------------------------

    public_qc = (
        workspace.results_dir
        / "post_minimization_QC.csv"
    )

    compat_qc = (
        compat_results
        / "post_minimization_QC.csv"
    )


    make_compatibility_table(
        public_qc,
        compat_qc,
    )


    return (
        compat_results,
        n_ligands,
    )


# ============================================================
# NORMALIZE LIGAND-INDEXED OUTPUT TABLE
# ============================================================

def normalize_ranked_table(
    source_file: Path,
    destination_file: Path,
    mapping: pd.DataFrame,
):

    try:

        df = pd.read_csv(
            source_file
        )

    except pd.errors.EmptyDataError:

        df = pd.DataFrame()


    # --------------------------------------------------------
    # COMPLETELY EMPTY OUTPUT
    # --------------------------------------------------------

    if len(df.columns) == 0:

        pd.DataFrame(
            columns=[
                "ligand_id",
                "internal_ordinal",
                "compound_id",
                "pose_index",
                "title",
                "source_filename",
                "source_rank",
            ]
        ).to_csv(
            destination_file,
            index=False,
        )

        return


    if "rank" not in df.columns:

        raise RuntimeError(
            "\nExpected compatibility rank column in "
            "ligand-indexed enrichment output:\n"
            f"  {source_file}"
        )


    mapping_for_merge = (
        mapping.drop(
            columns=[
                "internal_ordinal"
            ],
            errors="ignore",
        )
    )


    merged = df.merge(
        mapping_for_merge,
        on="rank",
        how="left",
        validate="many_to_one",
    )


    merged = merged.rename(
        columns={
            "rank":
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


    merged = merged[
        front
        + remainder
    ]


    merged.to_csv(
        destination_file,
        index=False,
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_enrichment_summary(
    enriched_file: Path,
):

    try:

        df = pd.read_csv(
            enriched_file
        )

    except pd.errors.EmptyDataError:

        print()
        print(
            "Enriched contact table is empty."
        )

        return


    print()
    print("=" * 100)
    print("GENERALIZED CONTACT FEATURE ENRICHMENT")
    print("=" * 100)


    if len(df) == 0:

        print(
            "No contact rows were produced."
        )

        return


    print(
        f"Contact rows: {len(df)}"
    )


    if "ligand_id" in df.columns:

        print(
            "Ligands represented: "
            f"{df['ligand_id'].nunique()}"
        )


    preferred = [
        "ligand_id",
        "internal_ordinal",
        "protein_chain",
        "protein_resname",
        "protein_resid",
        "ligand_feature",
        "protein_feature",
    ]


    display_columns = [
        column
        for column
        in preferred
        if column
        in df.columns
    ]


    if display_columns:

        print()
        print(
            df[
                display_columns
            ]
            .head(
                15
            )
            .to_string(
                index=False
            )
        )


# ============================================================
# RUN ENRICHMENT
# ============================================================

def run_enrichment(
    workspace: Workspace,
    dry_run: bool = False,
    keep_staging: bool = False,
):

    workspace.ensure_results_dir()


    if not LEGACY_ENRICH.exists():

        raise RuntimeError(
            "\nValidated contact-enrichment engine is missing:\n"
            f"  {LEGACY_ENRICH}"
        )


    validate_inputs(
        workspace
    )


    staging_root = (
        workspace.root
        / ".dockpost"
        / "native_enrich_staging"
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
                LEGACY_ENRICH
            ),
            str(
                n_ligands
            ),
        ]


        print()
        print("=" * 80)
        print("DOCK-POSTPROCESS NATIVE CONTACT ENRICHMENT")
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
            f"  {LEGACY_ENRICH}"
        )


        if dry_run:

            print()
            print(
                "Staged minimized ligands:"
            )


            for ligand in workspace.ligand_records():

                print(
                    f"  "
                    f"{ligand.internal_ordinal:03d} "
                    f"-> {ligand.ligand_id}"
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
        # RUN VALIDATED SCIENTIFIC ENGINE
        # ====================================================

        subprocess.run(
            command,
            cwd=str(
                staging_root
            ),
            check=True,
        )


        # ====================================================
        # EXPECTED OUTPUTS
        # ====================================================

        enriched_source = (
            compat_results
            / "post_minimization_contacts_enriched.csv"
        )

        preferences_source = (
            compat_results
            / "residue_feature_preferences.csv"
        )


        if not enriched_source.exists():

            raise RuntimeError(
                "\nContact-enrichment engine did not produce:\n"
                f"  {enriched_source}"
            )


        if not preferences_source.exists():

            raise RuntimeError(
                "\nContact-enrichment engine did not produce:\n"
                f"  {preferences_source}"
            )


        # ====================================================
        # PUBLIC OUTPUTS
        # ====================================================

        enriched_output = (
            workspace.results_dir
            / "post_minimization_contacts_enriched.csv"
        )

        preferences_output = (
            workspace.results_dir
            / "residue_feature_preferences.csv"
        )


        mapping = build_mapping(
            workspace
        )


        normalize_ranked_table(
            enriched_source,
            enriched_output,
            mapping,
        )


        # This is an aggregate residue-level table rather than a
        # per-ligand rank table, so preserve its scientific contents.
        shutil.copy2(
            preferences_source,
            preferences_output,
        )


        print_enrichment_summary(
            enriched_output
        )


        print()
        print("=" * 80)
        print("CONTACT ENRICHMENT COMPLETE")
        print("=" * 80)

        print(
            "Wrote:"
        )

        print(
            f"  {enriched_output}"
        )

        print(
            f"  {preferences_output}"
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
            "Generalized protein-ligand contact feature enrichment."
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
            "Show the compatibility staging and execution "
            "plan without running enrichment."
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


    run_enrichment(
        workspace,
        dry_run=args.dry_run,
        keep_staging=args.keep_staging,
    )


if __name__ == "__main__":
    main()
