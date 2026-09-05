#!/usr/bin/env python3

"""
Generalized ligand-strain adapter for dock-postprocess.

Public inputs
-------------
analysis/
    ligand_manifest.csv
    ligands/
        L0001.sdf
        ...

    results/
        L0001/
            ligand_minimized.sdf
            complex_minimized.pdb
        ...

Public outputs
--------------
results/
    ligand_strain/
        L0001/
        L0002/
        ...
        strain_summary.csv
        strain_summary_enriched.csv
        *.png
        ...

The validated historical scientific engine still operates internally on
rank-numbered compatibility directories. This adapter stages those inputs
temporarily and converts all public identity back to stable L#### IDs.

The generalized dock-strain operation automatically performs both:
    1. ligand strain calculation
    2. strain postprocessing/classification

No separate public strain-post step is required.
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

LEGACY_STRAIN = (
    CORE_DIR
    / "ligand_strain.py"
)

LEGACY_STRAIN_POST = (
    CORE_DIR
    / "strain_postprocess.py"
)


sys.path.insert(
    0,
    str(
        THIS_FILE.parent.parent
    ),
)


from dockpost.workspace import Workspace  # noqa: E402


PUBLIC_STRAIN_DIRNAME = (
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
        })

    return pd.DataFrame(
        rows
    )


def ordinal_to_ligand_id(
    workspace: Workspace,
) -> dict[int, str]:

    return {
        int(
            ligand.internal_ordinal
        ):
        ligand.ligand_id

        for ligand
        in workspace.ligand_records()
    }


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    workspace: Workspace,
):

    missing = []


    for ligand in workspace.ligand_records():

        minimized_sdf = (
            workspace.minimized_ligand_sdf(
                ligand.ligand_id
            )
        )

        if not minimized_sdf.exists():

            missing.append(
                str(
                    minimized_sdf
                )
            )


    if missing:

        raise RuntimeError(
            "\nLigand-strain analysis requires a minimized SDF "
            "for every ligand in the workspace manifest.\n\n"
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


    for ligand in records:

        ordinal = (
            ligand.internal_ordinal
        )


        compat_dir = (
            compat_results
            / f"{ordinal:03d}"
        )

        compat_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ----------------------------------------------------
        # MINIMIZED BOUND LIGAND
        # ----------------------------------------------------

        compat_minimized_sdf = (
            compat_dir
            / (
                f"{ordinal:03d}_"
                f"{ligand.ligand_id}_"
                "ligand_minimized.sdf"
            )
        )


        shutil.copy2(
            workspace.minimized_ligand_sdf(
                ligand.ligand_id
            ),
            compat_minimized_sdf,
        )


        # ----------------------------------------------------
        # ORIGINAL POSE
        #
        # Stage this as well in case the validated strain engine
        # or future diagnostics reference the original geometry.
        # ----------------------------------------------------

        compat_original_sdf = (
            staging_root
            / (
                f"{ordinal:03d}__"
                f"{ligand.ligand_id}__"
                "original.sdf"
            )
        )


        shutil.copy2(
            ligand.canonical_sdf,
            compat_original_sdf,
        )


    # --------------------------------------------------------
    # MINIMIZATION SUMMARY
    # --------------------------------------------------------

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
# NORMALIZE RANKED CSV
# ============================================================

def normalize_ranked_csv(
    source_file: Path,
    destination_file: Path,
    mapping: pd.DataFrame,
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


    if "rank" not in df.columns:

        df.to_csv(
            destination_file,
            index=False,
        )

        return


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
# EXPORT STRAIN DIRECTORY
# ============================================================

def export_strain_results(
    legacy_strain_dir: Path,
    public_strain_dir: Path,
    workspace: Workspace,
):

    if public_strain_dir.exists():

        shutil.rmtree(
            public_strain_dir
        )


    public_strain_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    mapping = build_mapping(
        workspace
    )

    ligand_ids = ordinal_to_ligand_id(
        workspace
    )


    for source_path in sorted(
        legacy_strain_dir.rglob(
            "*"
        )
    ):

        relative = (
            source_path.relative_to(
                legacy_strain_dir
            )
        )


        parts = list(
            relative.parts
        )


        # ----------------------------------------------------
        # CONVERT TOP-LEVEL 001/002/003 DIRECTORIES TO L####
        # ----------------------------------------------------

        if len(parts) > 0:

            first = (
                parts[0]
            )


            if (
                first.isdigit()
                and
                int(first)
                in ligand_ids
            ):

                parts[
                    0
                ] = ligand_ids[
                    int(
                        first
                    )
                ]


        destination_relative = Path(
            *parts
        )


        destination_path = (
            public_strain_dir
            / destination_relative
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

            normalize_ranked_csv(
                source_path,
                destination_path,
                mapping,
            )


        else:

            shutil.copy2(
                source_path,
                destination_path,
            )


# ============================================================
# SUMMARIZE RESULTS
# ============================================================

def print_strain_summary(
    public_strain_dir: Path,
):

    enriched = (
        public_strain_dir
        / "strain_summary_enriched.csv"
    )


    raw = (
        public_strain_dir
        / "strain_summary.csv"
    )


    summary_file = (
        enriched
        if enriched.exists()
        else raw
    )


    print()
    print("=" * 100)
    print("GENERALIZED LIGAND STRAIN")
    print("=" * 100)


    if not summary_file.exists():

        print(
            "No strain summary file found."
        )

        return


    try:

        df = pd.read_csv(
            summary_file
        )

    except pd.errors.EmptyDataError:

        print(
            "Strain summary is empty."
        )

        return


    preferred = [
        "ligand_id",
        "internal_ordinal",
        "strain_bound_fixed_kcal_mol",
        "receptor_held_distortion_kcal_mol",
        "strain_bound_relaxed_kcal_mol",
        "closest_solution_RMSD_A",
        "conformational_strain_class",
        "bound_distortion_class",
        "strain_interpretation",
        "preorganization_score",
    ]


    display_columns = [
        column
        for column
        in preferred
        if column
        in df.columns
    ]


    if display_columns:

        print(
            df[
                display_columns
            ]
            .round(
                3
            )
            .to_string(
                index=False
            )
        )


    else:

        print(
            f"Rows: {len(df)}"
        )


# ============================================================
# RUN STRAIN ANALYSIS
# ============================================================

def run_strain(
    workspace: Workspace,
    conformers: int = 300,
    temperature: float = 298.15,
    seed: int = 2026,
    prune_rms: float = 0.5,
    minimize_tolerance: float = 0.01,
    max_iterations: int = 5000,
    dry_run: bool = False,
    keep_staging: bool = False,
):

    workspace.ensure_results_dir()


    if not LEGACY_STRAIN.exists():

        raise RuntimeError(
            "\nValidated ligand-strain engine is missing:\n"
            f"  {LEGACY_STRAIN}"
        )


    if not LEGACY_STRAIN_POST.exists():

        raise RuntimeError(
            "\nValidated strain-postprocessing engine is missing:\n"
            f"  {LEGACY_STRAIN_POST}"
        )


    validate_inputs(
        workspace
    )


    staging_root = (
        workspace.root
        / ".dockpost"
        / "native_strain_staging"
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


        strain_command = [
            sys.executable,
            str(
                LEGACY_STRAIN
            ),
            str(
                n_ligands
            ),
            "--conformers",
            str(
                conformers
            ),
            "--temperature",
            str(
                temperature
            ),
            "--seed",
            str(
                seed
            ),
            "--prune-rms",
            str(
                prune_rms
            ),
            "--minimize-tolerance",
            str(
                minimize_tolerance
            ),
            "--max-iterations",
            str(
                max_iterations
            ),
        ]


        post_command = [
            sys.executable,
            str(
                LEGACY_STRAIN_POST
            ),
            str(
                n_ligands
            ),
        ]


        print()
        print("=" * 80)
        print("DOCK-POSTPROCESS NATIVE LIGAND STRAIN")
        print("=" * 80)

        print(
            f"Workspace:  {workspace.root}"
        )

        print(
            f"Ligands:    {n_ligands}"
        )

        print(
            f"Conformers: {conformers}"
        )

        print(
            f"Results:    {workspace.results_dir}"
        )

        print()
        print(
            "Scientific engines:"
        )

        print(
            f"  {LEGACY_STRAIN}"
        )

        print(
            f"  {LEGACY_STRAIN_POST}"
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
                "DRY RUN - strain calculation:"
            )

            print(
                f"  cwd = {staging_root}"
            )

            print(
                "  "
                + " ".join(
                    strain_command
                )
            )


            print()
            print(
                "DRY RUN - automatic postprocessing:"
            )

            print(
                "  "
                + " ".join(
                    post_command
                )
            )

            return


        # ====================================================
        # RUN STRAIN ENGINE
        # ====================================================

        subprocess.run(
            strain_command,
            cwd=str(
                staging_root
            ),
            check=True,
        )


        # ====================================================
        # RUN POSTPROCESSING AUTOMATICALLY
        # ====================================================

        subprocess.run(
            post_command,
            cwd=str(
                staging_root
            ),
            check=True,
        )


        legacy_strain_dir = (
            compat_results
            / PUBLIC_STRAIN_DIRNAME
        )


        if not legacy_strain_dir.exists():

            raise RuntimeError(
                "\nStrain engine completed but expected output "
                "directory was not produced:\n"
                f"  {legacy_strain_dir}"
            )


        public_strain_dir = (
            workspace.results_dir
            / PUBLIC_STRAIN_DIRNAME
        )


        export_strain_results(
            legacy_strain_dir,
            public_strain_dir,
            workspace,
        )


        print_strain_summary(
            public_strain_dir
        )


        print()
        print("=" * 80)
        print("LIGAND STRAIN COMPLETE")
        print("=" * 80)

        print(
            "Wrote:"
        )

        print(
            f"  {public_strain_dir}"
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
            "Generalized isolated-ligand conformational strain analysis."
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
        "--conformers",
        type=int,
        default=300,
        help=(
            "Number of isolated ligand conformers to attempt. "
            "Default: 300."
        ),
    )


    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        help=(
            "Temperature in K used for energy weighting. "
            "Default: 298.15."
        ),
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Random seed for conformer generation. "
            "Default: 2026."
        ),
    )


    parser.add_argument(
        "--prune-rms",
        type=float,
        default=0.5,
        help=(
            "ETKDG conformer pruning RMS threshold in Angstrom. "
            "Default: 0.5."
        ),
    )


    parser.add_argument(
        "--minimize-tolerance",
        type=float,
        default=0.01,
        help=(
            "OpenMM minimization tolerance passed to the validated "
            "strain engine. Default: 0.01."
        ),
    )


    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5000,
        help=(
            "Maximum minimization iterations per conformer. "
            "Default: 5000."
        ),
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show staging and commands without running strain analysis."
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


    run_strain(
        workspace,
        conformers=args.conformers,
        temperature=args.temperature,
        seed=args.seed,
        prune_rms=args.prune_rms,
        minimize_tolerance=args.minimize_tolerance,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run,
        keep_staging=args.keep_staging,
    )


if __name__ == "__main__":
    main()
