#!/usr/bin/env python3

"""
Generalized post-minimization QC for dock-postprocess.

The validated historical QC engine is currently retained for the core
geometric QC calculations:

    - ligand heavy-atom displacement
    - centroid displacement
    - protein-ligand contact distances
    - van der Waals clash analysis
    - contacting residues
    - PASS/FAIL classification

Historical target-specific Y220 metrics are removed from all public output.

Optional receptor-agnostic focus-residue analysis is implemented natively:

    --focus-residue B:220
    --focus-residue A:85

The option may be repeated. Focus-residue results are written to:

    results/focus_residue_QC.csv

This works with one-chain or multichain receptors.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from rdkit import Chem


THIS_FILE = Path(__file__).resolve()

PACKAGE_ROOT = (
    THIS_FILE.parents[2]
)

CORE_DIR = (
    PACKAGE_ROOT
    / "core"
)

LEGACY_QC = (
    CORE_DIR
    / "qc_top_minimized.py"
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
# VALIDATE NATIVE INPUTS
# ============================================================

def validate_native_results(
    workspace: Workspace,
):

    missing = []

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
            "\nQC requires original and minimized structures "
            "for every ligand in the workspace manifest.\n\n"
            "Missing files:\n  "
            + "\n  ".join(
                missing
            )
        )


# ============================================================
# STAGE LEGACY QC INPUT
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


        # ----------------------------------------------------
        # ORIGINAL POSE
        # ----------------------------------------------------

        compat_original = (
            staging_root
            / (
                f"{ordinal:03d}__"
                f"{ligand.ligand_id}__"
                "original.sdf"
            )
        )

        shutil.copy2(
            ligand.canonical_sdf,
            compat_original,
        )


        # ----------------------------------------------------
        # MINIMIZED STRUCTURES
        # ----------------------------------------------------

        compat_dir = (
            compat_results
            / f"{ordinal:03d}"
        )

        compat_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        compat_sdf = (
            compat_dir
            / (
                f"{ordinal:03d}_"
                f"{ligand.ligand_id}_"
                "ligand_minimized.sdf"
            )
        )

        compat_pdb = (
            compat_dir
            / (
                f"{ordinal:03d}_"
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


    public_summary = (
        workspace.minimization_summary()
    )

    if public_summary.exists():

        summary = pd.read_csv(
            public_summary
        )

        compat_summary = (
            summary.copy()
        )

        if (
            "internal_ordinal"
            in compat_summary.columns
        ):

            compat_summary[
                "rank"
            ] = (
                compat_summary[
                    "internal_ordinal"
                ]
            )

        compat_summary.to_csv(
            compat_results
            / "minimization_summary.csv",
            index=False,
        )


    return (
        compat_results,
        n_ligands,
    )


# ============================================================
# NORMALIZE LEGACY TABLE
# ============================================================

def normalize_table(
    source_file: Path,
    destination_file: Path,
    mapping: pd.DataFrame,
    allow_empty: bool = False,
    remove_target_specific: bool = False,
):

    try:

        df = pd.read_csv(
            source_file
        )

    except pd.errors.EmptyDataError:

        if not allow_empty:

            raise

        df = pd.DataFrame()


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
            "\nExpected compatibility rank column in:\n"
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


    # --------------------------------------------------------
    # REMOVE HISTORICAL TARGET-SPECIFIC QC
    # --------------------------------------------------------

    if remove_target_specific:

        target_specific_columns = [
            column
            for column
            in merged.columns
            if column.startswith(
                "Y220_"
            )
        ]

        merged = merged.drop(
            columns=target_specific_columns,
            errors="ignore",
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
# FOCUS-RESIDUE SPECIFICATION
# ============================================================

def parse_focus_residue(
    specification: str,
) -> tuple[str, str]:

    text = (
        str(
            specification
        )
        .strip()
    )


    if ":" not in text:

        raise argparse.ArgumentTypeError(
            "Focus residue must have format CHAIN:RESID, "
            "for example B:220."
        )


    chain, resid = (
        text.split(
            ":",
            1,
        )
    )


    chain = chain.strip()
    resid = resid.strip()


    if not chain:

        raise argparse.ArgumentTypeError(
            "Focus-residue chain cannot be empty."
        )


    if not resid:

        raise argparse.ArgumentTypeError(
            "Focus-residue residue ID cannot be empty."
        )


    return (
        chain,
        resid,
    )


# ============================================================
# PDB HEAVY-ATOM PARSING
# ============================================================

def pdb_element(
    line: str,
) -> str:

    element = ""

    if len(line) >= 78:

        element = (
            line[
                76:78
            ]
            .strip()
            .upper()
        )


    if element:

        return element


    atom_name = (
        line[
            12:16
        ]
        .strip()
    )


    letters = [
        character
        for character
        in atom_name
        if character.isalpha()
    ]


    if not letters:

        return ""


    # For ordinary protein PDB atom naming, the first alphabetic
    # character is sufficient for distinguishing hydrogen from
    # heavy atoms when the element field is absent.
    return (
        letters[0]
        .upper()
    )


def load_focus_residue_atoms(
    pdb_file: Path,
    chain_requested: str,
    resid_requested: str,
) -> list[dict]:

    atoms = []


    with open(
        pdb_file
    ) as handle:

        for line in handle:

            if not (
                line.startswith(
                    "ATOM"
                )
                or
                line.startswith(
                    "HETATM"
                )
            ):

                continue


            chain = (
                line[
                    21:22
                ]
                .strip()
            )


            resseq = (
                line[
                    22:26
                ]
                .strip()
            )


            icode = (
                line[
                    26:27
                ]
                .strip()
            )


            residue_id = (
                resseq
                + icode
            )


            if (
                chain
                != chain_requested
            ):

                continue


            if (
                residue_id
                != resid_requested
            ):

                continue


            element = pdb_element(
                line
            )


            if element in {
                "H",
                "D",
            }:

                continue


            try:

                x = float(
                    line[
                        30:38
                    ]
                )

                y = float(
                    line[
                        38:46
                    ]
                )

                z = float(
                    line[
                        46:54
                    ]
                )

            except ValueError:

                continue


            atoms.append({
                "atom_name":
                    line[
                        12:16
                    ]
                    .strip(),

                "resname":
                    line[
                        17:20
                    ]
                    .strip(),

                "element":
                    element,

                "x":
                    x,

                "y":
                    y,

                "z":
                    z,
            })


    return atoms


# ============================================================
# LIGAND HEAVY ATOMS
# ============================================================

def load_ligand_heavy_atoms(
    sdf_file: Path,
) -> list[dict]:

    supplier = Chem.SDMolSupplier(
        str(
            sdf_file
        ),
        removeHs=False,
    )


    mol = next(
        (
            molecule
            for molecule
            in supplier
            if molecule is not None
        ),
        None,
    )


    if mol is None:

        raise RuntimeError(
            "\nCould not read minimized ligand SDF:\n"
            f"  {sdf_file}"
        )


    if mol.GetNumConformers() == 0:

        raise RuntimeError(
            "\nLigand has no 3D conformer:\n"
            f"  {sdf_file}"
        )


    conformer = (
        mol.GetConformer()
    )


    atoms = []


    for atom in mol.GetAtoms():

        if (
            atom.GetAtomicNum()
            <= 1
        ):

            continue


        index = (
            atom.GetIdx()
        )

        position = (
            conformer.GetAtomPosition(
                index
            )
        )


        atoms.append({
            "rdkit_index":
                index,

            "sdf_serial":
                index + 1,

            "element":
                atom.GetSymbol(),

            "x":
                float(
                    position.x
                ),

            "y":
                float(
                    position.y
                ),

            "z":
                float(
                    position.z
                ),
        })


    return atoms


# ============================================================
# FOCUS-RESIDUE GEOMETRY
# ============================================================

def analyze_focus_residue(
    workspace: Workspace,
    ligand,
    chain: str,
    resid: str,
    contact_cutoff_A: float = 4.0,
) -> dict:

    complex_pdb = (
        workspace.minimized_complex_pdb(
            ligand.ligand_id
        )
    )

    ligand_sdf = (
        workspace.minimized_ligand_sdf(
            ligand.ligand_id
        )
    )


    residue_atoms = load_focus_residue_atoms(
        complex_pdb,
        chain,
        resid,
    )


    base = {
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

        "focus_chain":
            chain,

        "focus_resid":
            resid,

        "focus_residue":
            f"{chain}:{resid}",
    }


    if not residue_atoms:

        base.update({
            "focus_resname":
                None,

            "focus_residue_status":
                "MISSING",

            "nearest_distance_A":
                np.nan,

            "nearest_protein_atom":
                None,

            "nearest_ligand_atom_index_0based":
                np.nan,

            "nearest_ligand_atom_serial_1based":
                np.nan,

            "nearest_ligand_element":
                None,

            "contact_atom_pairs_le_4A":
                0,

            "focus_heavy_atoms_contacting_le_4A":
                0,

            "ligand_heavy_atoms_contacting_le_4A":
                0,
        })

        return base


    ligand_atoms = load_ligand_heavy_atoms(
        ligand_sdf
    )


    if not ligand_atoms:

        raise RuntimeError(
            f"No ligand heavy atoms found for {ligand.ligand_id}."
        )


    residue_xyz = np.array(
        [
            [
                atom[
                    "x"
                ],
                atom[
                    "y"
                ],
                atom[
                    "z"
                ],
            ]
            for atom
            in residue_atoms
        ],
        dtype=float,
    )


    ligand_xyz = np.array(
        [
            [
                atom[
                    "x"
                ],
                atom[
                    "y"
                ],
                atom[
                    "z"
                ],
            ]
            for atom
            in ligand_atoms
        ],
        dtype=float,
    )


    delta = (
        residue_xyz[
            :,
            None,
            :
        ]
        -
        ligand_xyz[
            None,
            :,
            :
        ]
    )


    distances = np.sqrt(
        np.sum(
            delta
            * delta,
            axis=2,
        )
    )


    flat_index = int(
        np.argmin(
            distances
        )
    )


    residue_index, ligand_index = (
        np.unravel_index(
            flat_index,
            distances.shape,
        )
    )


    nearest_distance = float(
        distances[
            residue_index,
            ligand_index
        ]
    )


    contact_mask = (
        distances
        <= contact_cutoff_A
    )


    contact_pair_count = int(
        np.sum(
            contact_mask
        )
    )


    residue_atoms_contacting = int(
        np.sum(
            np.any(
                contact_mask,
                axis=1,
            )
        )
    )


    ligand_atoms_contacting = int(
        np.sum(
            np.any(
                contact_mask,
                axis=0,
            )
        )
    )


    nearest_residue_atom = (
        residue_atoms[
            residue_index
        ]
    )

    nearest_ligand_atom = (
        ligand_atoms[
            ligand_index
        ]
    )


    base.update({
        "focus_resname":
            nearest_residue_atom[
                "resname"
            ],

        "focus_residue_status":
            "FOUND",

        "nearest_distance_A":
            nearest_distance,

        "nearest_protein_atom":
            nearest_residue_atom[
                "atom_name"
            ],

        "nearest_ligand_atom_index_0based":
            nearest_ligand_atom[
                "rdkit_index"
            ],

        "nearest_ligand_atom_serial_1based":
            nearest_ligand_atom[
                "sdf_serial"
            ],

        "nearest_ligand_element":
            nearest_ligand_atom[
                "element"
            ],

        "contact_atom_pairs_le_4A":
            contact_pair_count,

        "focus_heavy_atoms_contacting_le_4A":
            residue_atoms_contacting,

        "ligand_heavy_atoms_contacting_le_4A":
            ligand_atoms_contacting,
    })


    return base


# ============================================================
# FOCUS-RESIDUE TABLE
# ============================================================

def run_focus_residue_analysis(
    workspace: Workspace,
    focus_residues: list[tuple[str, str]],
):

    output_file = (
        workspace.results_dir
        / "focus_residue_QC.csv"
    )


    # No focus residues requested. Remove stale previous file.
    if not focus_residues:

        if output_file.exists():

            output_file.unlink()

        return None


    rows = []


    for ligand in workspace.ligand_records():

        for (
            chain,
            resid,
        ) in focus_residues:

            row = analyze_focus_residue(
                workspace,
                ligand,
                chain,
                resid,
            )

            rows.append(
                row
            )


            if (
                row[
                    "focus_residue_status"
                ]
                == "MISSING"
            ):

                print()
                print(
                    "WARNING:"
                )

                print(
                    f"  Focus residue {chain}:{resid} "
                    f"was not found for {ligand.ligand_id}."
                )


    df = pd.DataFrame(
        rows
    )


    df.to_csv(
        output_file,
        index=False,
    )


    print()
    print("=" * 100)
    print("FOCUS-RESIDUE QC")
    print("=" * 100)


    display_columns = [
        "ligand_id",
        "focus_residue",
        "focus_resname",
        "focus_residue_status",
        "nearest_distance_A",
        "nearest_protein_atom",
        "nearest_ligand_atom_serial_1based",
        "contact_atom_pairs_le_4A",
    ]


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


    print()
    print(
        f"Wrote:\n  {output_file}"
    )


    return output_file


# ============================================================
# PUBLIC QC SUMMARY
# ============================================================

def print_public_qc_summary(
    qc_file: Path,
):

    qc = pd.read_csv(
        qc_file
    )


    preferred = [
        "ligand_id",
        "internal_ordinal",
        "QC_status",
        "QC_reasons",
        "heavy_atom_RMS_displacement_A",
        "centroid_shift_A",
        "minimum_protein_ligand_distance_A",
        "protein_ligand_contacts_le_4A",
        "vdw_clashes_ge_0.4A",
        "severe_clashes_ge_0.8A",
    ]


    columns = [
        column
        for column
        in preferred
        if column
        in qc.columns
    ]


    print()
    print("=" * 100)
    print("GENERALIZED POST-MINIMIZATION QC")
    print("=" * 100)


    print(
        qc[
            columns
        ]
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# RUN QC
# ============================================================

def run_qc(
    workspace: Workspace,
    focus_residues: list[tuple[str, str]] | None = None,
    dry_run: bool = False,
    keep_staging: bool = False,
):

    workspace.ensure_results_dir()


    if focus_residues is None:

        focus_residues = []


    if not LEGACY_QC.exists():

        raise RuntimeError(
            "\nValidated QC engine is missing:\n"
            f"  {LEGACY_QC}"
        )


    validate_native_results(
        workspace
    )


    staging_root = (
        workspace.root
        / ".dockpost"
        / "native_qc_staging"
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
                LEGACY_QC
            ),
            str(
                n_ligands
            ),
        ]


        print()
        print("=" * 80)
        print("DOCK-POSTPROCESS GENERALIZED QC")
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


        if focus_residues:

            print()
            print(
                "Focus residues:"
            )

            for (
                chain,
                resid,
            ) in focus_residues:

                print(
                    f"  {chain}:{resid}"
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
        # VALIDATED GENERIC QC ENGINE
        # ====================================================

        subprocess.run(
            command,
            cwd=str(
                staging_root
            ),
            check=True,
        )


        mapping = build_mapping(
            workspace
        )


        qc_source = (
            compat_results
            / "post_minimization_QC.csv"
        )

        contacts_source = (
            compat_results
            / "post_minimization_contacts.csv"
        )


        if not qc_source.exists():

            raise RuntimeError(
                "\nQC engine did not produce:\n"
                f"  {qc_source}"
            )


        if not contacts_source.exists():

            raise RuntimeError(
                "\nQC engine did not produce:\n"
                f"  {contacts_source}"
            )


        qc_output = (
            workspace.results_dir
            / "post_minimization_QC.csv"
        )

        contacts_output = (
            workspace.results_dir
            / "post_minimization_contacts.csv"
        )


        # Public QC deliberately strips Y220-specific fields.
        normalize_table(
            qc_source,
            qc_output,
            mapping,
            allow_empty=False,
            remove_target_specific=True,
        )


        normalize_table(
            contacts_source,
            contacts_output,
            mapping,
            allow_empty=True,
            remove_target_specific=False,
        )


        print_public_qc_summary(
            qc_output
        )


        run_focus_residue_analysis(
            workspace,
            focus_residues,
        )


        print()
        print("=" * 80)
        print("QC COMPLETE")
        print("=" * 80)

        print(
            "Wrote:"
        )

        print(
            f"  {qc_output}"
        )

        print(
            f"  {contacts_output}"
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
            "Generalized minimized-pose QC with optional "
            "focus-residue analysis."
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
        "--focus-residue",
        action="append",
        default=[],
        metavar="CHAIN:RESID",
        help=(
            "Optional receptor residue for focused geometric QC. "
            "May be supplied multiple times. "
            "Example: --focus-residue B:220"
        ),
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show execution plan without running QC."
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


    focus_residues = [
        parse_focus_residue(
            specification
        )
        for specification
        in args.focus_residue
    ]


    workspace = Workspace.from_path(
        Path(
            args.results
        )
    )


    run_qc(
        workspace,
        focus_residues=focus_residues,
        dry_run=args.dry_run,
        keep_staging=args.keep_staging,
    )


if __name__ == "__main__":
    main()
