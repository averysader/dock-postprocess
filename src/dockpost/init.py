#!/usr/bin/env python3

"""
dock-init

Create a generalized dock-postprocess workspace from a receptor PDB and
one or more ligand SDF files.

The scientific input-layer implementation lives in dockpost.manifest.
This module provides the public command-line interface only.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd

from dockpost.manifest import create_workspace


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a dock-postprocess analysis workspace from "
            "a receptor PDB and ligand pose SDF files."
        )
    )


    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input directory containing the receptor and ligand "
            "pose files."
        ),
    )


    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output analysis workspace directory."
        ),
    )


    parser.add_argument(
        "--receptor",
        default=None,
        help=(
            "Receptor PDB filename or path. If omitted, exactly "
            "one PDB must be discoverable in the input directory."
        ),
    )


    parser.add_argument(
        "--ligands",
        action="append",
        dest="ligand_patterns",
        default=None,
        help=(
            "Ligand SDF filename, glob pattern, or path. "
            "May be supplied multiple times. If omitted, ligand "
            "SDF files are discovered automatically."
        ),
    )


    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Search recursively for ligand SDF files."
        ),
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Import at most this many ligand records. Useful for "
            "testing or small validation runs."
        ),
    )


    args = parser.parse_args()


    input_dir = (
        Path(
            args.input
        )
        .expanduser()
        .resolve()
    )


    output_dir = (
        Path(
            args.output
        )
        .expanduser()
        .resolve()
    )


    if (
        args.limit is not None
        and
        args.limit <= 0
    ):

        parser.error(
            "--limit must be a positive integer."
        )


    print()
    print("=" * 80)
    print("DOCK-POSTPROCESS WORKSPACE INITIALIZATION")
    print("=" * 80)

    print(
        f"Input:     {input_dir}"
    )

    print(
        f"Workspace: {output_dir}"
    )


    if args.receptor:

        print(
            f"Receptor:  {args.receptor}"
        )

    else:

        print(
            "Receptor:  auto-discover"
        )


    if args.ligand_patterns:

        print(
            "Ligand inputs:"
        )

        for pattern in args.ligand_patterns:

            print(
                f"  {pattern}"
            )

    else:

        print(
            "Ligands:   auto-discover SDF files"
        )


    print(
        f"Recursive: {'yes' if args.recursive else 'no'}"
    )


    if args.limit is not None:

        print(
            f"Limit:     {args.limit}"
        )


    create_workspace(
        input_dir=input_dir,
        output_dir=output_dir,
        receptor=args.receptor,
        ligand_patterns=args.ligand_patterns,
        recursive=args.recursive,
        limit=args.limit,
    )


    manifest_file = (
        output_dir
        / "ligand_manifest.csv"
    )

    workspace_file = (
        output_dir
        / ".dockpost"
        / "workspace.json"
    )


    if not manifest_file.exists():

        raise RuntimeError(
            "Workspace creation completed without producing "
            f"the expected manifest:\n  {manifest_file}"
        )


    if not workspace_file.exists():

        raise RuntimeError(
            "Workspace creation completed without producing "
            f"the expected workspace metadata:\n  {workspace_file}"
        )


    manifest = pd.read_csv(
        manifest_file
    )


    with open(
        workspace_file
    ) as handle:

        workspace_info = json.load(
            handle
        )


    print()
    print("=" * 80)
    print("WORKSPACE CREATED")
    print("=" * 80)

    print(
        f"Ligand records: {len(manifest)}"
    )

    print(
        f"Receptor:       {output_dir / 'receptor.pdb'}"
    )

    print(
        f"Manifest:       {manifest_file}"
    )

    print(
        f"Workspace info: {workspace_file}"
    )


    if (
        "coordinate_frame_status"
        in manifest.columns
    ):

        counts = (
            manifest[
                "coordinate_frame_status"
            ]
            .fillna(
                "UNKNOWN"
            )
            .value_counts()
        )


        print()
        print(
            "Coordinate-frame QC:"
        )


        for status, count in counts.items():

            print(
                f"  {status}: {count}"
            )


    warning_count = 0


    if (
        "coordinate_frame_status"
        in manifest.columns
    ):

        warning_count = int(
            (
                manifest[
                    "coordinate_frame_status"
                ]
                != "PASS"
            )
            .sum()
        )


    print()
    print(
        "Next step:"
    )

    print(
        f"  dock-minimize --results {output_dir}"
    )


    if warning_count:

        print()
        print(
            "WARNING:"
        )

        print(
            f"  {warning_count} ligand record(s) failed the "
            "coordinate-frame proximity check."
        )

        print(
            "  Inspect those poses before interpreting downstream "
            "results."
        )


if __name__ == "__main__":

    main()
