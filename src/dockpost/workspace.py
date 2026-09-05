#!/usr/bin/env python3

"""
Workspace utilities for dock-postprocess.

The public workspace model is independent of:
    - ranked filenames
    - topN directory names
    - one-SDF-per-ligand assumptions
    - project-specific paths

Canonical generalized layout
----------------------------

analysis/
    ligand_manifest.csv
    receptor.pdb
    ligands/
        L0001.sdf
        L0002.sdf
        ...
    results/
        L0001/
        L0002/
        ...
    .dockpost/
        workspace.json

The manifest is the authoritative mapping between source data and
internal ligand-pose identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass(frozen=True)
class LigandRecord:
    ligand_id: str
    internal_ordinal: int
    compound_id: str
    pose_index: int
    title: str
    source_file: Path
    source_filename: str
    source_rank: float | None
    canonical_sdf: Path


class Workspace:
    """
    Generalized dock-postprocess workspace.
    """

    def __init__(
        self,
        root: Path,
    ):

        self.root = (
            Path(root)
            .expanduser()
            .resolve()
        )

        self.metadata_file = (
            self.root
            / ".dockpost"
            / "workspace.json"
        )

        self.manifest_file = (
            self.root
            / "ligand_manifest.csv"
        )

        self.receptor = (
            self.root
            / "receptor.pdb"
        )

        self.ligands_dir = (
            self.root
            / "ligands"
        )

        self.results_dir = (
            self.root
            / "results"
        )


        if not self.metadata_file.exists():

            raise RuntimeError(
                "\nNot a dock-postprocess workspace:\n"
                f"  {self.root}\n\n"
                "Missing:\n"
                f"  {self.metadata_file}"
            )


        if not self.manifest_file.exists():

            raise RuntimeError(
                "\nWorkspace is missing ligand manifest:\n"
                f"  {self.manifest_file}"
            )


        if not self.receptor.exists():

            raise RuntimeError(
                "\nWorkspace is missing canonical receptor:\n"
                f"  {self.receptor}"
            )


        with open(
            self.metadata_file
        ) as handle:

            self.metadata = json.load(
                handle
            )


        self.manifest = pd.read_csv(
            self.manifest_file
        )


        if len(
            self.manifest
        ) == 0:

            raise RuntimeError(
                "Ligand manifest contains no ligand poses."
            )


        required = {
            "ordinal",
            "ligand_id",
            "compound_id",
            "pose_index",
            "source_file",
            "source_filename",
            "title",
            "canonical_sdf",
        }


        missing = (
            required
            - set(
                self.manifest.columns
            )
        )


        if missing:

            raise RuntimeError(
                "Manifest missing required columns:\n  "
                + "\n  ".join(
                    sorted(
                        missing
                    )
                )
            )


    @classmethod
    def from_path(
        cls,
        path: Path,
    ) -> "Workspace":

        path = (
            Path(path)
            .expanduser()
            .resolve()
        )


        # Workspace root.
        if (
            path
            / ".dockpost"
            / "workspace.json"
        ).exists():

            return cls(
                path
            )


        # Public results directory.
        if (
            path.name
            == "results"
            and
            (
                path.parent
                / ".dockpost"
                / "workspace.json"
            ).exists()
        ):

            return cls(
                path.parent
            )


        raise RuntimeError(
            "\nCould not resolve dock-postprocess workspace from:\n"
            f"  {path}\n\n"
            "Pass either the analysis workspace or its results/ directory."
        )


    def ligand_records(
        self,
    ) -> list[LigandRecord]:

        records = []


        for _, row in (
            self.manifest
            .sort_values(
                "ordinal"
            )
            .iterrows()
        ):

            source_rank = (
                None
                if pd.isna(
                    row.get(
                        "source_rank"
                    )
                )
                else float(
                    row[
                        "source_rank"
                    ]
                )
            )


            records.append(
                LigandRecord(
                    ligand_id=str(
                        row[
                            "ligand_id"
                        ]
                    ),

                    internal_ordinal=int(
                        row[
                            "ordinal"
                        ]
                    ),

                    compound_id=str(
                        row[
                            "compound_id"
                        ]
                    ),

                    pose_index=int(
                        row[
                            "pose_index"
                        ]
                    ),

                    title=str(
                        row[
                            "title"
                        ]
                    ),

                    source_file=Path(
                        row[
                            "source_file"
                        ]
                    ),

                    source_filename=str(
                        row[
                            "source_filename"
                        ]
                    ),

                    source_rank=source_rank,

                    canonical_sdf=Path(
                        row[
                            "canonical_sdf"
                        ]
                    ),
                )
            )


        return records


    def ligand(
        self,
        ligand_id: str,
    ) -> LigandRecord:

        ligand_id = (
            str(
                ligand_id
            )
            .upper()
        )


        for record in self.ligand_records():

            if (
                record.ligand_id.upper()
                == ligand_id
            ):

                return record


        raise KeyError(
            f"Unknown ligand ID: {ligand_id}"
        )


    def result_dir(
        self,
        ligand_id: str,
    ) -> Path:

        return (
            self.results_dir
            / ligand_id
        )


    def minimized_ligand_sdf(
        self,
        ligand_id: str,
    ) -> Path:

        return (
            self.result_dir(
                ligand_id
            )
            / "ligand_minimized.sdf"
        )


    def minimized_complex_pdb(
        self,
        ligand_id: str,
    ) -> Path:

        return (
            self.result_dir(
                ligand_id
            )
            / "complex_minimized.pdb"
        )


    def minimization_summary(
        self,
    ) -> Path:

        return (
            self.results_dir
            / "minimization_summary.csv"
        )


    def ensure_results_dir(
        self,
    ):

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


    def describe(
        self,
    ):

        print()
        print("=" * 80)
        print("DOCK-POSTPROCESS WORKSPACE")
        print("=" * 80)

        print(
            f"Workspace: {self.root}"
        )

        print(
            f"Receptor:  {self.receptor}"
        )

        print(
            f"Ligands:   {len(self.manifest)}"
        )

        print(
            f"Results:   {self.results_dir}"
        )

