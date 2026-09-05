#!/usr/bin/env python3

"""
Input discovery and workspace normalization for dock-postprocess.

User-facing assumptions
-----------------------
A project consists of:

    one receptor PDB
    one or more docked ligand poses in SDF format

Ligand poses must already be in the receptor Cartesian coordinate frame.

No filename ranking convention is required.

SDF files may contain one or many records. Each record becomes one
independent ligand-pose entry with an internal ligand ID:

    L0001
    L0002
    ...

The compatibility staging area uses temporary numeric prefixes only because
the validated legacy scientific core currently expects them.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import re
import shutil

import numpy as np

from rdkit import Chem


RANK_RE = re.compile(
    r"^(\d{1,8})(?:[_\-.]|$)"
)


def safe_name(
    value: str,
    fallback: str = "ligand",
) -> str:

    value = (
        value
        .strip()
        .replace(" ", "_")
    )

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    value = value.strip(
        "._-"
    )

    return value or fallback


def infer_source_rank(
    filename: str,
):

    match = RANK_RE.match(
        Path(filename).name
    )

    if match is None:
        return None

    try:
        return int(
            match.group(1)
        )

    except ValueError:
        return None


def discover_receptor(
    input_dir: Path,
    receptor: str | None,
) -> Path:

    if receptor is not None:

        receptor_path = (
            Path(receptor)
            .expanduser()
            .resolve()
        )

        if not receptor_path.exists():

            raise RuntimeError(
                f"Receptor does not exist:\n"
                f"  {receptor_path}"
            )

        if receptor_path.suffix.lower() != ".pdb":

            raise RuntimeError(
                "Receptor must be a PDB file."
            )

        return receptor_path


    pdbs = sorted(
        input_dir.glob(
            "*.pdb"
        )
    )


    if len(pdbs) == 0:

        raise RuntimeError(
            "\nNo receptor PDB found in:\n"
            f"  {input_dir}\n\n"
            "Specify one explicitly with --receptor."
        )


    if len(pdbs) > 1:

        text = "\n".join(
            f"  {p.name}"
            for p in pdbs
        )

        raise RuntimeError(
            "\nMultiple PDB files found. "
            "Refusing to guess which is the receptor:\n"
            f"{text}\n\n"
            "Specify one with --receptor."
        )


    return pdbs[0].resolve()


def discover_sdf_files(
    input_dir: Path,
    ligand_patterns: list[str] | None,
    recursive: bool = False,
) -> list[Path]:

    files = []


    if ligand_patterns:

        for pattern in ligand_patterns:

            expanded = (
                Path(pattern)
                .expanduser()
            )


            # Explicit file.
            if expanded.exists():

                if expanded.is_file():

                    if (
                        expanded.suffix.lower()
                        == ".sdf"
                    ):

                        files.append(
                            expanded.resolve()
                        )

                    continue


                if expanded.is_dir():

                    iterator = (
                        expanded.rglob(
                            "*.sdf"
                        )
                        if recursive
                        else expanded.glob(
                            "*.sdf"
                        )
                    )

                    files.extend(
                        p.resolve()
                        for p in iterator
                    )

                    continue


            # Treat as glob relative to current working directory.
            parent = (
                expanded.parent
                if str(
                    expanded.parent
                )
                != "."
                else Path.cwd()
            )

            pattern_name = (
                expanded.name
            )

            iterator = (
                parent.rglob(
                    pattern_name
                )
                if recursive
                else parent.glob(
                    pattern_name
                )
            )

            files.extend(
                p.resolve()
                for p in iterator
                if (
                    p.is_file()
                    and
                    p.suffix.lower()
                    == ".sdf"
                )
            )


    else:

        iterator = (
            input_dir.rglob(
                "*.sdf"
            )
            if recursive
            else input_dir.glob(
                "*.sdf"
            )
        )

        files.extend(
            p.resolve()
            for p in iterator
        )


    # Deduplicate while preserving deterministic ordering.
    files = sorted(
        set(files),
        key=lambda p: str(p),
    )


    if not files:

        raise RuntimeError(
            "\nNo ligand SDF files were found.\n"
            "Provide SDF files in --input or use --ligands."
        )


    return files


def molecule_title(
    mol: Chem.Mol,
    source_file: Path,
    record_index: int,
) -> str:

    if mol.HasProp(
        "_Name"
    ):

        title = mol.GetProp(
            "_Name"
        ).strip()

        if title:
            return title


    if record_index == 1:

        return source_file.stem


    return (
        f"{source_file.stem}_pose{record_index}"
    )


def molecule_smiles(
    mol: Chem.Mol,
) -> str:

    try:

        heavy = Chem.RemoveHs(
            mol
        )

        return Chem.MolToSmiles(
            heavy,
            isomericSmiles=True,
        )

    except Exception:
        return ""


def molecule_formal_charge(
    mol: Chem.Mol,
) -> int:

    return int(
        sum(
            atom.GetFormalCharge()
            for atom in mol.GetAtoms()
        )
    )


def molecule_heavy_atoms(
    mol: Chem.Mol,
) -> int:

    return int(
        sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() > 1
        )
    )


def record_properties(
    mol: Chem.Mol,
) -> dict:

    result = {}

    for name in mol.GetPropNames():

        try:

            value = mol.GetProp(
                name
            )

        except Exception:
            continue

        column = (
            "sdfprop_"
            + safe_name(
                name,
                fallback="property",
            )
        )

        result[
            column
        ] = value


    return result


def write_one_record_sdf(
    mol: Chem.Mol,
    filename: Path,
):

    writer = Chem.SDWriter(
        str(filename)
    )

    writer.write(
        mol
    )

    writer.close()


def load_receptor_heavy_coordinates(
    receptor_pdb: Path,
) -> np.ndarray:

    coords = []

    with open(
        receptor_pdb,
    ) as handle:

        for line in handle:

            if not (
                line.startswith("ATOM")
                or
                line.startswith("HETATM")
            ):
                continue

            element = (
                line[76:78]
                .strip()
                .upper()
            )

            atom_name = (
                line[12:16]
                .strip()
            )

            # Fallback when element column is missing.
            if not element:

                letters = "".join(
                    c
                    for c in atom_name
                    if c.isalpha()
                )

                element = (
                    letters[:2]
                    .upper()
                )

            if element == "H":
                continue

            try:

                x = float(
                    line[30:38]
                )

                y = float(
                    line[38:46]
                )

                z = float(
                    line[46:54]
                )

            except ValueError:
                continue

            coords.append(
                [
                    x,
                    y,
                    z,
                ]
            )

    if not coords:

        raise RuntimeError(
            f"No receptor heavy-atom coordinates found in:\n"
            f"  {receptor_pdb}"
        )

    return np.asarray(
        coords,
        dtype=float,
    )


def ligand_heavy_coordinates(
    mol: Chem.Mol,
) -> np.ndarray:

    conf = mol.GetConformer()

    coords = []

    for atom in mol.GetAtoms():

        if atom.GetAtomicNum() == 1:
            continue

        pos = conf.GetAtomPosition(
            atom.GetIdx()
        )

        coords.append(
            [
                pos.x,
                pos.y,
                pos.z,
            ]
        )

    return np.asarray(
        coords,
        dtype=float,
    )


def coordinate_frame_metrics(
    mol: Chem.Mol,
    receptor_xyz: np.ndarray,
) -> dict:

    ligand_xyz = ligand_heavy_coordinates(
        mol
    )

    if len(
        ligand_xyz
    ) == 0:

        return {
            "nearest_receptor_heavy_distance_A":
                np.nan,

            "receptor_heavy_atoms_within_4A":
                0,

            "receptor_heavy_atoms_within_6A":
                0,

            "ligand_centroid_to_receptor_centroid_A":
                np.nan,

            "coordinate_frame_status":
                "WARNING_NO_LIGAND_HEAVY_ATOMS",
        }


    # Pairwise distances:
    #
    # ligand atoms x receptor atoms

    diff = (
        ligand_xyz[:, None, :]
        -
        receptor_xyz[None, :, :]
    )

    distances = np.linalg.norm(
        diff,
        axis=2,
    )


    nearest = float(
        np.min(
            distances
        )
    )


    receptor_min_distances = (
        np.min(
            distances,
            axis=0,
        )
    )


    within_4 = int(
        np.sum(
            receptor_min_distances
            <= 4.0
        )
    )


    within_6 = int(
        np.sum(
            receptor_min_distances
            <= 6.0
        )
    )


    ligand_centroid = (
        ligand_xyz.mean(
            axis=0
        )
    )

    receptor_centroid = (
        receptor_xyz.mean(
            axis=0
        )
    )


    centroid_distance = float(
        np.linalg.norm(
            ligand_centroid
            -
            receptor_centroid
        )
    )


    if (
        nearest > 8.0
        and
        within_6 == 0
    ):

        status = (
            "WARNING_NO_NEARBY_RECEPTOR"
        )

    else:

        status = "PASS"


    return {
        "nearest_receptor_heavy_distance_A":
            nearest,

        "receptor_heavy_atoms_within_4A":
            within_4,

        "receptor_heavy_atoms_within_6A":
            within_6,

        "ligand_centroid_to_receptor_centroid_A":
            centroid_distance,

        "coordinate_frame_status":
            status,
    }


def create_workspace(
    input_dir: Path,
    output_dir: Path,
    receptor: str | None = None,
    ligand_patterns: list[str] | None = None,
    recursive: bool = False,
    limit: int | None = None,
) -> dict:

    input_dir = (
        input_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        output_dir
        .expanduser()
        .resolve()
    )


    if not input_dir.exists():

        raise RuntimeError(
            f"Input directory does not exist:\n"
            f"  {input_dir}"
        )


    receptor_path = discover_receptor(
        input_dir,
        receptor,
    )


    receptor_xyz = (
        load_receptor_heavy_coordinates(
            receptor_path
        )
    )
    

    sdf_files = discover_sdf_files(
        input_dir,
        ligand_patterns,
        recursive=recursive,
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    ligand_dir = (
        output_dir
        / "ligands"
    )

    ligand_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    dockpost_dir = (
        output_dir
        / ".dockpost"
    )

    compat_dir = (
        dockpost_dir
        / "compat"
    )

    compat_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Canonical receptor copies.
    canonical_receptor = (
        output_dir
        / "receptor.pdb"
    )

    shutil.copy2(
        receptor_path,
        canonical_receptor,
    )


    compat_receptor = (
        compat_dir
        / "receptor.pdb"
    )

    shutil.copy2(
        receptor_path,
        compat_receptor,
    )


    entries = []

    ligand_counter = 0


    for source_file in sdf_files:

        supplier = Chem.SDMolSupplier(
            str(source_file),
            removeHs=False,
            sanitize=True,
        )


        for zero_index, mol in enumerate(
            supplier
        ):

            record_index = (
                zero_index + 1
            )


            if mol is None:

                print(
                    "WARNING: unreadable SDF record "
                    f"{record_index} in {source_file}"
                )

                continue


            if mol.GetNumConformers() == 0:

                print(
                    "WARNING: skipping record with no coordinates: "
                    f"{source_file} record {record_index}"
                )

                continue


            ligand_counter += 1


            if (
                limit is not None
                and
                ligand_counter > limit
            ):

                ligand_counter -= 1
                break


            ligand_id = (
                f"L{ligand_counter:04d}"
            )


            title = molecule_title(
                mol,
                source_file,
                record_index,
            )


            compound_id = safe_name(
                title,
                fallback=ligand_id,
            )


            canonical_sdf = (
                ligand_dir
                / f"{ligand_id}.sdf"
            )


            write_one_record_sdf(
                mol,
                canonical_sdf,
            )


            temporary_name = (
                f"{ligand_counter:03d}"
                f"__{ligand_id}"
                f"__{safe_name(title)}.sdf"
            )


            compatibility_sdf = (
                compat_dir
                / temporary_name
            )


            shutil.copy2(
                canonical_sdf,
                compatibility_sdf,
            )


            entry = {
                "ordinal":
                    ligand_counter,

                "ligand_id":
                    ligand_id,

                "compound_id":
                    compound_id,

                "pose_index":
                    record_index,

                "source_file":
                    str(
                        source_file
                    ),

                "source_filename":
                    source_file.name,

                "record_index":
                    record_index,

                "title":
                    title,

                "source_rank":
                    infer_source_rank(
                        source_file.name
                    ),

                "smiles":
                    molecule_smiles(
                        mol
                    ),

                "formal_charge":
                    molecule_formal_charge(
                        mol
                    ),

                "heavy_atoms":
                    molecule_heavy_atoms(
                        mol
                    ),

                "canonical_sdf":
                    str(
                        canonical_sdf
                    ),

                "compatibility_sdf":
                    str(
                        compatibility_sdf
                    ),
            }


            frame_metrics = (
                coordinate_frame_metrics(
                    mol,
                    receptor_xyz,
                )
            )

            
            entry.update(
                frame_metrics
            )


            if (
                frame_metrics[
                    "coordinate_frame_status"
                ]
                != "PASS"
            ):

                print()
                print(
                    f"WARNING {ligand_id}:"
                )

                print(
                    "  Ligand may not share the receptor "
                    "coordinate frame."
                )

                print(
                    "  Nearest receptor heavy atom: "
                    f"{frame_metrics['nearest_receptor_heavy_distance_A']:.2f} A"
                )

                print(
                    "  Receptor heavy atoms within 6 A: "
                    f"{frame_metrics['receptor_heavy_atoms_within_6A']}"
                )            


            entry.update(
                record_properties(
                    mol
                )
            )


            entries.append(
                entry
            )


        if (
            limit is not None
            and
            ligand_counter >= limit
        ):
            break


    if not entries:

        raise RuntimeError(
            "No valid ligand poses were imported."
        )


    # Collect all manifest columns, including arbitrary SDF props.
    columns = [
        "ordinal",
        "ligand_id",
        "compound_id",
        "pose_index",
        "source_file",
        "source_filename",
        "record_index",
        "title",
        "source_rank",
        "smiles",
        "formal_charge",
        "heavy_atoms",
        "canonical_sdf",
        "compatibility_sdf",
    ]


    extra_columns = sorted(
        {
            key
            for entry in entries
            for key in entry
            if key not in columns
        }
    )


    columns.extend(
        extra_columns
    )


    manifest_file = (
        output_dir
        / "ligand_manifest.csv"
    )


    with open(
        manifest_file,
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for entry in entries:

            writer.writerow(
                entry
            )


    workspace_info = {
        "format_version":
            1,

        "input_directory":
            str(
                input_dir
            ),

        "workspace_directory":
            str(
                output_dir
            ),

        "receptor_source":
            str(
                receptor_path
            ),

        "canonical_receptor":
            str(
                canonical_receptor
            ),

        "manifest":
            str(
                manifest_file
            ),

        "compat_directory":
            str(
                compat_dir
            ),

        "n_ligands":
            len(
                entries
            ),
    }


    workspace_file = (
        dockpost_dir
        / "workspace.json"
    )


    with open(
        workspace_file,
        "w",
    ) as handle:

        json.dump(
            workspace_info,
            handle,
            indent=2,
        )


    return workspace_info


def load_workspace(
    path: Path,
) -> dict:

    path = (
        path
        .expanduser()
        .resolve()
    )


    # User passed workspace root.
    candidate = (
        path
        / ".dockpost"
        / "workspace.json"
    )


    if candidate.exists():

        workspace_file = (
            candidate
        )


    # User passed public results symlink/directory.
    elif (
        path.name
        == "results"
        and
        (
            path.parent
            / ".dockpost"
            / "workspace.json"
        ).exists()
    ):

        workspace_file = (
            path.parent
            / ".dockpost"
            / "workspace.json"
        )


    else:

        raise RuntimeError(
            "\nNot a dock-postprocess generalized workspace:\n"
            f"  {path}\n\n"
            "Expected .dockpost/workspace.json."
        )


    with open(
        workspace_file,
    ) as handle:

        return json.load(
            handle
        )

