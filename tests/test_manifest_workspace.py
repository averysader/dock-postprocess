from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from dockpost.manifest import (
    create_workspace,
)


def write_test_receptor(
    filename: Path,
):

    filename.write_text(
        "ATOM      1  CA  ALA A   1       "
        "0.000   0.000   0.000  1.00 20.00           C\n"
        "END\n"
    )


def write_test_ligand(
    filename: Path,
):

    mol = Chem.AddHs(
        Chem.MolFromSmiles(
            "CCO"
        )
    )

    status = AllChem.EmbedMolecule(
        mol,
        randomSeed=2026,
    )

    assert status == 0

    mol.SetProp(
        "_Name",
        "ethanol_test"
    )

    writer = Chem.SDWriter(
        str(
            filename
        )
    )

    writer.write(
        mol
    )

    writer.close()


def test_create_workspace(
    tmp_path,
):

    input_dir = (
        tmp_path
        / "input"
    )

    output_dir = (
        tmp_path
        / "analysis"
    )

    input_dir.mkdir()


    write_test_receptor(
        input_dir
        / "receptor.pdb"
    )


    write_test_ligand(
        input_dir
        / "pose.sdf"
    )


    create_workspace(
        input_dir=input_dir,
        output_dir=output_dir,
    )


    assert (
        output_dir
        / "receptor.pdb"
    ).exists()


    assert (
        output_dir
        / "ligand_manifest.csv"
    ).exists()


    assert (
        output_dir
        / "ligands"
        / "L0001.sdf"
    ).exists()


    assert (
        output_dir
        / ".dockpost"
        / "workspace.json"
    ).exists()


    manifest = pd.read_csv(
        output_dir
        / "ligand_manifest.csv"
    )


    assert len(
        manifest
    ) == 1


    row = manifest.iloc[
        0
    ]


    assert (
        row[
            "ligand_id"
        ]
        == "L0001"
    )

    assert (
        row[
            "title"
        ]
        == "ethanol_test"
    )

    assert (
        row[
            "coordinate_frame_status"
        ]
        == "PASS"
    )
