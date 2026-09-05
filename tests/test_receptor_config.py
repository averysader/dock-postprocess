import json

from dockpost.receptor_config import (
    load_receptor_config,
)


def test_default_receptor_config():

    config = load_receptor_config(
        None
    )

    assert (
        "amber14/protein.ff14SB.xml"
        in config.forcefield_xmls
    )

    assert (
        "amber14/tip3p.xml"
        in config.forcefield_xmls
    )

    assert (
        config.residue_variants
        == []
    )

    assert (
        config.metal_restraints
        == []
    )

    assert (
        config.backbone_restraint_schedule_kcal_mol_A2
        == [
            100.0,
            10.0,
            1.0,
        ]
    )


def test_configured_receptor(tmp_path):

    filename = (
        tmp_path
        / "receptor.json"
    )

    data = {
        "forcefield_xmls": [
            "amber14/protein.ff14SB.xml",
            "amber14/tip3p.xml",
        ],

        "residue_variants": [
            {
                "chain": "B",
                "resid": "176",
                "residue_name": "CYM",
            }
        ],

        "metal_restraints": [
            {
                "metal": {
                    "resname": "ZN",
                    "atom": "ZN",
                },

                "partner": {
                    "chain": "B",
                    "resid": "176",
                    "atom": "SG",
                },

                "distance_A": None,

                "force_constant_kcal_mol_A2":
                    1000.0,
            }
        ],
    }

    filename.write_text(
        json.dumps(
            data
        )
    )

    config = load_receptor_config(
        filename
    )

    assert (
        len(
            config.residue_variants
        )
        == 1
    )

    assert (
        config.residue_variants[
            0
        ].residue_name
        == "CYM"
    )

    assert (
        config.residue_variants[
            0
        ].topology_load_name
        == "CYS"
    )

    assert (
        len(
            config.metal_restraints
        )
        == 1
    )

    assert (
        config.metal_restraints[
            0
        ].force_constant_kcal_mol_A2
        == 1000.0
    )
