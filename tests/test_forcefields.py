import pytest

from dockpost.forcefields import (
    installed_gaff_forcefields,
    installed_openff_forcefields,
    resolve_forcefield_selection,
    resolve_ligand_forcefield,
    resolve_protein_forcefield,
    resolve_water_model,
)


def test_default_forcefield_selection():

    selection = (
        resolve_forcefield_selection()
    )


    assert (
        selection.protein.name
        == "ff14SB"
    )

    assert (
        selection.protein.xml
        == "amber14/protein.ff14SB.xml"
    )

    assert (
        selection.ligand.name
        == "openff-2.3.0"
    )

    assert (
        selection.ligand.backend
        == "smirnoff"
    )

    assert (
        selection.water.name
        == "tip3p"
    )

    assert (
        selection.water.xml
        == "amber14/tip3p.xml"
    )


def test_ff19sb_uses_amber19_water_namespace():

    protein = (
        resolve_protein_forcefield(
            "ff19SB"
        )
    )


    water = resolve_water_model(
        "opc",
        protein.family,
    )


    assert (
        protein.xml
        == "amber19/protein.ff19SB.xml"
    )

    assert (
        water.xml
        == "amber19/opc.xml"
    )

    assert (
        water.modeller_model
        == "tip4pew"
    )

    assert (
        water.sites
        == 4
    )


@pytest.mark.parametrize(
    "requested,expected",
    [
        (
            "tip3p-fb",
            "tip3pfb",
        ),
        (
            "tip4p-ew",
            "tip4pew",
        ),
        (
            "tip4p-fb",
            "tip4pfb",
        ),
        (
            "spc/e",
            "spce",
        ),
        (
            "opc-3",
            "opc3",
        ),
    ],
)
def test_water_aliases(
    requested,
    expected,
):

    water = resolve_water_model(
        requested,
        "amber14",
    )


    assert (
        water.name
        == expected
    )


def test_installed_openff_contains_sage_230():

    assert (
        "openff-2.3.0"
        in installed_openff_forcefields()
    )


def test_installed_gaff_contains_211():

    assert (
        "gaff-2.11"
        in installed_gaff_forcefields()
    )


def test_resolve_gaff():

    spec = resolve_ligand_forcefield(
        "gaff-2.11"
    )


    assert (
        spec.backend
        == "gaff"
    )

    assert (
        spec.name
        == "gaff-2.11"
    )


def test_unknown_protein_forcefield_fails():

    with pytest.raises(
        ValueError
    ):

        resolve_protein_forcefield(
            "made-up-forcefield"
        )


def test_unknown_ligand_forcefield_fails():

    with pytest.raises(
        ValueError
    ):

        resolve_ligand_forcefield(
            "made-up-ligand-forcefield"
        )
