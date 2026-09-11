import pytest

import openmm.app as app
from openmm import unit

from dockpost.solvation import (
    make_solvation_config,
    require_periodic_for_npt,
    system_creation_kwargs,
)


def test_default_dry_solvation():

    config = make_solvation_config()

    assert config.mode == "none"
    assert config.padding_nm == 1.0
    assert config.ionic_strength_molar == 0.15


def test_explicit_solvation():

    config = make_solvation_config(
        mode="explicit",
        padding_nm=1.2,
        ionic_strength_molar=0.10,
        box_shape="octahedron",
    )

    assert config.mode == "explicit"
    assert config.padding_nm == 1.2
    assert config.ionic_strength_molar == 0.10
    assert config.box_shape == "octahedron"


def test_dry_system_uses_no_cutoff():

    config = make_solvation_config(
        mode="none"
    )

    kwargs = system_creation_kwargs(
        config
    )

    assert (
        kwargs["nonbondedMethod"]
        == app.NoCutoff
    )

    assert (
        "nonbondedCutoff"
        not in kwargs
    )


def test_explicit_system_uses_pme():

    config = make_solvation_config(
        mode="explicit",
        nonbonded_cutoff_nm=1.1,
    )

    kwargs = system_creation_kwargs(
        config
    )

    assert (
        kwargs["nonbondedMethod"]
        == app.PME
    )

    cutoff_nm = (
        kwargs["nonbondedCutoff"]
        .value_in_unit(
            unit.nanometer
        )
    )

    assert cutoff_nm == pytest.approx(
        1.1
    )


def test_nvt_allows_dry_system():

    config = make_solvation_config(
        mode="none"
    )

    require_periodic_for_npt(
        "nvt",
        config,
    )


def test_npt_rejects_dry_system():

    config = make_solvation_config(
        mode="none"
    )

    with pytest.raises(
        ValueError,
        match="NPT simulations require",
    ):

        require_periodic_for_npt(
            "npt",
            config,
        )


def test_npt_accepts_explicit_system():

    config = make_solvation_config(
        mode="explicit"
    )

    require_periodic_for_npt(
        "npt",
        config,
    )


@pytest.mark.parametrize(
    "shape",
    [
        "cube",
        "dodecahedron",
        "octahedron",
    ],
)
def test_supported_box_shapes(
    shape,
):

    config = make_solvation_config(
        mode="explicit",
        box_shape=shape,
    )

    assert (
        config.box_shape
        == shape
    )


def test_invalid_box_shape_fails():

    with pytest.raises(
        ValueError
    ):

        make_solvation_config(
            mode="explicit",
            box_shape="sphere",
        )


def test_negative_ionic_strength_fails():

    with pytest.raises(
        ValueError
    ):

        make_solvation_config(
            mode="explicit",
            ionic_strength_molar=-0.1,
        )
