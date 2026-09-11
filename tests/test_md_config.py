import pytest

from dockpost.md import MDConfig, steps_for_ps, validate_md_config
from dockpost.solvation import make_solvation_config, require_periodic_for_npt


def test_md_defaults_validate():
    cfg = validate_md_config(MDConfig())
    assert cfg.ensemble == "npt"
    assert cfg.temperature_K == 300.0
    assert cfg.timestep_fs == 2.0


def test_steps_for_ps():
    assert steps_for_ps(10.0, 2.0) == 5000
    assert steps_for_ps(0.0, 2.0) == 0


def test_npt_requires_explicit_solvent():
    dry = make_solvation_config(mode="none")
    with pytest.raises(ValueError):
        require_periodic_for_npt("npt", dry)


def test_nvt_allows_dry():
    dry = make_solvation_config(mode="none")
    require_periodic_for_npt("nvt", dry)


def test_invalid_md_timestep_rejected():
    with pytest.raises(ValueError):
        validate_md_config(MDConfig(timestep_fs=0.0))
