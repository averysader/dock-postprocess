#!/usr/bin/env python3

"""
Shared solvent and periodic-system configuration for dock-postprocess.

This module is used by both minimization and molecular dynamics.

Modes
-----
none
    No bulk solvent is added.  System construction uses NoCutoff.

explicit
    Add an explicit periodic water/ion box with OpenMM Modeller.addSolvent().
    System construction uses PME.

The selected Amber water XML supplies the actual water and ion parameters.
The modeller_model stored in dockpost.forcefields controls only the initial
water topology used by Modeller.addSolvent().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import openmm.app as app
from openmm import unit

from dockpost.forcefields import ForceFieldSelection


DEFAULT_SOLVENT_MODE = "none"
DEFAULT_PADDING_NM = 1.0
DEFAULT_IONIC_STRENGTH_MOLAR = 0.15
DEFAULT_POSITIVE_ION = "Na+"
DEFAULT_NEGATIVE_ION = "Cl-"
DEFAULT_BOX_SHAPE = "dodecahedron"
DEFAULT_NONBONDED_CUTOFF_NM = 1.0


SUPPORTED_SOLVENT_MODES = (
    "none",
    "explicit",
)

SUPPORTED_BOX_SHAPES = (
    "cube",
    "dodecahedron",
    "octahedron",
)

SUPPORTED_POSITIVE_IONS = (
    "Cs+",
    "K+",
    "Li+",
    "Na+",
    "Rb+",
)

SUPPORTED_NEGATIVE_IONS = (
    "Br-",
    "Cl-",
    "F-",
    "I-",
)


@dataclass(frozen=True)
class SolvationConfig:
    mode: str = DEFAULT_SOLVENT_MODE

    padding_nm: float = DEFAULT_PADDING_NM

    ionic_strength_molar: float = (
        DEFAULT_IONIC_STRENGTH_MOLAR
    )

    positive_ion: str = DEFAULT_POSITIVE_ION

    negative_ion: str = DEFAULT_NEGATIVE_ION

    box_shape: str = DEFAULT_BOX_SHAPE

    nonbonded_cutoff_nm: float = (
        DEFAULT_NONBONDED_CUTOFF_NM
    )

    neutralize: bool = True


def _normalize_mode(value: str) -> str:

    return str(value).strip().lower()


def validate_solvation_config(
    config: SolvationConfig,
) -> SolvationConfig:

    mode = _normalize_mode(
        config.mode
    )

    if mode not in SUPPORTED_SOLVENT_MODES:

        raise ValueError(
            f"Unsupported solvent mode '{config.mode}'. "
            "Supported values: "
            + ", ".join(SUPPORTED_SOLVENT_MODES)
        )

    if config.padding_nm <= 0:

        raise ValueError(
            "padding_nm must be greater than zero."
        )

    if config.ionic_strength_molar < 0:

        raise ValueError(
            "ionic_strength_molar cannot be negative."
        )

    if config.nonbonded_cutoff_nm <= 0:

        raise ValueError(
            "nonbonded_cutoff_nm must be greater than zero."
        )

    if config.box_shape not in SUPPORTED_BOX_SHAPES:

        raise ValueError(
            f"Unsupported box shape '{config.box_shape}'. "
            "Supported values: "
            + ", ".join(SUPPORTED_BOX_SHAPES)
        )

    if config.positive_ion not in SUPPORTED_POSITIVE_IONS:

        raise ValueError(
            f"Unsupported positive ion '{config.positive_ion}'. "
            "Supported values: "
            + ", ".join(SUPPORTED_POSITIVE_IONS)
        )

    if config.negative_ion not in SUPPORTED_NEGATIVE_IONS:

        raise ValueError(
            f"Unsupported negative ion '{config.negative_ion}'. "
            "Supported values: "
            + ", ".join(SUPPORTED_NEGATIVE_IONS)
        )

    return SolvationConfig(
        mode=mode,
        padding_nm=float(config.padding_nm),
        ionic_strength_molar=float(
            config.ionic_strength_molar
        ),
        positive_ion=config.positive_ion,
        negative_ion=config.negative_ion,
        box_shape=config.box_shape,
        nonbonded_cutoff_nm=float(
            config.nonbonded_cutoff_nm
        ),
        neutralize=bool(config.neutralize),
    )


def make_solvation_config(
    mode: str = DEFAULT_SOLVENT_MODE,
    padding_nm: float = DEFAULT_PADDING_NM,
    ionic_strength_molar: float = (
        DEFAULT_IONIC_STRENGTH_MOLAR
    ),
    positive_ion: str = DEFAULT_POSITIVE_ION,
    negative_ion: str = DEFAULT_NEGATIVE_ION,
    box_shape: str = DEFAULT_BOX_SHAPE,
    nonbonded_cutoff_nm: float = (
        DEFAULT_NONBONDED_CUTOFF_NM
    ),
    neutralize: bool = True,
) -> SolvationConfig:

    return validate_solvation_config(
        SolvationConfig(
            mode=mode,
            padding_nm=padding_nm,
            ionic_strength_molar=(
                ionic_strength_molar
            ),
            positive_ion=positive_ion,
            negative_ion=negative_ion,
            box_shape=box_shape,
            nonbonded_cutoff_nm=(
                nonbonded_cutoff_nm
            ),
            neutralize=neutralize,
        )
    )


def is_explicit(
    config: SolvationConfig,
) -> bool:

    return (
        _normalize_mode(config.mode)
        == "explicit"
    )


def solvate_modeller(
    modeller: app.Modeller,
    forcefield: app.ForceField,
    selection: ForceFieldSelection,
    config: SolvationConfig,
) -> app.Modeller:

    """
    Apply the requested solvent model to an OpenMM Modeller.

    The Modeller is modified in place and also returned for convenience.
    """

    config = validate_solvation_config(
        config
    )

    if config.mode == "none":

        # This is harmless for ordinary protein/ligand systems and makes
        # topology preparation robust if a force field requires extra
        # particles for any residues already present.
        modeller.addExtraParticles(
            forcefield
        )

        return modeller

    modeller.addSolvent(
        forcefield,
        model=(
            selection.water.modeller_model
        ),
        padding=(
            config.padding_nm
            * unit.nanometer
        ),
        ionicStrength=(
            config.ionic_strength_molar
            * unit.molar
        ),
        positiveIon=(
            config.positive_ion
        ),
        negativeIon=(
            config.negative_ion
        ),
        neutralize=(
            config.neutralize
        ),
        boxShape=(
            config.box_shape
        ),
    )

    # Ensure virtual sites / extra particles correspond to the actual
    # selected force-field XML rather than relying solely on the
    # topology-building solvent model used by addSolvent().
    modeller.addExtraParticles(
        forcefield
    )

    return modeller


def system_creation_kwargs(
    config: SolvationConfig,
) -> dict:

    """
    Return OpenMM ForceField.createSystem() nonbonded settings.
    """

    config = validate_solvation_config(
        config
    )

    if config.mode == "explicit":

        return {
            "nonbondedMethod":
                app.PME,

            "nonbondedCutoff":
                (
                    config.nonbonded_cutoff_nm
                    * unit.nanometer
                ),
        }

    return {
        "nonbondedMethod":
            app.NoCutoff,
    }


def require_periodic_for_npt(
    ensemble: str,
    config: SolvationConfig,
):

    ensemble_normalized = (
        str(ensemble)
        .strip()
        .lower()
    )

    if ensemble_normalized not in {
        "nvt",
        "npt",
    }:

        raise ValueError(
            f"Unsupported ensemble '{ensemble}'. "
            "Supported values: nvt, npt"
        )

    if (
        ensemble_normalized == "npt"
        and not is_explicit(config)
    ):

        raise ValueError(
            "NPT simulations require an explicit periodic "
            "solvent box. Use --solvent explicit or choose NVT."
        )


def topology_counts(
    topology: app.Topology,
) -> dict:

    atoms = list(
        topology.atoms()
    )

    residues = list(
        topology.residues()
    )

    chains = list(
        topology.chains()
    )

    waters = [
        residue
        for residue in residues
        if residue.name in {
            "HOH",
            "WAT",
        }
    ]

    common_ions = {
        "NA",
        "CL",
        "K",
        "LI",
        "RB",
        "CS",
        "BR",
        "F",
        "I",
    }

    ions = [
        residue
        for residue in residues
        if residue.name.upper()
        in common_ions
    ]

    return {
        "atoms":
            len(atoms),

        "residues":
            len(residues),

        "chains":
            len(chains),

        "waters":
            len(waters),

        "ions":
            len(ions),
    }


def solvation_metadata(
    selection: ForceFieldSelection,
    config: SolvationConfig,
    topology: app.Topology | None = None,
) -> dict:

    config = validate_solvation_config(
        config
    )

    data = {
        "solvation":
            asdict(config),

        "protein_forcefield":
            selection.protein.name,

        "protein_forcefield_xml":
            selection.protein.xml,

        "ligand_forcefield":
            selection.ligand.name,

        "ligand_backend":
            selection.ligand.backend,

        "water_model":
            selection.water.name,

        "water_forcefield_xml":
            selection.water.xml,

        "water_modeller_model":
            selection.water.modeller_model,

        "water_sites":
            selection.water.sites,
    }

    if topology is not None:

        data["topology"] = (
            topology_counts(topology)
        )

        vectors = (
            topology
            .getPeriodicBoxVectors()
        )

        data["periodic_box_present"] = (
            vectors is not None
        )

    return data


def write_solvation_metadata(
    path: str | Path,
    selection: ForceFieldSelection,
    config: SolvationConfig,
    topology: app.Topology | None = None,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = solvation_metadata(
        selection=selection,
        config=config,
        topology=topology,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )


def describe_solvation(
    selection: ForceFieldSelection,
    config: SolvationConfig,
):

    config = validate_solvation_config(
        config
    )

    print()
    print("=" * 80)
    print("SOLVATION")
    print("=" * 80)

    print(
        f"Mode:              {config.mode}"
    )

    if config.mode == "none":

        print(
            "Nonbonded method:  NoCutoff"
        )

        return

    print(
        f"Water model:       {selection.water.name}"
    )

    print(
        f"Water XML:         {selection.water.xml}"
    )

    print(
        "Topology builder:  "
        f"{selection.water.modeller_model}"
    )

    print(
        f"Padding:           {config.padding_nm:.3f} nm"
    )

    print(
        "Ionic strength:    "
        f"{config.ionic_strength_molar:.3f} M"
    )

    print(
        f"Ions:              "
        f"{config.positive_ion} / "
        f"{config.negative_ion}"
    )

    print(
        f"Box shape:         {config.box_shape}"
    )

    print(
        "Nonbonded method:  PME"
    )

    print(
        "Nonbonded cutoff:  "
        f"{config.nonbonded_cutoff_nm:.3f} nm"
    )
