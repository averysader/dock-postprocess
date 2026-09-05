#!/usr/bin/env python3

"""
Receptor configuration for dock-postprocess.

The default configuration is receptor-agnostic.

It makes no assumptions about:
    - receptor chain IDs
    - number of receptor chains
    - metal sites
    - mutation sites
    - target-specific residues

Optional configuration may specify:
    - protein force-field XML files
    - residue variants such as CYM/HIE
    - topology-loading aliases for nonstandard residue names
    - explicit metal coordination restraints
    - staged receptor-backbone restraint strengths

JSON is used so dock-postprocess requires no YAML dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import json


DEFAULT_FORCEFIELDS = [
    "amber14/protein.ff14SB.xml",
    "amber14/tip3p.xml",
]

DEFAULT_BACKBONE_RESTRAINTS = [
    100.0,
    10.0,
    1.0,
]


@dataclass(frozen=True)
class ResidueVariant:
    chain: str
    resid: str
    residue_name: str
    topology_load_name: str | None = None


@dataclass(frozen=True)
class AtomReference:
    atom: str
    chain: str | None = None
    resid: str | None = None
    resname: str | None = None


@dataclass(frozen=True)
class MetalRestraint:
    metal: AtomReference
    partner: AtomReference

    distance_A: float | None = None

    force_constant_kcal_mol_A2: float = (
        1000.0
    )


@dataclass
class ReceptorConfig:
    forcefield_xmls: list[str] = field(
        default_factory=lambda:
            list(
                DEFAULT_FORCEFIELDS
            )
    )

    residue_variants: list[
        ResidueVariant
    ] = field(
        default_factory=list
    )

    metal_restraints: list[
        MetalRestraint
    ] = field(
        default_factory=list
    )

    backbone_restraint_schedule_kcal_mol_A2: list[
        float
    ] = field(
        default_factory=lambda:
            list(
                DEFAULT_BACKBONE_RESTRAINTS
            )
    )


def _required_string(
    data: dict,
    key: str,
    context: str,
) -> str:

    value = data.get(
        key
    )

    if value is None:

        raise ValueError(
            f"{context}: missing required field '{key}'"
        )


    value = str(
        value
    ).strip()


    if not value:

        raise ValueError(
            f"{context}: field '{key}' cannot be empty"
        )


    return value


def parse_atom_reference(
    data: dict,
    context: str,
) -> AtomReference:

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            f"{context} must be an object."
        )


    atom = _required_string(
        data,
        "atom",
        context,
    )


    def optional(
        key,
    ):

        value = data.get(
            key
        )

        if value is None:

            return None


        value = str(
            value
        ).strip()


        return (
            value
            if value
            else None
        )


    return AtomReference(
        atom=atom,
        chain=optional(
            "chain"
        ),
        resid=optional(
            "resid"
        ),
        resname=optional(
            "resname"
        ),
    )


def parse_residue_variant(
    data: dict,
    index: int,
) -> ResidueVariant:

    context = (
        f"residue_variants[{index}]"
    )


    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            f"{context} must be an object."
        )


    topology_load_name = data.get(
        "topology_load_name"
    )


    if topology_load_name is not None:

        topology_load_name = (
            str(
                topology_load_name
            )
            .strip()
            .upper()
        )


        if not topology_load_name:

            topology_load_name = None


    residue_name = _required_string(
        data,
        "residue_name",
        context,
    ).upper()


    # CYM frequently needs to be read initially as CYS so
    # OpenMM PDBFile constructs the standard peptide/internal
    # topology before we restore the CYM force-field identity.
    if (
        residue_name == "CYM"
        and
        topology_load_name is None
    ):

        topology_load_name = "CYS"


    return ResidueVariant(
        chain=_required_string(
            data,
            "chain",
            context,
        ),

        resid=_required_string(
            data,
            "resid",
            context,
        ),

        residue_name=residue_name,

        topology_load_name=topology_load_name,
    )


def parse_metal_restraint(
    data: dict,
    index: int,
) -> MetalRestraint:

    context = (
        f"metal_restraints[{index}]"
    )


    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            f"{context} must be an object."
        )


    if "metal" not in data:

        raise ValueError(
            f"{context}: missing 'metal'"
        )


    if "partner" not in data:

        raise ValueError(
            f"{context}: missing 'partner'"
        )


    distance = data.get(
        "distance_A"
    )


    if distance is not None:

        distance = float(
            distance
        )


        if distance <= 0:

            raise ValueError(
                f"{context}: distance_A must be positive."
            )


    force_constant = float(
        data.get(
            "force_constant_kcal_mol_A2",
            1000.0,
        )
    )


    if force_constant <= 0:

        raise ValueError(
            f"{context}: force constant must be positive."
        )


    return MetalRestraint(
        metal=parse_atom_reference(
            data[
                "metal"
            ],
            f"{context}.metal",
        ),

        partner=parse_atom_reference(
            data[
                "partner"
            ],
            f"{context}.partner",
        ),

        distance_A=distance,

        force_constant_kcal_mol_A2=force_constant,
    )


def load_receptor_config(
    filename: Path | None,
) -> ReceptorConfig:

    if filename is None:

        return ReceptorConfig()


    filename = (
        Path(
            filename
        )
        .expanduser()
        .resolve()
    )


    if not filename.exists():

        raise FileNotFoundError(
            "\nReceptor configuration file does not exist:\n"
            f"  {filename}"
        )


    with open(
        filename
    ) as handle:

        data = json.load(
            handle
        )


    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "Top-level receptor configuration must be an object."
        )


    forcefields = data.get(
        "forcefield_xmls",
        DEFAULT_FORCEFIELDS,
    )


    if not isinstance(
        forcefields,
        list,
    ):

        raise ValueError(
            "forcefield_xmls must be a list."
        )


    forcefields = [
        str(
            value
        ).strip()
        for value
        in forcefields
    ]


    if (
        not forcefields
        or
        any(
            not value
            for value
            in forcefields
        )
    ):

        raise ValueError(
            "forcefield_xmls cannot be empty."
        )


    residue_variants = [
        parse_residue_variant(
            item,
            index,
        )
        for index, item
        in enumerate(
            data.get(
                "residue_variants",
                []
            )
        )
    ]


    metal_restraints = [
        parse_metal_restraint(
            item,
            index,
        )
        for index, item
        in enumerate(
            data.get(
                "metal_restraints",
                []
            )
        )
    ]


    schedule = [
        float(
            value
        )
        for value
        in data.get(
            "backbone_restraint_schedule_kcal_mol_A2",
            DEFAULT_BACKBONE_RESTRAINTS,
        )
    ]


    if not schedule:

        raise ValueError(
            "Backbone restraint schedule cannot be empty."
        )


    if any(
        value < 0
        for value
        in schedule
    ):

        raise ValueError(
            "Backbone restraint values cannot be negative."
        )


    return ReceptorConfig(
        forcefield_xmls=forcefields,
        residue_variants=residue_variants,
        metal_restraints=metal_restraints,
        backbone_restraint_schedule_kcal_mol_A2=schedule,
    )


def config_to_dict(
    config: ReceptorConfig,
) -> dict:

    return {
        "forcefield_xmls":
            list(
                config.forcefield_xmls
            ),

        "backbone_restraint_schedule_kcal_mol_A2":
            list(
                config.backbone_restraint_schedule_kcal_mol_A2
            ),

        "residue_variants": [
            {
                "chain":
                    item.chain,

                "resid":
                    item.resid,

                "residue_name":
                    item.residue_name,

                "topology_load_name":
                    item.topology_load_name,
            }
            for item
            in config.residue_variants
        ],

        "metal_restraints": [
            {
                "metal": {
                    "atom":
                        item.metal.atom,

                    "chain":
                        item.metal.chain,

                    "resid":
                        item.metal.resid,

                    "resname":
                        item.metal.resname,
                },

                "partner": {
                    "atom":
                        item.partner.atom,

                    "chain":
                        item.partner.chain,

                    "resid":
                        item.partner.resid,

                    "resname":
                        item.partner.resname,
                },

                "distance_A":
                    item.distance_A,

                "force_constant_kcal_mol_A2":
                    item.force_constant_kcal_mol_A2,
            }
            for item
            in config.metal_restraints
        ],
    }


def save_receptor_config(
    config: ReceptorConfig,
    filename: Path,
):

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        filename,
        "w",
    ) as handle:

        json.dump(
            config_to_dict(
                config
            ),
            handle,
            indent=2,
        )

        handle.write(
            "\n"
        )


def describe_receptor_config(
    config: ReceptorConfig,
):

    print()
    print("=" * 80)
    print("RECEPTOR CONFIGURATION")
    print("=" * 80)


    print(
        "Force-field XMLs:"
    )


    for filename in config.forcefield_xmls:

        print(
            f"  {filename}"
        )


    print(
        "Backbone restraint schedule: "
        + ", ".join(
            f"{value:g}"
            for value
            in config.backbone_restraint_schedule_kcal_mol_A2
        )
        + " kcal/mol/A^2"
    )


    print()
    print(
        f"Residue variants: {len(config.residue_variants)}"
    )


    for item in config.residue_variants:

        text = (
            f"  {item.chain}:{item.resid} "
            f"-> {item.residue_name}"
        )


        if item.topology_load_name:

            text += (
                f" "
                f"(load as {item.topology_load_name})"
            )


        print(
            text
        )


    print()
    print(
        f"Metal restraints: {len(config.metal_restraints)}"
    )


    for index, restraint in enumerate(
        config.metal_restraints,
        start=1,
    ):

        target = (
            "starting geometry"
            if restraint.distance_A is None
            else f"{restraint.distance_A:.3f} A"
        )


        print(
            f"  {index}: "
            f"{atom_reference_label(restraint.metal)}"
            f" -- "
            f"{atom_reference_label(restraint.partner)}"
            f" | target={target}"
            f" | k="
            f"{restraint.force_constant_kcal_mol_A2:g}"
        )


def atom_reference_label(
    reference: AtomReference,
) -> str:

    parts = []


    if reference.chain:

        parts.append(
            reference.chain
        )


    if reference.resid:

        parts.append(
            reference.resid
        )


    if reference.resname:

        parts.append(
            reference.resname
        )


    parts.append(
        reference.atom
    )


    return ":".join(
        parts
    )
