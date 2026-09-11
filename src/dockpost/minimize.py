#!/usr/bin/env python3

"""
Native generalized OpenMM minimization for dock-postprocess.

This implementation preserves the validated scientific workflow while
removing historical assumptions about:
    - ranked filenames
    - topN directories
    - chain B
    - p53
    - Zn
    - fixed residue numbers

Public input
------------
analysis/
    receptor.pdb
    ligand_manifest.csv
    ligands/L####.sdf

Public output
-------------
results/
    L####/
        complex_minimized.pdb
        ligand_minimized.sdf

    minimization_summary.csv
    receptor_config_used.json

Scientific defaults retained from the validated workflow:
    - OpenFF 2.3.0 ligand force field
    - Amber ff14SB protein force field
    - Amber TIP3P/ion XML
    - NoCutoff
    - HBond constraints
    - backbone restraint schedule 100 -> 10 -> 1 kcal/mol/A^2
    - minimization tolerance 1 kJ/mol/nm
    - maximum 5000 iterations per stage

Special receptor chemistry is explicit through --receptor-config.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

from rdkit import Chem

from openff.toolkit import Molecule

from openff.units import unit as off_unit

from openmm import (
    CustomBondForce,
    CustomExternalForce,
    LangevinMiddleIntegrator,
    Platform,
    unit,
)

from openmm.app import (
    PDBFile,
    Modeller,
    Simulation,
    HBonds,
)


THIS_FILE = Path(__file__).resolve()


sys.path.insert(
    0,
    str(
        THIS_FILE.parent.parent
    ),
)


from dockpost.workspace import Workspace  # noqa: E402

from dockpost.forcefields import (  # noqa: E402
    create_forcefield,
    describe_forcefield_selection,
)

from dockpost.solvation import (  # noqa: E402
    describe_solvation,
    make_solvation_config,
    solvate_modeller,
    system_creation_kwargs,
    write_solvation_metadata,
)

from dockpost.receptor_config import (  # noqa: E402
    AtomReference,
    MetalRestraint,
    ReceptorConfig,
    atom_reference_label,
    describe_receptor_config,
    load_receptor_config,
    save_receptor_config,
)



BACKBONE_ATOMS = {
    "N",
    "CA",
    "C",
    "O",
}

DEFAULT_MAX_ITERATIONS = (
    5000
)

DEFAULT_MINIMIZATION_TOLERANCE = (
    1.0
    * unit.kilojoule_per_mole
    / unit.nanometer
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def energy_kcal(
    context,
) -> float:

    state = context.getState(
        getEnergy=True
    )


    return (
        state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilocalorie_per_mole
        )
    )


def get_position_array_A(
    positions,
) -> np.ndarray:

    array = np.array(
        [
            [
                position.x,
                position.y,
                position.z,
            ]
            for position
            in positions
        ]
    )


    return (
        array
        * 10.0
    )


def distance_A(
    position_a,
    position_b,
) -> float:

    dx = (
        position_a.x
        - position_b.x
    )

    dy = (
        position_a.y
        - position_b.y
    )

    dz = (
        position_a.z
        - position_b.z
    )


    return (
        float(
            np.sqrt(
                dx * dx
                + dy * dy
                + dz * dz
            )
        )
        * 10.0
    )


def sanitize_column_component(
    text: str,
) -> str:

    result = []


    for character in str(
        text
    ):

        if (
            character.isalnum()
            or
            character == "_"
        ):

            result.append(
                character
            )

        else:

            result.append(
                "_"
            )


    return "".join(
        result
    )


# ============================================================
# LIGAND LOADING
# ============================================================

def load_ligand_sdf(
    sdf_path: Path,
):

    supplier = Chem.SDMolSupplier(
        str(
            sdf_path
        ),
        removeHs=False,
        sanitize=True,
    )


    mol = next(
        (
            candidate
            for candidate
            in supplier
            if candidate is not None
        ),
        None,
    )


    if mol is None:

        raise RuntimeError(
            f"Could not read valid molecule from {sdf_path}"
        )


    if mol.GetNumConformers() != 1:

        raise RuntimeError(
            f"{sdf_path}: expected one conformer, "
            f"found {mol.GetNumConformers()}"
        )


    heavy_before = (
        mol.GetNumHeavyAtoms()
    )


    explicit_hydrogens = sum(
        1
        for atom
        in mol.GetAtoms()
        if atom.GetAtomicNum() == 1
    )


    if explicit_hydrogens == 0:

        print(
            "  Ligand contains no explicit H; "
            "adding hydrogens with coordinates."
        )


        mol = Chem.AddHs(
            mol,
            addCoords=True,
        )


    Chem.AssignAtomChiralTagsFromStructure(
        mol,
        confId=0,
        replaceExistingTags=False,
    )


    Chem.SanitizeMol(
        mol
    )


    print(
        f"  Ligand atoms: {mol.GetNumAtoms()} "
        f"({heavy_before} heavy)"
    )


    print(
        "  Formal charge: "
        f"{Chem.GetFormalCharge(mol):+d}"
    )


    return mol


def rdkit_to_openff(
    rdkit_mol,
):

    return Molecule.from_rdkit(
        rdkit_mol,
        allow_undefined_stereo=False,
        hydrogens_are_explicit=True,
    )


# ============================================================
# RECEPTOR PDB PREPARATION
# ============================================================

def pdb_residue_identity(
    line: str,
) -> tuple[str, str, str]:

    resname = (
        line[
            17:20
        ]
        .strip()
    )

    chain = (
        line[
            21:22
        ]
        .strip()
    )

    resid = (
        line[
            22:26
        ]
        .strip()
    )

    insertion_code = (
        line[
            26:27
        ]
        .strip()
    )


    if insertion_code:

        resid = (
            resid
            + insertion_code
        )


    return (
        chain,
        resid,
        resname,
    )


def variant_map(
    config: ReceptorConfig,
):

    return {
        (
            variant.chain,
            variant.resid,
        ):
        variant

        for variant
        in config.residue_variants
    }


def load_receptor(
    receptor_path: Path,
    config: ReceptorConfig,
):

    if not receptor_path.exists():

        raise FileNotFoundError(
            receptor_path
        )


    variants = variant_map(
        config
    )


    temporary_path = (
        receptor_path.parent
        / ".dockpost_receptor_topology_tmp.pdb"
    )


    output_lines = []


    for line in (
        receptor_path
        .read_text()
        .splitlines()
    ):

        if line.startswith(
            (
                "ATOM  ",
                "HETATM",
            )
        ):

            (
                chain,
                resid,
                resname,
            ) = pdb_residue_identity(
                line
            )


            variant = variants.get(
                (
                    chain,
                    resid,
                )
            )


            if (
                variant is not None
                and
                variant.topology_load_name
                is not None
                and
                resname
                != variant.topology_load_name
            ):

                load_name = (
                    variant.topology_load_name
                )


                if len(
                    load_name
                ) > 3:

                    raise RuntimeError(
                        "PDB residue names must be <=3 characters "
                        f"for topology loading: {load_name}"
                    )


                line = (
                    line[
                        :17
                    ]
                    + f"{load_name:>3s}"
                    + line[
                        20:
                    ]
                )


        output_lines.append(
            line
        )


    temporary_path.write_text(
        "\n".join(
            output_lines
        )
        + "\n"
    )


    try:

        pdb = PDBFile(
            str(
                temporary_path
            )
        )


    finally:

        temporary_path.unlink(
            missing_ok=True
        )


    # --------------------------------------------------------
    # RESTORE EXPLICIT FORCE-FIELD RESIDUE IDENTITIES
    # --------------------------------------------------------

    if config.residue_variants:

        print(
            "Restoring configured receptor residue variants:"
        )


    found_variants = set()


    for residue in pdb.topology.residues():

        key = (
            residue.chain.id,
            str(
                residue.id
            ),
        )


        variant = variants.get(
            key
        )


        if variant is None:

            continue


        print(
            f"  "
            f"{residue.chain.id}:"
            f"{residue.id} "
            f"{residue.name} "
            f"-> {variant.residue_name}"
        )


        residue.name = (
            variant.residue_name
        )


        found_variants.add(
            key
        )


    missing_variants = (
        set(
            variants
        )
        - found_variants
    )


    if missing_variants:

        text = ", ".join(
            f"{chain}:{resid}"
            for (
                chain,
                resid,
            )
            in sorted(
                missing_variants
            )
        )


        raise RuntimeError(
            "Configured residue variants were not found "
            f"in receptor: {text}"
        )


    return pdb


# ============================================================
# BUILD COMPLEX
# ============================================================

def build_complex(
    receptor_pdb,
    offmol,
):

    modeller = Modeller(
        receptor_pdb.topology,
        receptor_pdb.positions,
    )


    receptor_atom_count = sum(
        1
        for _
        in modeller.topology.atoms()
    )


    ligand_topology = (
        offmol
        .to_topology()
        .to_openmm()
    )


    ligand_xyz_nm = (
        offmol.conformers[
            0
        ]
        .m_as(
            off_unit.nanometer
        )
    )


    ligand_positions = (
        ligand_xyz_nm
        * unit.nanometer
    )


    modeller.add(
        ligand_topology,
        ligand_positions,
    )


    total_atoms = sum(
        1
        for _
        in modeller.topology.atoms()
    )


    ligand_atom_count = (
        total_atoms
        - receptor_atom_count
    )


    print(
        f"  Receptor atoms: {receptor_atom_count}"
    )

    print(
        f"  Ligand atoms:   {ligand_atom_count}"
    )

    print(
        f"  Total atoms:    {total_atoms}"
    )


    return (
        modeller,
        receptor_atom_count,
        ligand_atom_count,
    )


# ============================================================
# BACKBONE RESTRAINTS
# ============================================================

def add_backbone_restraints(
    system,
    topology,
    positions,
    receptor_atom_count: int,
):

    force = CustomExternalForce(
        "0.5*k_backbone*periodicdistance("
        "x,y,z,x0,y0,z0)^2"
    )


    force.addGlobalParameter(
        "k_backbone",
        0.0,
    )


    for name in [
        "x0",
        "y0",
        "z0",
    ]:

        force.addPerParticleParameter(
            name
        )


    count = 0


    for atom in topology.atoms():

        if (
            atom.index
            >= receptor_atom_count
        ):

            continue


        if (
            atom.name
            not in BACKBONE_ATOMS
        ):

            continue


        position = (
            positions[
                atom.index
            ]
        )


        force.addParticle(
            atom.index,
            [
                position.x,
                position.y,
                position.z,
            ],
        )


        count += 1


    system.addForce(
        force
    )


    print(
        "  Restrained receptor backbone atoms: "
        f"{count}"
    )


    return force


# ============================================================
# GENERIC ATOM LOOKUP
# ============================================================

def atom_matches_reference(
    atom,
    reference: AtomReference,
) -> bool:

    if (
        atom.name
        != reference.atom
    ):

        return False


    residue = (
        atom.residue
    )


    if (
        reference.chain is not None
        and
        residue.chain.id
        != reference.chain
    ):

        return False


    if (
        reference.resid is not None
        and
        str(
            residue.id
        )
        != str(
            reference.resid
        )
    ):

        return False


    if (
        reference.resname is not None
        and
        residue.name.upper()
        != reference.resname.upper()
    ):

        return False


    return True


def find_unique_atom(
    topology,
    reference: AtomReference,
    receptor_atom_count: int,
):

    matches = [
        atom
        for atom
        in topology.atoms()
        if (
            atom.index
            < receptor_atom_count
            and
            atom_matches_reference(
                atom,
                reference,
            )
        )
    ]


    if len(matches) != 1:

        raise RuntimeError(
            "Expected exactly one receptor atom matching "
            f"{atom_reference_label(reference)}, "
            f"found {len(matches)}."
        )


    return matches[
        0
    ]


# ============================================================
# GENERIC METAL RESTRAINTS
# ============================================================

def restraint_key(
    restraint: MetalRestraint,
) -> str:

    metal = sanitize_column_component(
        restraint.metal.atom
    )


    partner_parts = []


    if restraint.partner.chain:

        partner_parts.append(
            restraint.partner.chain
        )


    if restraint.partner.resid:

        partner_parts.append(
            restraint.partner.resid
        )


    partner_parts.append(
        restraint.partner.atom
    )


    partner = sanitize_column_component(
        "_".join(
            partner_parts
        )
    )


    return (
        f"{metal}_{partner}_A"
    )


def add_metal_restraints(
    system,
    topology,
    positions,
    receptor_atom_count: int,
    config: ReceptorConfig,
):

    if not config.metal_restraints:

        print(
            "  No configured metal restraints."
        )

        return (
            None,
            [],
        )


    force = CustomBondForce(
        "0.5*k*(r-r0)^2"
    )


    force.addPerBondParameter(
        "r0"
    )

    force.addPerBondParameter(
        "k"
    )


    restraint_records = []


    for index, restraint in enumerate(
        config.metal_restraints,
        start=1,
    ):

        metal_atom = find_unique_atom(
            topology,
            restraint.metal,
            receptor_atom_count,
        )


        partner_atom = find_unique_atom(
            topology,
            restraint.partner,
            receptor_atom_count,
        )


        starting_distance_A = distance_A(
            positions[
                metal_atom.index
            ],
            positions[
                partner_atom.index
            ],
        )


        target_distance_A = (
            starting_distance_A
            if restraint.distance_A is None
            else restraint.distance_A
        )


        k_internal = (
            restraint.force_constant_kcal_mol_A2
            * unit.kilocalorie_per_mole
            / unit.angstrom**2
        ).value_in_unit(
            unit.kilojoule_per_mole
            / unit.nanometer**2
        )


        force.addBond(
            metal_atom.index,
            partner_atom.index,
            [
                target_distance_A
                / 10.0,

                k_internal,
            ],
        )


        record = {
            "index":
                index,

            "key":
                restraint_key(
                    restraint
                ),

            "metal_label":
                atom_reference_label(
                    restraint.metal
                ),

            "partner_label":
                atom_reference_label(
                    restraint.partner
                ),

            "metal_index":
                metal_atom.index,

            "partner_index":
                partner_atom.index,

            "starting_distance_A":
                starting_distance_A,

            "target_distance_A":
                target_distance_A,

            "force_constant_kcal_mol_A2":
                restraint.force_constant_kcal_mol_A2,
        }


        restraint_records.append(
            record
        )


    system.addForce(
        force
    )


    print(
        f"  Added {len(restraint_records)} "
        "configured metal coordination restraints:"
    )


    for record in restraint_records:

        print(
            "    "
            f"{record['metal_label']} "
            "-- "
            f"{record['partner_label']} "
            f"r0={record['target_distance_A']:.3f} A "
            f"k={record['force_constant_kcal_mol_A2']:.1f}"
        )


    return (
        force,
        restraint_records,
    )


def report_metal_distances(
    positions,
    restraint_records: list[dict],
) -> dict:

    result = {}


    for record in restraint_records:

        distance = distance_A(
            positions[
                record[
                    "metal_index"
                ]
            ],
            positions[
                record[
                    "partner_index"
                ]
            ],
        )


        result[
            record[
                "key"
            ]
        ] = distance


    return result


# ============================================================
# PLATFORM
# ============================================================

def select_platform(
    requested: str,
):

    requested = (
        requested.strip()
    )


    if requested.lower() != "auto":

        platform = (
            Platform.getPlatformByName(
                requested
            )
        )


        return (
            platform,
            requested,
        )


    available = [
        Platform.getPlatform(
            index
        ).getName()
        for index
        in range(
            Platform.getNumPlatforms()
        )
    ]


    for preferred in [
        "CUDA",
        "OpenCL",
        "CPU",
    ]:

        if preferred in available:

            return (
                Platform.getPlatformByName(
                    preferred
                ),
                preferred,
            )


    raise RuntimeError(
        "No supported OpenMM platform found. "
        f"Available: {available}"
    )


# ============================================================
# MINIMIZE ONE LIGAND
# ============================================================

def minimize_ligand(
    workspace: Workspace,
    ligand,
    receptor_pdb,
    config: ReceptorConfig,
    platform_name: str,
    precision: str,
    max_iterations: int,
    protein_forcefield: str,
    ligand_forcefield: str,
    water_model: str,
    solvent_mode: str,
    padding_nm: float,
    ionic_strength_molar: float,
    positive_ion: str,
    negative_ion: str,
    box_shape: str,
    nonbonded_cutoff_nm: float,
):

    print()
    print("=" * 80)
    print(
        f"LIGAND {ligand.ligand_id}"
    )
    print("=" * 80)

    print(
        f"Source: {ligand.source_filename}"
    )

    print(
        f"Title:  {ligand.title}"
    )

    print(
        f"Input:  {ligand.canonical_sdf}"
    )


    # --------------------------------------------------------
    # LIGAND
    # --------------------------------------------------------

    rdkit_mol = load_ligand_sdf(
        ligand.canonical_sdf
    )


    initial_conformer = (
        rdkit_mol.GetConformer()
    )


    initial_ligand_xyz = np.array(
        [
            list(
                initial_conformer.GetAtomPosition(
                    index
                )
            )
            for index
            in range(
                rdkit_mol.GetNumAtoms()
            )
        ]
    )


    offmol = rdkit_to_openff(
        rdkit_mol
    )


    # --------------------------------------------------------
    # FORCE FIELD
    # --------------------------------------------------------

    (
        forcefield,
        ff_selection,
        _ligand_generator,
    ) = create_forcefield(
        offmol=offmol,
        protein_forcefield=protein_forcefield,
        ligand_forcefield=ligand_forcefield,
        water_model=water_model,
    )

    solvation = make_solvation_config(
        mode=solvent_mode,
        padding_nm=padding_nm,
        ionic_strength_molar=ionic_strength_molar,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
        box_shape=box_shape,
        nonbonded_cutoff_nm=nonbonded_cutoff_nm,
    )


    # --------------------------------------------------------
    # COMPLEX
    # --------------------------------------------------------

    (
        modeller,
        receptor_atom_count,
        ligand_atom_count,
    ) = build_complex(
        receptor_pdb,
        offmol,
    )

    # Keep a solute-only topology so the public minimized complex remains
    # compatible with the v0.1.0 QC/analysis workflow even when bulk
    # solvent is requested.
    solute_modeller = Modeller(
        modeller.topology,
        modeller.positions,
    )
    solute_atom_count = sum(1 for _ in solute_modeller.topology.atoms())

    solvate_modeller(
        modeller=modeller,
        forcefield=forcefield,
        selection=ff_selection,
        config=solvation,
    )

    topology = modeller.topology
    positions = modeller.positions


    if (
        ligand_atom_count
        != rdkit_mol.GetNumAtoms()
    ):

        raise RuntimeError(
            "Ligand atom count mismatch after "
            "complex construction."
        )


    # --------------------------------------------------------
    # SYSTEM
    # --------------------------------------------------------

    print(
        "  Creating OpenMM system..."
    )


    creation_kwargs = system_creation_kwargs(
        solvation
    )

    system = forcefield.createSystem(
        topology,
        constraints=HBonds,
        **creation_kwargs,
    )


    add_backbone_restraints(
        system,
        topology,
        positions,
        receptor_atom_count,
    )


    (
        _,
        metal_restraint_records,
    ) = add_metal_restraints(
        system,
        topology,
        positions,
        receptor_atom_count,
        config,
    )


    # --------------------------------------------------------
    # SIMULATION
    # --------------------------------------------------------

    integrator = LangevinMiddleIntegrator(
        300
        * unit.kelvin,

        1.0
        / unit.picosecond,

        0.002
        * unit.picoseconds,
    )


    (
        platform,
        selected_platform,
    ) = select_platform(
        platform_name
    )


    properties = {}


    if selected_platform in {
        "CUDA",
        "OpenCL",
    }:

        properties[
            "Precision"
        ] = precision


    print(
        f"  OpenMM platform: {selected_platform}"
    )


    if properties:

        print(
            f"  Precision: {precision}"
        )


    simulation = Simulation(
        topology,
        system,
        integrator,
        platform,
        properties,
    )


    simulation.context.setPositions(
        positions
    )


    initial_energy = energy_kcal(
        simulation.context
    )


    print(
        "  Initial energy: "
        f"{initial_energy:.3f} kcal/mol"
    )


    initial_metal = report_metal_distances(
        positions,
        metal_restraint_records,
    )


    # --------------------------------------------------------
    # STAGED MINIMIZATION
    # --------------------------------------------------------

    for stage, k_value in enumerate(
        config.backbone_restraint_schedule_kcal_mol_A2,
        start=1,
    ):

        k_internal = (
            k_value
            * unit.kilocalorie_per_mole
            / unit.angstrom**2
        ).value_in_unit(
            unit.kilojoule_per_mole
            / unit.nanometer**2
        )


        simulation.context.setParameter(
            "k_backbone",
            k_internal,
        )


        before = energy_kcal(
            simulation.context
        )


        print(
            f"  Stage {stage}: "
            f"backbone k={k_value:.1f} "
            "kcal/mol/A^2"
        )

        print(
            f"    before: {before:.3f}"
        )


        simulation.minimizeEnergy(
            tolerance=(
                DEFAULT_MINIMIZATION_TOLERANCE
            ),
            maxIterations=max_iterations,
        )


        after = energy_kcal(
            simulation.context
        )


        print(
            f"    after:  {after:.3f}"
        )


    # --------------------------------------------------------
    # FINAL STATE
    # --------------------------------------------------------

    state = simulation.context.getState(
        getPositions=True,
        getEnergy=True,
    )


    final_positions = (
        state.getPositions()
    )


    final_energy = (
        state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilocalorie_per_mole
        )
    )


    print(
        "  Final energy: "
        f"{final_energy:.3f} kcal/mol"
    )


    final_metal = report_metal_distances(
        final_positions,
        metal_restraint_records,
    )


    # --------------------------------------------------------
    # LIGAND DISPLACEMENT
    # --------------------------------------------------------

    final_xyz_A = get_position_array_A(
        final_positions
    )


    final_ligand_xyz = (
        final_xyz_A[
            receptor_atom_count:
            receptor_atom_count
            + ligand_atom_count
        ]
    )


    if (
        final_ligand_xyz.shape
        != initial_ligand_xyz.shape
    ):

        raise RuntimeError(
            "Ligand coordinate shape mismatch."
        )


    atom_displacement = np.linalg.norm(
        final_ligand_xyz
        - initial_ligand_xyz,
        axis=1,
    )


    ligand_rms_displacement = float(
        np.sqrt(
            np.mean(
                atom_displacement**2
            )
        )
    )


    maximum_ligand_displacement = float(
        np.max(
            atom_displacement
        )
    )


    print(
        "  Ligand coordinate RMS displacement: "
        f"{ligand_rms_displacement:.3f} A"
    )

    print(
        "  Maximum ligand atom displacement: "
        f"{maximum_ligand_displacement:.3f} A"
    )


    # --------------------------------------------------------
    # PUBLIC OUTPUT
    # --------------------------------------------------------

    result_dir = (
        workspace.result_dir(
            ligand.ligand_id
        )
    )


    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    complex_pdb = result_dir / "complex_minimized.pdb"
    receptor_pdb_out = result_dir / "receptor_minimized.pdb"
    ligand_sdf = result_dir / "ligand_minimized.sdf"

    final_solute_positions = final_positions[:solute_atom_count]
    final_receptor_positions = final_positions[:receptor_atom_count]

    with open(complex_pdb, "w") as handle:
        PDBFile.writeFile(
            solute_modeller.topology,
            final_solute_positions,
            handle,
            keepIds=True,
        )

    with open(receptor_pdb_out, "w") as handle:
        PDBFile.writeFile(
            receptor_pdb.topology,
            final_receptor_positions,
            handle,
            keepIds=True,
        )

    solvated_pdb = None
    if solvation.mode == "explicit":
        solvated_dir = result_dir / "solvated"
        solvated_dir.mkdir(parents=True, exist_ok=True)
        solvated_pdb = solvated_dir / "complex_minimized_solvated.pdb"
        with open(solvated_pdb, "w") as handle:
            PDBFile.writeFile(
                topology,
                final_positions,
                handle,
                keepIds=True,
            )
        write_solvation_metadata(
            solvated_dir / "system_metadata.json",
            selection=ff_selection,
            config=solvation,
            topology=topology,
        )


    output_mol = Chem.Mol(
        rdkit_mol
    )


    conformer = (
        output_mol.GetConformer()
    )


    for index in range(
        output_mol.GetNumAtoms()
    ):

        (
            x,
            y,
            z,
        ) = final_ligand_xyz[
            index
        ]


        conformer.SetAtomPosition(
            index,
            (
                float(
                    x
                ),
                float(
                    y
                ),
                float(
                    z
                ),
            ),
        )


    # Preserve original SDF title where possible.
    output_mol.SetProp(
        "_Name",
        ligand.title,
    )


    output_mol.SetProp(
        "OpenMM_initial_energy_kcal_mol",
        f"{initial_energy:.6f}",
    )

    output_mol.SetProp(
        "OpenMM_final_energy_kcal_mol",
        f"{final_energy:.6f}",
    )

    output_mol.SetProp(
        "OpenMM_coordinate_RMS_displacement_A",
        f"{ligand_rms_displacement:.6f}",
    )


    writer = Chem.SDWriter(
        str(
            ligand_sdf
        )
    )


    writer.write(
        output_mol
    )

    writer.close()


    print(
        f"  Wrote complex: {complex_pdb}"
    )

    print(
        f"  Wrote ligand:  {ligand_sdf}"
    )
    print(f"  Wrote receptor: {receptor_pdb_out}")
    if solvated_pdb is not None:
        print(f"  Wrote solvated: {solvated_pdb}")


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    row = {
        "ligand_id":
            ligand.ligand_id,

        "internal_ordinal":
            ligand.internal_ordinal,

        "compound_id":
            ligand.compound_id,

        "pose_index":
            ligand.pose_index,

        "title":
            ligand.title,

        "source_filename":
            ligand.source_filename,

        "source_rank":
            ligand.source_rank,

        "input_sdf":
            str(
                ligand.canonical_sdf
            ),

        "minimized_sdf":
            str(
                ligand_sdf
            ),

        "complex_pdb": str(complex_pdb),
        "receptor_minimized_pdb": str(receptor_pdb_out),
        "solvated_complex_pdb": (str(solvated_pdb) if solvated_pdb else None),
        "protein_forcefield": ff_selection.protein.name,
        "ligand_forcefield": ff_selection.ligand.name,
        "water_model": ff_selection.water.name,
        "solvent_mode": solvation.mode,
        "padding_nm": solvation.padding_nm if solvation.mode == "explicit" else None,
        "ionic_strength_molar": solvation.ionic_strength_molar if solvation.mode == "explicit" else None,
        "box_shape": solvation.box_shape if solvation.mode == "explicit" else None,

        "ligand_atoms":
            ligand_atom_count,

        "ligand_heavy_atoms":
            rdkit_mol.GetNumHeavyAtoms(),

        "formal_charge":
            Chem.GetFormalCharge(
                rdkit_mol
            ),

        "initial_energy_kcal_mol":
            initial_energy,

        "final_energy_kcal_mol":
            final_energy,

        "energy_change_kcal_mol":
            final_energy
            - initial_energy,

        "ligand_RMS_displacement_A":
            ligand_rms_displacement,

        "ligand_max_atom_displacement_A":
            maximum_ligand_displacement,

        "openmm_platform":
            selected_platform,

        "openmm_precision":
            (
                precision
                if selected_platform
                in {
                    "CUDA",
                    "OpenCL",
                }
                else None
            ),
    }


    for key, value in initial_metal.items():

        row[
            "initial_"
            + key
        ] = value


    for key, value in final_metal.items():

        row[
            "final_"
            + key
        ] = value


    return row


# ============================================================
# SUMMARY UPSERT
# ============================================================

def upsert_summary(
    workspace: Workspace,
    rows: list[dict],
):

    if not rows:

        return


    new_rows = pd.DataFrame(
        rows
    )


    output_file = (
        workspace.minimization_summary()
    )


    if output_file.exists():

        existing = pd.read_csv(
            output_file
        )


        if (
            len(existing) > 0
            and
            "ligand_id"
            not in existing.columns
        ):

            raise RuntimeError(
                "Existing minimization summary does not "
                "contain ligand_id."
            )


        rerun_ids = set(
            new_rows[
                "ligand_id"
            ].astype(str)
        )


        existing = (
            existing[
                ~existing[
                    "ligand_id"
                ]
                .astype(str)
                .isin(
                    rerun_ids
                )
            ]
            .copy()
        )


        summary = pd.concat(
            [
                existing,
                new_rows,
            ],
            ignore_index=True,
            sort=False,
        )


    else:

        summary = (
            new_rows.copy()
        )


    summary = (
        summary
        .drop_duplicates(
            subset=[
                "ligand_id"
            ],
            keep="last",
        )
        .sort_values(
            "internal_ordinal"
        )
        .reset_index(
            drop=True
        )
    )


    preferred_front = [
        "ligand_id",
        "internal_ordinal",
        "compound_id",
        "pose_index",
        "title",
        "source_filename",
        "source_rank",
        "input_sdf",
        "minimized_sdf",
        "complex_pdb",
    ]


    front = [
        column
        for column
        in preferred_front
        if column
        in summary.columns
    ]


    remainder = [
        column
        for column
        in summary.columns
        if column
        not in front
    ]


    summary[
        front
        + remainder
    ].to_csv(
        output_file,
        index=False,
    )


# ============================================================
# WORKSPACE MINIMIZATION
# ============================================================

def minimize_workspace(
    workspace: Workspace,
    config: ReceptorConfig,
    ligand_ids: list[str] | None = None,
    platform_name: str = "auto",
    precision: str = "mixed",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    protein_forcefield: str = "ff14SB",
    ligand_forcefield: str = "openff-2.3.0",
    water_model: str = "tip3p",
    solvent_mode: str = "none",
    padding_nm: float = 1.0,
    ionic_strength_molar: float = 0.15,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    box_shape: str = "dodecahedron",
    nonbonded_cutoff_nm: float = 1.0,
    dry_run: bool = False,
):

    workspace.ensure_results_dir()


    all_records = (
        workspace.ligand_records()
    )


    if ligand_ids:

        requested = {
            str(
                value
            ).upper()
            for value
            in ligand_ids
        }


        records = [
            ligand
            for ligand
            in all_records
            if ligand.ligand_id.upper()
            in requested
        ]


        found = {
            ligand.ligand_id.upper()
            for ligand
            in records
        }


        missing = (
            requested
            - found
        )


        if missing:

            raise RuntimeError(
                "Unknown ligand IDs:\n  "
                + "\n  ".join(
                    sorted(
                        missing
                    )
                )
            )


    else:

        records = (
            all_records
        )


    print()
    print("=" * 80)
    print("DOCK-POSTPROCESS NATIVE OPENMM MINIMIZATION")
    print("=" * 80)

    print(
        f"Workspace: {workspace.root}"
    )

    print(
        f"Receptor:  {workspace.receptor}"
    )

    print(
        f"Ligands:   {len(records)}"
    )


    describe_receptor_config(
        config
    )

    # Resolve and display the simulation settings once.  Ligand-specific
    # template registration occurs later for each molecule.
    from dockpost.forcefields import resolve_forcefield_selection
    ff_preview = resolve_forcefield_selection(
        protein_forcefield=protein_forcefield,
        ligand_forcefield=ligand_forcefield,
        water_model=water_model,
    )
    solv_preview = make_solvation_config(
        mode=solvent_mode,
        padding_nm=padding_nm,
        ionic_strength_molar=ionic_strength_molar,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
        box_shape=box_shape,
        nonbonded_cutoff_nm=nonbonded_cutoff_nm,
    )
    describe_forcefield_selection(ff_preview)
    describe_solvation(ff_preview, solv_preview)

    if dry_run:

        print()
        print(
            "Ligands selected:"
        )


        for ligand in records:

            print(
                f"  {ligand.ligand_id} "
                f"<- {ligand.source_filename}"
            )


        print()
        print(
            "DRY RUN COMPLETE"
        )

        return


    # --------------------------------------------------------
    # SAVE EXACT CONFIGURATION USED
    # --------------------------------------------------------

    save_receptor_config(
        config,
        workspace.results_dir
        / "receptor_config_used.json",
    )


    # --------------------------------------------------------
    # LOAD RECEPTOR ONCE
    # --------------------------------------------------------

    print()
    print(
        f"Loading receptor: {workspace.receptor}"
    )


    receptor = load_receptor(
        workspace.receptor,
        config,
    )


    receptor_atoms = sum(
        1
        for _
        in receptor.topology.atoms()
    )


    chains = [
        chain.id
        for chain
        in receptor.topology.chains()
    ]


    print(
        f"Receptor atoms: {receptor_atoms}"
    )

    print(
        "Receptor chains: "
        + ", ".join(
            chains
        )
    )


    rows = []


    for ligand in records:

        try:

            row = minimize_ligand(
                workspace=workspace,
                ligand=ligand,
                receptor_pdb=receptor,
                config=config,
                platform_name=platform_name,
                precision=precision,
                max_iterations=max_iterations,
                protein_forcefield=protein_forcefield,
                ligand_forcefield=ligand_forcefield,
                water_model=water_model,
                solvent_mode=solvent_mode,
                padding_nm=padding_nm,
                ionic_strength_molar=ionic_strength_molar,
                positive_ion=positive_ion,
                negative_ion=negative_ion,
                box_shape=box_shape,
                nonbonded_cutoff_nm=nonbonded_cutoff_nm,
            )


            row[
                "status"
            ] = "PASS"

            row[
                "error"
            ] = ""


        except Exception as exc:

            print()
            print(
                f"ERROR on {ligand.ligand_id}: {exc}"
            )


            traceback.print_exc()


            row = {
                "ligand_id":
                    ligand.ligand_id,

                "internal_ordinal":
                    ligand.internal_ordinal,

                "compound_id":
                    ligand.compound_id,

                "pose_index":
                    ligand.pose_index,

                "title":
                    ligand.title,

                "source_filename":
                    ligand.source_filename,

                "source_rank":
                    ligand.source_rank,

                "input_sdf":
                    str(
                        ligand.canonical_sdf
                    ),

                "status":
                    "FAIL",

                "error":
                    str(
                        exc
                    ),
            }


        rows.append(
            row
        )


        # Preserve progress after every ligand.
        upsert_summary(
            workspace,
            [
                row
            ],
        )


    print()
    print("=" * 80)
    print("MINIMIZATION COMPLETE")
    print("=" * 80)


    summary = pd.read_csv(
        workspace.minimization_summary()
    )


    selected_ids = {
        ligand.ligand_id
        for ligand
        in records
    }


    display = summary[
        summary[
            "ligand_id"
        ].isin(
            selected_ids
        )
    ]


    display_columns = [
        column
        for column
        in [
            "ligand_id",
            "status",
            "initial_energy_kcal_mol",
            "final_energy_kcal_mol",
            "energy_change_kcal_mol",
            "ligand_RMS_displacement_A",
            "ligand_max_atom_displacement_A",
            "error",
        ]
        if column
        in display.columns
    ]


    print(
        display[
            display_columns
        ]
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


    print()
    print(
        "Wrote:"
    )

    print(
        f"  {workspace.minimization_summary()}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Native generalized OpenMM minimization."
        )
    )


    parser.add_argument(
        "--results",
        required=True,
        help=(
            "Generalized analysis workspace "
            "or its results directory."
        ),
    )


    parser.add_argument(
        "--receptor-config",
        default=None,
        help=(
            "Optional receptor chemistry JSON configuration. "
            "If omitted, generic standard-protein defaults "
            "with no metal restraints are used."
        ),
    )


    parser.add_argument(
        "--ligand",
        action="append",
        dest="ligands",
        help=(
            "Process only this L#### ligand ID. "
            "May be repeated."
        ),
    )


    parser.add_argument(
        "--platform",
        default="auto",
        help=(
            "OpenMM platform. Default: auto "
            "(CUDA, then OpenCL, then CPU)."
        ),
    )


    parser.add_argument(
        "--precision",
        choices=[
            "single",
            "mixed",
            "double",
        ],
        default="mixed",
        help=(
            "CUDA/OpenCL precision. Default: mixed."
        ),
    )


    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=(
            "Maximum minimizer iterations per restraint stage. "
            "Default: 5000."
        ),
    )


    parser.add_argument("--protein-forcefield", default="ff14SB", help="Protein force-field preset. Default: ff14SB.")
    parser.add_argument("--ligand-forcefield", default="openff-2.3.0", help="Installed OpenFF or GAFF force field. Default: openff-2.3.0.")
    parser.add_argument("--solvent", choices=["none", "explicit"], default="none", help="Bulk-solvent mode. Default: none.")
    parser.add_argument("--water-model", default="tip3p", help="Water/ion parameter model. Default: tip3p.")
    parser.add_argument("--padding-nm", type=float, default=1.0, help="Explicit-solvent padding in nm. Default: 1.0.")
    parser.add_argument("--ionic-strength", type=float, default=0.15, help="Salt concentration in mol/L for explicit solvent. Default: 0.15.")
    parser.add_argument("--positive-ion", default="Na+", help="Positive ion for explicit solvent. Default: Na+.")
    parser.add_argument("--negative-ion", default="Cl-", help="Negative ion for explicit solvent. Default: Cl-.")
    parser.add_argument("--box-shape", choices=["cube", "dodecahedron", "octahedron"], default="dodecahedron", help="Periodic box shape. Default: dodecahedron.")
    parser.add_argument("--nonbonded-cutoff-nm", type=float, default=1.0, help="PME real-space cutoff in nm. Default: 1.0.")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs/configuration and show selected "
            "ligands without running OpenMM."
        ),
    )


    args = parser.parse_args()


    workspace = Workspace.from_path(
        Path(
            args.results
        )
    )


    config_file = (
        None
        if args.receptor_config is None
        else Path(
            args.receptor_config
        )
    )


    config = load_receptor_config(
        config_file
    )


    minimize_workspace(
        workspace=workspace,
        config=config,
        ligand_ids=args.ligands,
        platform_name=args.platform,
        precision=args.precision,
        max_iterations=args.max_iterations,
        protein_forcefield=args.protein_forcefield,
        ligand_forcefield=args.ligand_forcefield,
        water_model=args.water_model,
        solvent_mode=args.solvent,
        padding_nm=args.padding_nm,
        ionic_strength_molar=args.ionic_strength,
        positive_ion=args.positive_ion,
        negative_ion=args.negative_ion,
        box_shape=args.box_shape,
        nonbonded_cutoff_nm=args.nonbonded_cutoff_nm,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
