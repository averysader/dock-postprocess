#!/usr/bin/env python3

"""Molecular dynamics for dock-postprocess.

The ``dock-md`` command extends the minimized docking workflow through:

1. system reconstruction from the minimized receptor and ligand,
2. optional explicit solvation and ions,
3. a short pre-MD energy minimization,
4. controlled NVT heating,
5. NVT or NPT equilibration,
6. NVT or NPT production sampling.

NPT requires an explicit periodic solvent box.  NVT can be run either with
explicit periodic solvent or with the legacy dry/NoCutoff model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import math
import sys
import traceback

from rdkit import Chem

from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    XmlSerializer,
    unit,
)
from openmm.app import (
    CheckpointReporter,
    DCDReporter,
    HBonds,
    PDBFile,
    Simulation,
    StateDataReporter,
)

from dockpost.workspace import Workspace
from dockpost.receptor_config import (
    ReceptorConfig,
    describe_receptor_config,
    load_receptor_config,
    save_receptor_config,
)
from dockpost.forcefields import (
    create_forcefield,
    describe_forcefield_selection,
    resolve_forcefield_selection,
)
from dockpost.solvation import (
    describe_solvation,
    make_solvation_config,
    require_periodic_for_npt,
    solvate_modeller,
    system_creation_kwargs,
    write_solvation_metadata,
)
from dockpost.minimize import (
    add_backbone_restraints,
    add_metal_restraints,
    build_complex,
    energy_kcal,
    load_ligand_sdf,
    load_receptor,
    rdkit_to_openff,
    report_metal_distances,
    select_platform,
)


@dataclass(frozen=True)
class MDConfig:
    ensemble: str = "npt"
    temperature_K: float = 300.0
    pressure_atm: float = 1.0
    start_temperature_K: float = 50.0
    heat_ps: float = 100.0
    equilibration_ps: float = 500.0
    production_ps: float = 1000.0
    timestep_fs: float = 2.0
    friction_per_ps: float = 1.0
    report_interval_ps: float = 10.0
    trajectory_interval_ps: float = 10.0
    checkpoint_interval_ps: float = 100.0
    barostat_frequency: int = 25
    heat_restraint_k_kcal_mol_A2: float = 1.0
    equilibration_restraint_k_kcal_mol_A2: float = 1.0
    production_restraint_k_kcal_mol_A2: float = 0.0
    pre_md_minimize_iterations: int = 1000
    seed: int = 2026


def validate_md_config(config: MDConfig) -> MDConfig:
    ensemble = str(config.ensemble).strip().lower()
    if ensemble not in {"nvt", "npt"}:
        raise ValueError("ensemble must be 'nvt' or 'npt'.")
    for name, value in [
        ("temperature_K", config.temperature_K),
        ("start_temperature_K", config.start_temperature_K),
        ("timestep_fs", config.timestep_fs),
        ("friction_per_ps", config.friction_per_ps),
        ("report_interval_ps", config.report_interval_ps),
        ("trajectory_interval_ps", config.trajectory_interval_ps),
        ("checkpoint_interval_ps", config.checkpoint_interval_ps),
    ]:
        if float(value) <= 0:
            raise ValueError(f"{name} must be greater than zero.")
    for name, value in [
        ("heat_ps", config.heat_ps),
        ("equilibration_ps", config.equilibration_ps),
        ("production_ps", config.production_ps),
        ("pressure_atm", config.pressure_atm),
    ]:
        if float(value) < 0:
            raise ValueError(f"{name} cannot be negative.")
    if config.production_ps <= 0:
        raise ValueError("production_ps must be greater than zero.")
    if config.barostat_frequency <= 0:
        raise ValueError("barostat_frequency must be greater than zero.")
    if config.pre_md_minimize_iterations < 0:
        raise ValueError("pre_md_minimize_iterations cannot be negative.")
    return MDConfig(**{**asdict(config), "ensemble": ensemble})


def steps_for_ps(duration_ps: float, timestep_fs: float) -> int:
    if duration_ps <= 0:
        return 0
    return max(1, int(round(duration_ps * 1000.0 / timestep_fs)))


def interval_steps(interval_ps: float, timestep_fs: float) -> int:
    return max(1, steps_for_ps(interval_ps, timestep_fs))


def set_backbone_k(context, value_kcal_mol_A2: float):
    internal = (
        value_kcal_mol_A2
        * unit.kilocalorie_per_mole
        / unit.angstrom**2
    ).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer**2
    )
    context.setParameter("k_backbone", internal)


def write_pdb(path: Path, topology, positions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        PDBFile.writeFile(topology, positions, handle, keepIds=True)


def get_positions(simulation: Simulation):
    return simulation.context.getState(getPositions=True).getPositions()


def run_heating(
    simulation: Simulation,
    integrator: LangevinMiddleIntegrator,
    config: MDConfig,
):
    total_steps = steps_for_ps(config.heat_ps, config.timestep_fs)
    if total_steps == 0:
        integrator.setTemperature(config.temperature_K * unit.kelvin)
        return

    # Twenty small temperature increments are enough to avoid an abrupt
    # thermostat jump while keeping command overhead negligible.
    stages = min(20, total_steps)
    base = total_steps // stages
    remainder = total_steps % stages

    for stage in range(1, stages + 1):
        fraction = stage / stages
        temperature = (
            config.start_temperature_K
            + fraction * (config.temperature_K - config.start_temperature_K)
        )
        integrator.setTemperature(temperature * unit.kelvin)
        nsteps = base + (1 if stage <= remainder else 0)
        simulation.step(nsteps)

    integrator.setTemperature(config.temperature_K * unit.kelvin)


def locate_minimized_inputs(workspace: Workspace, ligand_id: str):
    result_dir = workspace.result_dir(ligand_id)
    ligand_sdf = result_dir / "ligand_minimized.sdf"
    receptor_pdb = result_dir / "receptor_minimized.pdb"

    if not ligand_sdf.exists():
        raise FileNotFoundError(
            f"Missing minimized ligand for {ligand_id}: {ligand_sdf}. "
            "Run dock-minimize first."
        )

    if not receptor_pdb.exists():
        # v0.1.0 minimizations did not write a receptor-only minimized PDB.
        # Falling back keeps old workspaces usable, while v0.2.0 runs use the
        # minimized receptor automatically.
        receptor_pdb = workspace.receptor

    return result_dir, receptor_pdb, ligand_sdf


def run_md_ligand(
    workspace: Workspace,
    ligand,
    receptor_config: ReceptorConfig,
    md_config: MDConfig,
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
    platform_name: str,
    precision: str,
):
    result_dir, receptor_path, ligand_path = locate_minimized_inputs(
        workspace, ligand.ligand_id
    )
    md_dir = result_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"MD: {ligand.ligand_id}")
    print("=" * 80)
    print(f"Receptor: {receptor_path}")
    print(f"Ligand:   {ligand_path}")

    rdkit_mol = load_ligand_sdf(ligand_path)
    offmol = rdkit_to_openff(rdkit_mol)
    receptor = load_receptor(receptor_path, receptor_config)

    forcefield, ff_selection, _generator = create_forcefield(
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
    require_periodic_for_npt(md_config.ensemble, solvation)

    modeller, receptor_atom_count, ligand_atom_count = build_complex(
        receptor, offmol
    )
    solute_atom_count = sum(1 for _ in modeller.topology.atoms())
    # Preserve the pre-solvation topology for a final solute-only snapshot.
    from openmm.app import Modeller
    solute_modeller = Modeller(modeller.topology, modeller.positions)

    solvate_modeller(
        modeller=modeller,
        forcefield=forcefield,
        selection=ff_selection,
        config=solvation,
    )

    topology = modeller.topology
    positions = modeller.positions
    creation_kwargs = system_creation_kwargs(solvation)
    system = forcefield.createSystem(
        topology,
        constraints=HBonds,
        rigidWater=True,
        **creation_kwargs,
    )

    add_backbone_restraints(
        system,
        topology,
        positions,
        receptor_atom_count,
    )
    _, metal_records = add_metal_restraints(
        system,
        topology,
        positions,
        receptor_atom_count,
        receptor_config,
    )

    barostat = None
    if md_config.ensemble == "npt":
        barostat = MonteCarloBarostat(
            md_config.pressure_atm * unit.atmosphere,
            md_config.temperature_K * unit.kelvin,
            md_config.barostat_frequency,
        )
        barostat.setRandomNumberSeed(md_config.seed + 1)
        # Heating is always NVT.  The barostat is enabled only afterward.
        barostat.setFrequency(0)
        system.addForce(barostat)

    integrator = LangevinMiddleIntegrator(
        md_config.start_temperature_K * unit.kelvin,
        md_config.friction_per_ps / unit.picosecond,
        md_config.timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(md_config.seed)

    platform, selected_platform = select_platform(platform_name)
    properties = {}
    if selected_platform in {"CUDA", "OpenCL"}:
        properties["Precision"] = precision

    simulation = Simulation(
        topology,
        system,
        integrator,
        platform,
        properties,
    )
    simulation.context.setPositions(positions)
    set_backbone_k(
        simulation.context,
        md_config.heat_restraint_k_kcal_mol_A2,
    )

    print(f"Platform: {selected_platform}")
    if properties:
        print(f"Precision: {precision}")
    print(f"Initial potential energy: {energy_kcal(simulation.context):.3f} kcal/mol")

    if md_config.pre_md_minimize_iterations > 0:
        print("Pre-MD minimization...")
        simulation.minimizeEnergy(
            tolerance=1.0 * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=md_config.pre_md_minimize_iterations,
        )

    initial_positions = get_positions(simulation)
    write_pdb(md_dir / "initial.pdb", topology, initial_positions)

    simulation.context.setVelocitiesToTemperature(
        md_config.start_temperature_K * unit.kelvin,
        md_config.seed,
    )

    print(
        f"Heating: {md_config.start_temperature_K:g} -> "
        f"{md_config.temperature_K:g} K over {md_config.heat_ps:g} ps (NVT)"
    )
    run_heating(simulation, integrator, md_config)
    heated_positions = get_positions(simulation)
    write_pdb(md_dir / "heated.pdb", topology, heated_positions)

    if barostat is not None:
        barostat.setFrequency(md_config.barostat_frequency)

    set_backbone_k(
        simulation.context,
        md_config.equilibration_restraint_k_kcal_mol_A2,
    )
    equil_steps = steps_for_ps(md_config.equilibration_ps, md_config.timestep_fs)
    print(
        f"Equilibration: {md_config.equilibration_ps:g} ps "
        f"({md_config.ensemble.upper()})"
    )
    if equil_steps:
        simulation.step(equil_steps)
    equilibrated_positions = get_positions(simulation)
    write_pdb(md_dir / "equilibrated.pdb", topology, equilibrated_positions)

    set_backbone_k(
        simulation.context,
        md_config.production_restraint_k_kcal_mol_A2,
    )

    state_csv = md_dir / "production_state.csv"
    trajectory = md_dir / "production.dcd"
    checkpoint = md_dir / "checkpoint.chk"

    state_kwargs = dict(
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        progress=True,
        remainingTime=True,
        speed=True,
        totalSteps=steps_for_ps(md_config.production_ps, md_config.timestep_fs),
        separator=",",
    )
    if solvation.mode == "explicit":
        state_kwargs.update(volume=True, density=True)

    simulation.reporters.append(
        StateDataReporter(
            str(state_csv),
            interval_steps(md_config.report_interval_ps, md_config.timestep_fs),
            **state_kwargs,
        )
    )
    simulation.reporters.append(
        DCDReporter(
            str(trajectory),
            interval_steps(md_config.trajectory_interval_ps, md_config.timestep_fs),
            enforcePeriodicBox=(solvation.mode == "explicit"),
        )
    )
    simulation.reporters.append(
        CheckpointReporter(
            str(checkpoint),
            interval_steps(md_config.checkpoint_interval_ps, md_config.timestep_fs),
        )
    )

    production_steps = steps_for_ps(md_config.production_ps, md_config.timestep_fs)
    print(
        f"Production: {md_config.production_ps:g} ps "
        f"({md_config.ensemble.upper()}, {production_steps:,} steps)"
    )
    simulation.step(production_steps)

    final_state = simulation.context.getState(getPositions=True, getEnergy=True)
    final_positions = final_state.getPositions()
    final_energy = final_state.getPotentialEnergy().value_in_unit(
        unit.kilocalorie_per_mole
    )
    final_metal = report_metal_distances(final_positions, metal_records)

    write_pdb(md_dir / "final.pdb", topology, final_positions)
    write_pdb(
        md_dir / "final_solute.pdb",
        solute_modeller.topology,
        final_positions[:solute_atom_count],
    )
    simulation.saveCheckpoint(str(checkpoint))

    (md_dir / "system.xml").write_text(XmlSerializer.serialize(system))

    md_json = asdict(md_config)
    md_json.update(
        {
            "ligand_id": ligand.ligand_id,
            "protein_forcefield": ff_selection.protein.name,
            "ligand_forcefield": ff_selection.ligand.name,
            "water_model": ff_selection.water.name,
            "solvent_mode": solvation.mode,
            "platform": selected_platform,
            "precision": precision if properties else None,
        }
    )
    (md_dir / "md_config.json").write_text(
        json.dumps(md_json, indent=2, sort_keys=True) + "\n"
    )
    write_solvation_metadata(
        md_dir / "system_metadata.json",
        selection=ff_selection,
        config=solvation,
        topology=topology,
    )
    save_receptor_config(receptor_config, md_dir / "receptor_config_used.json")

    summary = {
        "ligand_id": ligand.ligand_id,
        "status": "PASS",
        "ensemble": md_config.ensemble,
        "temperature_K": md_config.temperature_K,
        "pressure_atm": md_config.pressure_atm if md_config.ensemble == "npt" else None,
        "production_ps": md_config.production_ps,
        "final_potential_energy_kcal_mol": final_energy,
        "platform": selected_platform,
        "precision": precision if properties else None,
        "trajectory": str(trajectory),
        "state_csv": str(state_csv),
        "checkpoint": str(checkpoint),
        "final_pdb": str(md_dir / "final.pdb"),
        "final_solute_pdb": str(md_dir / "final_solute.pdb"),
        "metal_distances_A": final_metal,
    }
    (md_dir / "md_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(f"Final potential energy: {final_energy:.3f} kcal/mol")
    print(f"Wrote: {md_dir}")
    return summary


def run_md_workspace(
    workspace: Workspace,
    receptor_config: ReceptorConfig,
    md_config: MDConfig,
    ligand_ids: list[str] | None = None,
    protein_forcefield: str = "ff14SB",
    ligand_forcefield: str = "openff-2.3.0",
    water_model: str = "tip3p",
    solvent_mode: str = "explicit",
    padding_nm: float = 1.0,
    ionic_strength_molar: float = 0.15,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    box_shape: str = "dodecahedron",
    nonbonded_cutoff_nm: float = 1.0,
    platform_name: str = "auto",
    precision: str = "mixed",
    dry_run: bool = False,
):
    md_config = validate_md_config(md_config)
    solvation = make_solvation_config(
        mode=solvent_mode,
        padding_nm=padding_nm,
        ionic_strength_molar=ionic_strength_molar,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
        box_shape=box_shape,
        nonbonded_cutoff_nm=nonbonded_cutoff_nm,
    )
    require_periodic_for_npt(md_config.ensemble, solvation)
    ff_preview = resolve_forcefield_selection(
        protein_forcefield=protein_forcefield,
        ligand_forcefield=ligand_forcefield,
        water_model=water_model,
    )

    records = workspace.ligand_records()
    if ligand_ids:
        requested = {x.upper() for x in ligand_ids}
        records = [r for r in records if r.ligand_id.upper() in requested]
        found = {r.ligand_id.upper() for r in records}
        missing = requested - found
        if missing:
            raise RuntimeError("Unknown ligand IDs: " + ", ".join(sorted(missing)))

    print()
    print("=" * 80)
    print("DOCK-POSTPROCESS MOLECULAR DYNAMICS")
    print("=" * 80)
    print(f"Workspace: {workspace.root}")
    print(f"Ligands:   {len(records)}")
    print(f"Ensemble:  {md_config.ensemble.upper()}")
    describe_receptor_config(receptor_config)
    describe_forcefield_selection(ff_preview)
    describe_solvation(ff_preview, solvation)
    print()
    print("MD protocol:")
    print(
        f"  heat {md_config.start_temperature_K:g}->{md_config.temperature_K:g} K: "
        f"{md_config.heat_ps:g} ps NVT"
    )
    print(f"  equilibrate: {md_config.equilibration_ps:g} ps {md_config.ensemble.upper()}")
    print(f"  production:  {md_config.production_ps:g} ps {md_config.ensemble.upper()}")
    print(f"  timestep:    {md_config.timestep_fs:g} fs")

    if dry_run:
        print("\nLigands selected:")
        for ligand in records:
            result_dir, receptor_path, ligand_path = locate_minimized_inputs(
                workspace, ligand.ligand_id
            )
            print(f"  {ligand.ligand_id}: {receptor_path.name} + {ligand_path.name}")
        print("\nDRY RUN COMPLETE")
        return []

    summaries = []
    for ligand in records:
        try:
            summaries.append(
                run_md_ligand(
                    workspace=workspace,
                    ligand=ligand,
                    receptor_config=receptor_config,
                    md_config=md_config,
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
                    platform_name=platform_name,
                    precision=precision,
                )
            )
        except Exception as exc:
            traceback.print_exc()
            summaries.append(
                {
                    "ligand_id": ligand.ligand_id,
                    "status": "FAIL",
                    "error": str(exc),
                }
            )

    summary_path = workspace.results_dir / "md_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print("\n" + "=" * 80)
    print("MD COMPLETE")
    print("=" * 80)
    print(f"Wrote: {summary_path}")
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Heat, equilibrate, and sample dock-postprocess complexes with OpenMM."
    )
    parser.add_argument("--results", required=True, help="Generalized analysis workspace or results directory.")
    parser.add_argument("--receptor-config", default=None, help="Optional receptor chemistry JSON configuration.")
    parser.add_argument("--ligand", action="append", dest="ligands", help="Process only this L#### ligand ID; repeatable.")

    parser.add_argument("--ensemble", choices=["nvt", "npt"], default="npt", help="Production ensemble. Default: npt.")
    parser.add_argument("--temperature", type=float, default=300.0, help="Target temperature in K. Default: 300.")
    parser.add_argument("--pressure", type=float, default=1.0, help="NPT pressure in atm. Default: 1.0.")
    parser.add_argument("--start-temperature", type=float, default=50.0, help="Heating start temperature in K. Default: 50.")
    parser.add_argument("--heat-ps", type=float, default=100.0, help="NVT heating duration in ps. Default: 100.")
    parser.add_argument("--equilibration-ps", type=float, default=500.0, help="Equilibration duration in ps. Default: 500.")
    parser.add_argument("--production-ps", type=float, default=1000.0, help="Production duration in ps. Default: 1000 (1 ns).")
    parser.add_argument("--production-ns", type=float, default=None, help="Production duration in ns; overrides --production-ps.")
    parser.add_argument("--timestep-fs", type=float, default=2.0, help="Integrator timestep in fs. Default: 2.")
    parser.add_argument("--friction", type=float, default=1.0, help="Langevin friction in 1/ps. Default: 1.")
    parser.add_argument("--report-interval-ps", type=float, default=10.0, help="State-data interval in ps. Default: 10.")
    parser.add_argument("--trajectory-interval-ps", type=float, default=10.0, help="DCD frame interval in ps. Default: 10.")
    parser.add_argument("--checkpoint-interval-ps", type=float, default=100.0, help="Checkpoint interval in ps. Default: 100.")
    parser.add_argument("--barostat-frequency", type=int, default=25, help="Monte Carlo barostat interval in steps. Default: 25.")
    parser.add_argument("--heat-restraint-k", type=float, default=1.0, help="Backbone restraint during heating, kcal/mol/A^2. Default: 1.")
    parser.add_argument("--equil-restraint-k", type=float, default=1.0, help="Backbone restraint during equilibration, kcal/mol/A^2. Default: 1.")
    parser.add_argument("--production-restraint-k", type=float, default=0.0, help="Backbone restraint during production, kcal/mol/A^2. Default: 0.")
    parser.add_argument("--pre-md-minimize-iterations", type=int, default=1000, help="Pre-MD minimization iterations. Default: 1000.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed. Default: 2026.")

    parser.add_argument("--protein-forcefield", default="ff14SB", help="Protein force-field preset. Default: ff14SB.")
    parser.add_argument("--ligand-forcefield", default="openff-2.3.0", help="Installed OpenFF or GAFF force field. Default: openff-2.3.0.")
    parser.add_argument("--solvent", choices=["none", "explicit"], default="explicit", help="Bulk solvent mode. Default: explicit.")
    parser.add_argument("--water-model", default="tip3p", help="Water model. Default: tip3p.")
    parser.add_argument("--padding-nm", type=float, default=1.0, help="Solvent padding in nm. Default: 1.0.")
    parser.add_argument("--ionic-strength", type=float, default=0.15, help="Salt concentration in mol/L. Default: 0.15.")
    parser.add_argument("--positive-ion", default="Na+", help="Positive ion. Default: Na+.")
    parser.add_argument("--negative-ion", default="Cl-", help="Negative ion. Default: Cl-.")
    parser.add_argument("--box-shape", choices=["cube", "dodecahedron", "octahedron"], default="dodecahedron", help="Periodic box shape. Default: dodecahedron.")
    parser.add_argument("--nonbonded-cutoff-nm", type=float, default=1.0, help="PME cutoff in nm. Default: 1.0.")
    parser.add_argument("--platform", default="auto", help="OpenMM platform. Default: auto.")
    parser.add_argument("--precision", choices=["single", "mixed", "double"], default="mixed", help="GPU precision. Default: mixed.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and inputs without running OpenMM.")

    args = parser.parse_args()
    production_ps = args.production_ps if args.production_ns is None else args.production_ns * 1000.0
    md_config = MDConfig(
        ensemble=args.ensemble,
        temperature_K=args.temperature,
        pressure_atm=args.pressure,
        start_temperature_K=args.start_temperature,
        heat_ps=args.heat_ps,
        equilibration_ps=args.equilibration_ps,
        production_ps=production_ps,
        timestep_fs=args.timestep_fs,
        friction_per_ps=args.friction,
        report_interval_ps=args.report_interval_ps,
        trajectory_interval_ps=args.trajectory_interval_ps,
        checkpoint_interval_ps=args.checkpoint_interval_ps,
        barostat_frequency=args.barostat_frequency,
        heat_restraint_k_kcal_mol_A2=args.heat_restraint_k,
        equilibration_restraint_k_kcal_mol_A2=args.equil_restraint_k,
        production_restraint_k_kcal_mol_A2=args.production_restraint_k,
        pre_md_minimize_iterations=args.pre_md_minimize_iterations,
        seed=args.seed,
    )

    workspace = Workspace.from_path(Path(args.results))
    receptor_config = load_receptor_config(
        None if args.receptor_config is None else Path(args.receptor_config)
    )
    run_md_workspace(
        workspace=workspace,
        receptor_config=receptor_config,
        md_config=md_config,
        ligand_ids=args.ligands,
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
        platform_name=args.platform,
        precision=args.precision,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
