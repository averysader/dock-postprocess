#!/usr/bin/env python3

"""
dock-strain

Estimate ligand conformational strain for minimized docking poses.

Workflow
--------
1. Read each *_ligand_minimized.sdf as the bound pose.
2. Preserve the exact bound coordinates and evaluate their isolated-ligand
   OpenFF/OpenMM potential energy without minimization.
3. Relax that bound pose as an isolated ligand.
4. Generate an independent conformer ensemble with RDKit ETKDGv3.
5. Minimize every conformer using the exact same OpenFF/OpenMM Hamiltonian.
6. Compare the bound-pose energy against the lowest minimized conformer.
7. Calculate RMSD of minimized solution conformers to the bound pose.
8. Write per-ligand conformer ensembles, tables, plots, and a master summary.

Important
---------
The current implementation evaluates intrinsic molecular mechanics
conformational strain. It does NOT include explicit or implicit solvent and
does NOT represent a rigorous aqueous conformational free energy.

All energies being compared use the same OpenFF 2.3.0 force field.
"""

from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from openff.toolkit import Molecule

from openmm import unit
from openmm import Platform
from openmm import VerletIntegrator
from openmm import Context

from openmm.app import ForceField
from openmmforcefields.generators import SMIRNOFFTemplateGenerator


# ============================================================
# COMMAND LINE
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Estimate conformational strain of minimized docking poses "
        "using RDKit + OpenFF 2.3.0 + OpenMM."
    )
)

parser.add_argument(
    "top_n",
    nargs="?",
    type=int,
    default=10,
    help="Analyze ranks 1 through N. Default: 10",
)

parser.add_argument(
    "--conformers",
    type=int,
    default=300,
    help="Number of ETKDG starting conformers per ligand. Default: 300",
)

parser.add_argument(
    "--temperature",
    type=float,
    default=298.15,
    help="Temperature for energy-based Boltzmann weights. Default: 298.15 K",
)

parser.add_argument(
    "--seed",
    type=int,
    default=20260904,
    help="RDKit random seed. Default: 20260904",
)

parser.add_argument(
    "--prune-rms",
    type=float,
    default=0.35,
    help=(
        "ETKDG heavy-atom RMS pruning threshold in Angstrom. "
        "Default: 0.35"
    ),
)

parser.add_argument(
    "--minimize-tolerance",
    type=float,
    default=1.0,
    help=(
        "OpenMM minimizer tolerance in kJ/mol/nm. "
        "Default: 1.0"
    ),
)

parser.add_argument(
    "--max-iterations",
    type=int,
    default=5000,
    help="Maximum OpenMM minimization iterations. Default: 5000",
)

args = parser.parse_args()


TOP_N = args.top_n
N_CONFORMERS = args.conformers
TEMPERATURE_K = args.temperature
RANDOM_SEED = args.seed
PRUNE_RMS_A = args.prune_rms
MIN_TOLERANCE = args.minimize_tolerance
MAX_ITERATIONS = args.max_iterations


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    f"top{TOP_N}_openmm_minimized"
)

OUT_DIR = (
    BASE_DIR
    / "ligand_strain"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_FILE = (
    OUT_DIR
    / "strain_summary.csv"
)


# ============================================================
# FORCE FIELD / PLATFORM
# ============================================================

OPENFF_FORCEFIELD = "openff-2.3.0"

PLATFORM_NAME = "CUDA"

PLATFORM_PROPERTIES = {
    "Precision": "mixed",
}


# ============================================================
# CONSTANTS
# ============================================================

R_KCAL = 0.00198720425864083

RT_KCAL = (
    R_KCAL
    * TEMPERATURE_K
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_single_sdf(
    filename,
):

    supplier = Chem.SDMolSupplier(
        str(filename),
        removeHs=False,
        sanitize=True,
    )

    mol = next(
        (
            m
            for m in supplier
            if m is not None
        ),
        None,
    )

    if mol is None:

        raise RuntimeError(
            f"Could not read {filename}"
        )

    if mol.GetNumConformers() == 0:

        raise RuntimeError(
            f"No coordinates in {filename}"
        )

    return mol


def find_rank_sdf(
    rank,
):

    rank_dir = (
        BASE_DIR
        / f"{rank:03d}"
    )

    if not rank_dir.exists():

        raise RuntimeError(
            f"Missing {rank_dir}"
        )

    matches = list(
        rank_dir.glob(
            "*_ligand_minimized.sdf"
        )
    )

    if len(matches) != 1:

        raise RuntimeError(
            f"{rank_dir}: expected one minimized ligand SDF, "
            f"found {len(matches)}"
        )

    return matches[0]


def coordinates_from_conformer(
    mol,
    conf_id,
):

    conf = mol.GetConformer(
        conf_id
    )

    xyz = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(
                mol.GetNumAtoms()
            )
        ],
        dtype=float,
    )

    return xyz


def set_conformer_coordinates(
    mol,
    conf_id,
    xyz,
):

    conf = mol.GetConformer(
        conf_id
    )

    for i, (
        x,
        y,
        z,
    ) in enumerate(
        xyz
    ):

        conf.SetAtomPosition(
            i,
            (
                float(x),
                float(y),
                float(z),
            ),
        )


def heavy_atom_indices(
    mol,
):

    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 1
    ]


def heavy_atom_direct_rmsd(
    xyz_a,
    xyz_b,
    heavy_indices,
):
    """
    RMSD without fitting.

    Appropriate for asking how far isolated minimization moved
    the original bound pose in its existing coordinate frame.
    """

    idx = np.asarray(
        heavy_indices,
        dtype=int,
    )

    diff = (
        xyz_a[idx]
        - xyz_b[idx]
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    diff * diff,
                    axis=1,
                )
            )
        )
    )


# ============================================================
# OPENMM SYSTEM CREATION
# ============================================================

def create_openmm_system(
    rdkit_mol,
):

    offmol = Molecule.from_rdkit(
        rdkit_mol,
        allow_undefined_stereo=False,
        hydrogens_are_explicit=True,
    )

    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=[
            offmol
        ],
        forcefield=OPENFF_FORCEFIELD,
    )

    forcefield = ForceField()

    forcefield.registerTemplateGenerator(
        smirnoff.generator
    )

    topology = (
        offmol
        .to_topology()
        .to_openmm()
    )

    system = forcefield.createSystem(
        topology,
    )

    return (
        offmol,
        topology,
        system,
    )


# ============================================================
# OPENMM CONTEXT
# ============================================================

def create_context(
    system,
):

    integrator = VerletIntegrator(
        1.0
        * unit.femtoseconds
    )

    platform = Platform.getPlatformByName(
        PLATFORM_NAME
    )

    context = Context(
        system,
        integrator,
        platform,
        PLATFORM_PROPERTIES,
    )

    return (
        context,
        integrator,
    )


# ============================================================
# ENERGY / MINIMIZATION
# ============================================================

def xyz_angstrom_to_positions(
    xyz,
):

    return (
        np.asarray(
            xyz,
            dtype=float,
        )
        * unit.angstrom
    )


def context_energy_kcal(
    context,
):

    state = context.getState(
        getEnergy=True
    )

    return float(
        state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilocalories_per_mole
        )
    )


def context_positions_angstrom(
    context,
):

    state = context.getState(
        getPositions=True
    )

    pos = (
        state
        .getPositions(
            asNumpy=True
        )
        .value_in_unit(
            unit.angstrom
        )
    )

    return np.asarray(
        pos,
        dtype=float,
    )


def evaluate_coordinates(
    context,
    xyz,
):

    context.setPositions(
        xyz_angstrom_to_positions(
            xyz
        )
    )

    return context_energy_kcal(
        context
    )


def minimize_coordinates(
    context,
    xyz,
):

    context.setPositions(
        xyz_angstrom_to_positions(
            xyz
        )
    )

    initial_energy = (
        context_energy_kcal(
            context
        )
    )

    from openmm import LocalEnergyMinimizer

    LocalEnergyMinimizer.minimize(
        context,
        tolerance=(
            MIN_TOLERANCE
            * unit.kilojoules_per_mole
            / unit.nanometer
        ),
        maxIterations=MAX_ITERATIONS,
    )

    final_energy = (
        context_energy_kcal(
            context
        )
    )

    final_xyz = (
        context_positions_angstrom(
            context
        )
    )

    return (
        initial_energy,
        final_energy,
        final_xyz,
    )


# ============================================================
# ETKDG CONFORMER GENERATION
# ============================================================

def generate_solution_conformers(
    bound_mol,
):

    # Work from the same molecule and atom order.
    mol = Chem.Mol(
        bound_mol
    )

    # Remove existing bound conformer.
    mol.RemoveAllConformers()

    params = (
        AllChem.ETKDGv3()
    )

    params.randomSeed = (
        RANDOM_SEED
    )

    params.pruneRmsThresh = (
        PRUNE_RMS_A
    )

    params.useSmallRingTorsions = True

    params.useMacrocycleTorsions = True

    params.enforceChirality = True

    params.numThreads = 0


    conf_ids = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=N_CONFORMERS,
            params=params,
        )
    )

    if not conf_ids:

        raise RuntimeError(
            "RDKit failed to generate any conformers."
        )

    return (
        mol,
        conf_ids,
    )


# ============================================================
# ALIGNED RMSD TO BOUND POSE
# ============================================================

def aligned_heavy_rmsd_to_bound(
    reference_bound_mol,
    probe_mol,
    probe_conf_id,
):

    heavy = (
        heavy_atom_indices(
            reference_bound_mol
        )
    )

    atom_map = [
        (
            i,
            i,
        )
        for i in heavy
    ]

    return float(
        rdMolAlign.AlignMol(
            probe_mol,
            reference_bound_mol,
            prbCid=probe_conf_id,
            refCid=0,
            atomMap=atom_map,
        )
    )


# ============================================================
# WRITE SDF
# ============================================================

def write_single_molecule(
    filename,
    mol,
    conf_id=0,
    properties=None,
):

    out = Chem.Mol(
        mol
    )

    xyz = coordinates_from_conformer(
        mol,
        conf_id,
    )

    out.RemoveAllConformers()

    conf = Chem.Conformer(
        out.GetNumAtoms()
    )

    for i, (
        x,
        y,
        z,
    ) in enumerate(
        xyz
    ):

        conf.SetAtomPosition(
            i,
            (
                float(x),
                float(y),
                float(z),
            ),
        )

    out.AddConformer(
        conf,
        assignId=True,
    )

    if properties:

        for key, value in properties.items():

            out.SetProp(
                str(key),
                str(value),
            )

    writer = Chem.SDWriter(
        str(filename)
    )

    writer.write(
        out
    )

    writer.close()


def write_multiconformer_sdf(
    filename,
    mol,
    rows,
):

    writer = Chem.SDWriter(
        str(filename)
    )

    for row in rows:

        conf_id = int(
            row[
                "conf_id"
            ]
        )

        out = Chem.Mol(
            mol
        )

        xyz = coordinates_from_conformer(
            mol,
            conf_id,
        )

        out.RemoveAllConformers()

        conf = Chem.Conformer(
            out.GetNumAtoms()
        )

        for i, (
            x,
            y,
            z,
        ) in enumerate(
            xyz
        ):

            conf.SetAtomPosition(
                i,
                (
                    float(x),
                    float(y),
                    float(z),
                ),
            )

        out.AddConformer(
            conf,
            assignId=True,
        )

        out.SetProp(
            "CONF_ID",
            str(
                conf_id
            ),
        )

        out.SetProp(
            "ENERGY_KCAL_MOL",
            f"{row['energy_kcal_mol']:.8f}",
        )

        out.SetProp(
            "DELTA_E_KCAL_MOL",
            f"{row['delta_E_kcal_mol']:.8f}",
        )

        out.SetProp(
            "RMSD_TO_BOUND_A",
            f"{row['rmsd_to_bound_A']:.6f}",
        )

        out.SetProp(
            "ENERGY_WEIGHT",
            f"{row['energy_weight']:.10f}",
        )

        writer.write(
            out
        )

    writer.close()


# ============================================================
# ANALYZE ONE LIGAND
# ============================================================

def analyze_rank(
    rank,
):

    print()
    print("=" * 80)
    print(
        f"LIGAND {rank:03d}"
    )
    print("=" * 80)


    input_sdf = find_rank_sdf(
        rank
    )


    print(
        f"Input: {input_sdf}"
    )


    rank_dir = (
        OUT_DIR
        / f"{rank:03d}"
    )

    rank_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # LOAD BOUND MOLECULE
    # --------------------------------------------------------

    bound_mol = load_single_sdf(
        input_sdf
    )


    bound_xyz = (
        coordinates_from_conformer(
            bound_mol,
            0,
        )
    )


    heavy = (
        heavy_atom_indices(
            bound_mol
        )
    )


    print(
        f"Atoms:       {bound_mol.GetNumAtoms()}"
    )

    print(
        f"Heavy atoms: {len(heavy)}"
    )


    # --------------------------------------------------------
    # CREATE ONE FORCE FIELD SYSTEM
    # --------------------------------------------------------

    (
        offmol,
        topology,
        system,
    ) = create_openmm_system(
        bound_mol
    )


    (
        context,
        integrator,
    ) = create_context(
        system
    )


    try:

        # ----------------------------------------------------
        # EXACT BOUND-POSE ENERGY
        # ----------------------------------------------------

        E_bound_fixed = (
            evaluate_coordinates(
                context,
                bound_xyz,
            )
        )


        # ----------------------------------------------------
        # ISOLATED RELAXATION OF BOUND POSE
        # ----------------------------------------------------

        (
            _,
            E_bound_relaxed,
            bound_relaxed_xyz,
        ) = minimize_coordinates(
            context,
            bound_xyz,
        )


        bound_relax_RMSD = (
            heavy_atom_direct_rmsd(
                bound_xyz,
                bound_relaxed_xyz,
                heavy,
            )
        )


        # Save bound pose.
        write_single_molecule(
            rank_dir
            / "bound_pose.sdf",
            bound_mol,
            properties={
                "BOUND_FIXED_ENERGY_KCAL_MOL":
                    E_bound_fixed,
            },
        )


        # Save relaxed bound pose.
        bound_relaxed_mol = (
            Chem.Mol(
                bound_mol
            )
        )

        set_conformer_coordinates(
            bound_relaxed_mol,
            0,
            bound_relaxed_xyz,
        )


        write_single_molecule(
            rank_dir
            / "bound_pose_relaxed_isolated.sdf",
            bound_relaxed_mol,
            properties={
                "BOUND_RELAXED_ENERGY_KCAL_MOL":
                    E_bound_relaxed,

                "DIRECT_HEAVY_RMSD_FROM_BOUND_A":
                    bound_relax_RMSD,
            },
        )


        print(
            f"Bound fixed energy:   "
            f"{E_bound_fixed:.3f} kcal/mol"
        )

        print(
            f"Bound relaxed energy: "
            f"{E_bound_relaxed:.3f} kcal/mol"
        )

        print(
            f"Bound relaxation RMSD:"
            f" {bound_relax_RMSD:.3f} A"
        )


        # ----------------------------------------------------
        # GENERATE SOLUTION CONFORMERS
        # ----------------------------------------------------

        (
            conf_mol,
            conf_ids,
        ) = generate_solution_conformers(
            bound_mol
        )


        print(
            f"ETKDG conformers generated: "
            f"{len(conf_ids)}"
        )


        # ----------------------------------------------------
        # MINIMIZE EVERY CONFORMER
        # ----------------------------------------------------

        conformer_rows = []


        for n, conf_id in enumerate(
            conf_ids,
            start=1,
        ):

            xyz_initial = (
                coordinates_from_conformer(
                    conf_mol,
                    conf_id,
                )
            )


            (
                initial_energy,
                final_energy,
                final_xyz,
            ) = minimize_coordinates(
                context,
                xyz_initial,
            )


            set_conformer_coordinates(
                conf_mol,
                conf_id,
                final_xyz,
            )


            rmsd_bound = (
                aligned_heavy_rmsd_to_bound(
                    bound_mol,
                    conf_mol,
                    conf_id,
                )
            )


            conformer_rows.append({
                "conf_id":
                    int(
                        conf_id
                    ),

                "initial_energy_kcal_mol":
                    initial_energy,

                "energy_kcal_mol":
                    final_energy,

                "minimization_drop_kcal_mol":
                    (
                        initial_energy
                        - final_energy
                    ),

                "rmsd_to_bound_A":
                    rmsd_bound,
            })


            if (
                n == 1
                or
                n % 25 == 0
                or
                n == len(
                    conf_ids
                )
            ):

                print(
                    f"  minimized "
                    f"{n:4d}/"
                    f"{len(conf_ids):4d}"
                )


        conf_df = pd.DataFrame(
            conformer_rows
        )


        # ----------------------------------------------------
        # RELATIVE ENERGIES
        # ----------------------------------------------------

        E_solution_min = float(
            conf_df[
                "energy_kcal_mol"
            ].min()
        )


        conf_df[
            "delta_E_kcal_mol"
        ] = (
            conf_df[
                "energy_kcal_mol"
            ]
            - E_solution_min
        )


        # ----------------------------------------------------
        # ENERGY-BASED WEIGHTS
        #
        # These are NOT rigorous solution populations.
        # They are Boltzmann weights of minimized potential
        # energies only.
        # ----------------------------------------------------

        exponent = (
            -conf_df[
                "delta_E_kcal_mol"
            ]
            / RT_KCAL
        )


        # Numerically safe because minimum delta E = 0.
        weights = np.exp(
            exponent
        )


        conf_df[
            "energy_weight"
        ] = (
            weights
            / weights.sum()
        )


        # ----------------------------------------------------
        # SORT LOWEST ENERGY FIRST
        # ----------------------------------------------------

        conf_df = (
            conf_df
            .sort_values(
                "energy_kcal_mol"
            )
            .reset_index(
                drop=True
            )
        )


        conf_df[
            "energy_rank"
        ] = (
            np.arange(
                1,
                len(
                    conf_df
                )
                + 1
            )
        )


        # ----------------------------------------------------
        # BOUND STRAIN
        # ----------------------------------------------------

        strain_fixed = (
            E_bound_fixed
            - E_solution_min
        )


        strain_relaxed = (
            E_bound_relaxed
            - E_solution_min
        )


        receptor_held_distortion = (
            E_bound_fixed
            - E_bound_relaxed
        )


        # ----------------------------------------------------
        # LOWEST-ENERGY BOUND-LIKE CONFORMERS
        #
        # Several RMSD thresholds are useful rather than
        # choosing one arbitrary definition.
        # ----------------------------------------------------

        bound_like_results = {}


        for cutoff in [
            0.5,
            1.0,
            1.5,
            2.0,
        ]:

            subset = conf_df[
                conf_df[
                    "rmsd_to_bound_A"
                ]
                <= cutoff
            ]


            if len(
                subset
            ) == 0:

                bound_like_results[
                    cutoff
                ] = {
                    "exists":
                        False,

                    "min_delta_E":
                        np.nan,

                    "weight_sum":
                        0.0,
                }

            else:

                bound_like_results[
                    cutoff
                ] = {
                    "exists":
                        True,

                    "min_delta_E":
                        float(
                            subset[
                                "delta_E_kcal_mol"
                            ].min()
                        ),

                    "weight_sum":
                        float(
                            subset[
                                "energy_weight"
                            ].sum()
                        ),
                }


        # ----------------------------------------------------
        # CLOSEST SOLUTION CONFORMER
        # ----------------------------------------------------

        closest_idx = (
            conf_df[
                "rmsd_to_bound_A"
            ]
            .idxmin()
        )


        closest = (
            conf_df.loc[
                closest_idx
            ]
        )


        # ----------------------------------------------------
        # SAVE TABLE
        # ----------------------------------------------------

        conf_df.to_csv(
            rank_dir
            / "conformer_energies.csv",
            index=False,
        )


        # ----------------------------------------------------
        # SAVE MINIMIZED CONFORMER ENSEMBLE
        # ----------------------------------------------------

        rows_for_sdf = (
            conf_df
            .sort_values(
                "conf_id"
            )
            .to_dict(
                "records"
            )
        )


        write_multiconformer_sdf(
            rank_dir
            / "solution_conformers_minimized.sdf",
            conf_mol,
            rows_for_sdf,
        )


        # ----------------------------------------------------
        # ENERGY VS RMSD PLOT
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(
                8,
                6,
            )
        )


        ax.scatter(
            conf_df[
                "rmsd_to_bound_A"
            ],
            conf_df[
                "delta_E_kcal_mol"
            ],
            s=24,
            alpha=0.7,
        )


        ax.axhline(
            strain_fixed,
            linestyle="--",
            linewidth=1.2,
            label=(
                "Exact bound-pose strain"
            ),
        )


        ax.set_xlabel(
            "Heavy-atom RMSD to bound pose (A)"
        )

        ax.set_ylabel(
            "Energy above lowest minimized conformer "
            "(kcal/mol)"
        )

        ax.set_title(
            f"Rank {rank:03d}: conformational landscape"
        )


        ax.legend()


        fig.tight_layout()


        fig.savefig(
            rank_dir
            / "energy_vs_bound_rmsd.png",
            dpi=300,
        )


        plt.close(
            fig
        )


        # ----------------------------------------------------
        # CONSOLE REPORT
        # ----------------------------------------------------

        print()
        print(
            f"Lowest solution conformer: "
            f"{E_solution_min:.3f} kcal/mol"
        )

        print(
            f"BOUND STRAIN:              "
            f"{strain_fixed:.3f} kcal/mol"
        )

        print(
            f"Relaxed-bound strain:      "
            f"{strain_relaxed:.3f} kcal/mol"
        )

        print(
            f"Receptor-held distortion:  "
            f"{receptor_held_distortion:.3f} kcal/mol"
        )

        print(
            f"Closest solution conformer: "
            f"{float(closest['rmsd_to_bound_A']):.3f} A"
        )


        # ----------------------------------------------------
        # SUMMARY ROW
        # ----------------------------------------------------

        summary = {
            "rank":
                rank,

            "input_sdf":
                input_sdf.name,

            "heavy_atoms":
                len(
                    heavy
                ),

            "conformers_generated":
                len(
                    conf_ids
                ),

            "E_bound_fixed_kcal_mol":
                E_bound_fixed,

            "E_bound_relaxed_kcal_mol":
                E_bound_relaxed,

            "E_solution_min_kcal_mol":
                E_solution_min,

            "strain_bound_fixed_kcal_mol":
                strain_fixed,

            "strain_bound_relaxed_kcal_mol":
                strain_relaxed,

            "receptor_held_distortion_kcal_mol":
                receptor_held_distortion,

            "bound_relax_direct_RMSD_A":
                bound_relax_RMSD,

            "closest_solution_RMSD_A":
                float(
                    closest[
                        "rmsd_to_bound_A"
                    ]
                ),

            "closest_solution_delta_E_kcal_mol":
                float(
                    closest[
                        "delta_E_kcal_mol"
                    ]
                ),
        }


        for cutoff in [
            0.5,
            1.0,
            1.5,
            2.0,
        ]:

            label = str(
                cutoff
            ).replace(
                ".",
                "p",
            )


            summary[
                f"boundlike_le_{label}A_exists"
            ] = (
                bound_like_results[
                    cutoff
                ][
                    "exists"
                ]
            )


            summary[
                f"boundlike_le_{label}A_min_delta_E_kcal_mol"
            ] = (
                bound_like_results[
                    cutoff
                ][
                    "min_delta_E"
                ]
            )


            summary[
                f"boundlike_le_{label}A_energy_weight"
            ] = (
                bound_like_results[
                    cutoff
                ][
                    "weight_sum"
                ]
            )


        return summary


    finally:

        del context
        del integrator


# ============================================================
# MAIN
# ============================================================

def main():

    if not BASE_DIR.exists():

        raise RuntimeError(
            f"Missing directory: {BASE_DIR}"
        )


    print()
    print("=" * 80)
    print("DOCK-STRAIN")
    print("=" * 80)

    print(
        f"Top N:                    {TOP_N}"
    )

    print(
        f"ETKDG conformers/ligand:  {N_CONFORMERS}"
    )

    print(
        f"ETKDG pruning RMSD:       {PRUNE_RMS_A:.2f} A"
    )

    print(
        f"OpenFF force field:       {OPENFF_FORCEFIELD}"
    )

    print(
        f"OpenMM platform:          {PLATFORM_NAME}"
    )

    print(
        f"Temperature for weights:  {TEMPERATURE_K:.2f} K"
    )

    print()
    print(
        "NOTE: Current energies are intrinsic ligand MM energies."
    )

    print(
        "No solvent contribution is included."
    )


    summaries = []


    for rank in range(
        1,
        TOP_N + 1,
    ):

        try:

            result = analyze_rank(
                rank
            )

            result[
                "status"
            ] = "PASS"

            result[
                "error"
            ] = ""

            summaries.append(
                result
            )


        except Exception as exc:

            print()
            print(
                f"ERROR rank "
                f"{rank:03d}: "
                f"{exc}"
            )


            summaries.append({
                "rank":
                    rank,

                "status":
                    "FAIL",

                "error":
                    str(
                        exc
                    ),
            })


    # ========================================================
    # MASTER SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )


    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )


    # ========================================================
    # SUMMARY PLOT
    # ========================================================

    passed = summary_df[
        summary_df[
            "status"
        ]
        == "PASS"
    ].copy()


    if len(
        passed
    ) > 0:

        passed = (
            passed
            .sort_values(
                "strain_bound_fixed_kcal_mol",
                ascending=True,
            )
        )


        labels = [
            f"{int(x):03d}"
            for x in passed[
                "rank"
            ]
        ]


        fig, ax = plt.subplots(
            figsize=(
                10,
                max(
                    6,
                    0.32
                    * len(
                        passed
                    ),
                ),
            )
        )


        ax.barh(
            labels,
            passed[
                "strain_bound_fixed_kcal_mol"
            ],
        )


        ax.set_xlabel(
            "Intrinsic bound conformational strain "
            "(kcal/mol)"
        )

        ax.set_ylabel(
            "Ligand rank"
        )

        ax.set_title(
            "Docked-pose ligand strain"
        )


        fig.tight_layout()


        fig.savefig(
            OUT_DIR
            / "strain_ranking.png",
            dpi=300,
        )


        plt.close(
            fig
        )


    print()
    print("=" * 100)
    print("STRAIN SUMMARY")
    print("=" * 100)


    display_columns = [
        "rank",
        "strain_bound_fixed_kcal_mol",
        "receptor_held_distortion_kcal_mol",
        "bound_relax_direct_RMSD_A",
        "closest_solution_RMSD_A",
        "closest_solution_delta_E_kcal_mol",
        "status",
    ]


    display_columns = [
        c
        for c in display_columns
        if c
        in summary_df.columns
    ]


    print(
        summary_df[
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
        f"Wrote:\n"
        f"  {SUMMARY_FILE}"
    )

    print(
        f"  {OUT_DIR / 'strain_ranking.png'}"
    )


if __name__ == "__main__":
    main()

