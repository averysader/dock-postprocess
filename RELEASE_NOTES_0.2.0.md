# dock-postprocess v0.2.0

Released: 2026-09-11

## Highlights

- Shared protein, ligand, and water force-field selection for minimization and MD.
- Optional explicit-solvent minimization with periodic PME.
- New `dock-md` workflow with controlled NVT heating, NVT/NPT equilibration, and NVT/NPT production.
- Support for Amber ff14SB, ff19SB, ff15ipq, Amber14/19 aggregate presets, installed OpenFF/SMIRNOFF ligand force fields, and installed GAFF force fields.
- Explicit-water support for TIP3P, TIP3P-FB, TIP4P-Ew, TIP4P-FB, SPC/E, OPC, and OPC3.
- Configurable padding, salt concentration, ion identity, box shape, PME cutoff, temperature, pressure, timestep, friction, reporting intervals, and CUDA precision.
- DCD trajectories, state CSV files, checkpoints, serialized OpenMM systems, structural snapshots, and JSON metadata.
- Existing residue-variant and metal-coordination restraint support retained for specialized receptor chemistry.

## CUDA validation

The release candidate was smoke-tested in the `sbdd` OpenMM/CUDA environment using the previously validated p53 zinc test system.

- Dry ff14SB/OpenFF 2.3.0 minimization reproduced the historical starting energy exactly at -6795.886 kcal/mol and completed successfully on CUDA mixed precision.
- Explicit TIP3P solvent minimization with 0.15 M NaCl, a 1.0 nm dodecahedral box, and PME completed successfully and wrote the solvated minimized structure.
- Explicit-solvent NPT MD completed successfully on CUDA using 2 ps heating, 2 ps equilibration, and 5 ps production, writing the MD output directory and summary.

These short MD calculations are software smoke tests and are not intended for scientific interpretation.
