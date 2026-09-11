# Changelog

All notable changes to dock-postprocess will be documented here.

## 0.2.0 - 2026-09-11

### Added

- Shared force-field registry used by minimization and MD.
- Protein force-field presets for Amber ff14SB, ff19SB, ff15ipq, amber14-all, and amber19-all.
- Ligand parameterization through installed OpenFF/SMIRNOFF or GAFF force fields.
- Shared explicit-solvent configuration with TIP3P, TIP3P-FB, TIP4P-Ew, TIP4P-FB, SPC/E, OPC, and OPC3.
- Explicit periodic solvent boxes with configurable padding, ionic strength, ion identity, box shape, and PME cutoff.
- `dock-md` command for pre-MD minimization, NVT heating, NVT/NPT equilibration, and NVT/NPT production sampling.
- Configurable temperature, pressure, timestep, Langevin friction, reporting intervals, random seed, platform, and GPU precision.
- Configurable backbone restraints for heating, equilibration, and production.
- DCD trajectories, CSV state reports, checkpoints, serialized OpenMM systems, and MD metadata.
- Solvated and solute-only structural snapshots.
- `receptor_minimized.pdb` output to carry minimized receptor coordinates into MD.

### Changed

- `dock-minimize` now uses the shared force-field and solvation layers.
- The v0.1.0 dry minimization model remains the default (`--solvent none`).
- Explicit-solvent minimization uses periodic PME while preserving a solute-only `complex_minimized.pdb` for existing QC and downstream analyses.
- Force-field and solvent choices are now simulation settings rather than target-specific receptor-chemistry settings.

### Compatibility

- Existing receptor residue variants and metal-coordination restraints remain supported.
- Existing v0.1.0 analysis workspaces remain readable. If `receptor_minimized.pdb` is absent, `dock-md` falls back to the workspace receptor and uses the minimized ligand coordinates.
- NPT is rejected for dry/nonperiodic systems.

### Validation

- Dry minimization regression completed successfully on CUDA mixed precision, reproducing the historical starting energy exactly for the validated p53 zinc test case.
- Explicit TIP3P/PME minimization completed successfully with 0.15 M NaCl in a dodecahedral periodic box.
- Short explicit-solvent NPT MD smoke testing completed successfully on CUDA, including heating, equilibration, production, and output generation.
- These short MD runs validate system construction and execution only and are not intended for scientific interpretation.

## 0.1.0 - 2026-09-05

Initial release candidate.

### Added

- `dock-init` for importing receptor/ligand docking datasets into a standardized workspace.
- Stable ligand identifiers (`L0001`, `L0002`, ...).
- Support for arbitrary SDF filenames.
- Support for multi-record SDF files.
- Preservation of original ligand titles and SDF properties.
- Coordinate-frame validation during workspace initialization.
- Native generalized OpenMM receptor-ligand minimization.
- Multichain receptor support.
- Configurable residue variants.
- Explicit configurable metal-coordination restraints.
- OpenFF 2.3.0 ligand parameterization.
- Amber ff14SB receptor parameterization.
- Staged receptor-backbone restraints during minimization.
- Generalized post-minimization pose QC.
- Optional focus-residue geometric QC.
- Protein-ligand contact enrichment.
- Chemistry-aware interaction landscape analysis.
- Ligand conformational strain analysis.
- Integrated master design table.
- Generalized public results layout.
- Installable `dock-*` console commands.

### Validated

The v0.1.0 workflow was regression-tested against the original validated
p53/BRD4 workflow.

The generalized minimizer reproduced the original starting OpenMM energy
exactly for the validation system, with only very small expected differences
in final GPU-minimized coordinates and energies.

The complete pipeline was validated through:

`dock-init -> dock-minimize -> dock-qc -> dock-enrich -> dock-landscape -> dock-strain -> dock-master`

### Notes

Some downstream scientific engines remain internally wrapped compatibility
implementations from the original validated workflow. Their public interfaces
and output layouts are generalized. These engines may be migrated into native
package modules in later releases without changing the public workflow.
