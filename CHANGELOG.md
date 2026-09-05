# Changelog

All notable changes to dock-postprocess will be documented here.

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
