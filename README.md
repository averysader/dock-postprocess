# dock-postprocess

`dock-postprocess` is an open-source structure-based drug design toolkit for
standardizing docking results and carrying them through restrained OpenMM
minimization, pose QC, protein-ligand interaction analysis, ligand
conformational strain analysis, and integrated design prioritization.

The workflow is designed for practical docking postprocessing where preserving
the docked binding geometry is important.

It supports conventional protein-ligand complexes as well as multichain
receptors such as molecular-glue and PROTAC ternary complexes.

## Workflow

~~~text
raw receptor + docking poses
            |
         dock-init
            |
      standardized workspace
            |
       dock-minimize
            |
          dock-qc
            |
        dock-enrich
            |
      dock-landscape
            |
         dock-strain
            |
         dock-master
~~~

## Main commands

~~~text
dock-init
dock-minimize
dock-qc
dock-enrich
dock-landscape
dock-strain
dock-master
~~~

## Installation

Conda is recommended because OpenMM, OpenFF, RDKit, AmberTools, and related
scientific dependencies are most reliably managed through conda-forge.

~~~bash
git clone https://github.com/averysader/dock-postprocess.git
cd dock-postprocess

conda env create -f environment.yml
conda activate dock-postprocess

python -m pip install -e . --no-deps
~~~

For development inside an existing compatible environment:

~~~bash
python -m pip install -e . --no-deps
~~~

## 1. Initialize a workspace

Start with a directory containing:

- one receptor PDB
- one or more docked ligand SDF files

Ligand filenames do not need to contain ranks.

SDF files may contain one ligand record or multiple records.

~~~bash
dock-init \
  --input ./docking-results \
  --output ./analysis
~~~

If multiple PDB files are present:

~~~bash
dock-init \
  --input ./docking-results \
  --output ./analysis \
  --receptor receptor.pdb
~~~

Specific ligand files or glob patterns may also be supplied:

~~~bash
dock-init \
  --input ./docking-results \
  --output ./analysis \
  --ligands 'poses/*.sdf'
~~~

Multiple ligand specifications are allowed:

~~~bash
dock-init \
  --input ./docking-results \
  --output ./analysis \
  --ligands 'series1/*.sdf' \
  --ligands 'series2/*.sdf'
~~~

Recursive ligand discovery is also available:

~~~bash
dock-init \
  --input ./docking-results \
  --output ./analysis \
  --recursive
~~~

For a small test import:

~~~bash
dock-init \
  --input ./docking-results \
  --output ./analysis-test \
  --limit 10
~~~

### What `dock-init` does

`dock-init` does not perform scoring or minimization.

It imports a raw docking dataset into a standardized workspace that all later
commands understand.

Each ligand record receives a stable identifier:

~~~text
L0001
L0002
L0003
...
~~~

The original source information remains available in `ligand_manifest.csv`.

The manifest records information such as:

- stable ligand ID
- source filename
- original SDF title
- record index
- optional source rank
- SMILES
- formal charge
- heavy atom count
- canonical SDF path
- coordinate-frame QC metrics
- original SDF properties

The initializer also checks whether ligand coordinates appear to share the same
Cartesian coordinate frame as the receptor.

A workspace has the form:

~~~text
analysis/
├── receptor.pdb
├── ligand_manifest.csv
├── ligands/
│   ├── L0001.sdf
│   ├── L0002.sdf
│   └── ...
└── .dockpost/
    └── workspace.json
~~~

The purpose of this initialization step is to separate the raw docking-file
layout from the scientific analysis. Downstream commands therefore do not need
to infer receptor names, ranked filenames, multi-record SDF handling, or ligand
identity independently.

## 2. Minimize docked complexes

For an ordinary standard-protein receptor:

~~~bash
dock-minimize \
  --results ./analysis
~~~

The generalized minimizer uses:

- Amber ff14SB for protein residues
- Amber TIP3P/ion templates
- OpenFF 2.3.0 for ligands
- OpenMM
- `NoCutoff`
- HBond constraints
- staged receptor-backbone restraints of 100, 10, and 1 kcal/mol/A²

Ligand atoms and receptor side chains remain free during minimization.

A specific ligand can be rerun:

~~~bash
dock-minimize \
  --results ./analysis \
  --ligand L0027
~~~

Multiple ligand IDs may be supplied:

~~~bash
dock-minimize \
  --results ./analysis \
  --ligand L0027 \
  --ligand L0041
~~~

The default OpenMM platform selection prefers CUDA when available.

An explicit platform may be selected:

~~~bash
dock-minimize \
  --results ./analysis \
  --platform CUDA \
  --precision mixed
~~~

### Special receptor chemistry

The program deliberately does not guess unusual protonation states,
metal-coordination chemistry, or other nonstandard receptor features.

These can be supplied using a receptor configuration:

~~~bash
dock-minimize \
  --results ./analysis \
  --receptor-config receptor_config.json
~~~

Example configurations are available under `examples/`.

A receptor configuration can specify:

- receptor force-field XML files
- nonstandard residue variants
- topology-loading aliases
- explicit metal-coordination restraints
- backbone restraint schedules

This allows systems such as zinc-binding proteins to be handled without
hard-coding target-specific chemistry into the software.

For example, a structural Zn site can use explicit ligand atoms and preserve
their starting coordination distances during minimization.

## 3. Post-minimization QC

~~~bash
dock-qc \
  --results ./analysis
~~~

QC evaluates geometric changes and obvious structural problems including:

- ligand heavy-atom RMS displacement
- ligand centroid shift
- minimum receptor-ligand distance
- receptor-ligand contact count
- van der Waals clashes
- severe clashes

The public QC table is target-independent.

### Focus residues

A mechanistically important residue can optionally be examined:

~~~bash
dock-qc \
  --results ./analysis \
  --focus-residue B:220
~~~

Multiple residues may be supplied:

~~~bash
dock-qc \
  --results ./analysis \
  --focus-residue A:55 \
  --focus-residue B:220
~~~

Focus-residue analysis reports quantities including:

- residue identity
- nearest ligand-residue heavy-atom distance
- nearest receptor atom
- nearest ligand atom
- number of atom-pair contacts within 4 Å
- number of receptor heavy atoms contacting the ligand
- number of ligand heavy atoms contacting the residue

Focus-residue analysis is a geometric diagnostic.

It is not a binding score.

## 4. Enrich protein-ligand contacts

~~~bash
dock-enrich \
  --results ./analysis
~~~

This adds ligand atom chemistry and residue-level information to the raw
protein-ligand contact table.

The enriched contact table includes information such as:

- ligand identity
- ligand atom index
- ligand element
- protein chain
- protein residue
- protein atom
- interatomic distance
- van der Waals overlap
- ligand formal charge
- aromaticity
- hybridization
- ligand feature families

Outputs include:

~~~text
post_minimization_contacts_enriched.csv
residue_feature_preferences.csv
~~~

## 5. Analyze the interaction landscape

~~~bash
dock-landscape \
  --results ./analysis
~~~

The interaction-landscape analysis evaluates:

- recurring residue contacts
- hydrogen-bond patterns
- aromatic contacts
- hydrophobic contacts
- salt-bridge interactions
- ligand interaction fingerprints
- recurring spatial interaction hotspots
- anchor residues
- ligand-pair complementarity

Outputs are written under:

~~~text
results/contact_landscape/
~~~

Representative outputs include:

~~~text
anchor_residues.csv
candidate_merge_pairs.csv
interaction_hotspots_3D.csv
interaction_hotspots_3D.pdb
ligand_interaction_fingerprint.csv
ligand_pair_anchor_compatibility.csv
residue_contact_frequency.csv
residue_interaction_type_matrix.csv
true_interactions.csv
~~~

Heat maps and other diagnostic plots are also generated.

The interaction-hotspot PDB can be loaded into molecular-visualization
software such as ChimeraX or PyMOL.

## 6. Ligand conformational strain

~~~bash
dock-strain \
  --results ./analysis
~~~

The default isolated-ligand conformer search attempts 300 conformers per
ligand.

A faster exploratory calculation can use:

~~~bash
dock-strain \
  --results ./analysis \
  --conformers 50
~~~

The strain workflow compares the minimized bound ligand geometry with isolated
OpenFF ligand conformers.

Reported quantities include:

- fixed bound-pose strain
- relaxed bound-pose strain
- bound-pose distortion
- bound relaxation RMSD
- closest low-energy solution conformer
- closest-solution RMSD
- descriptive conformational-strain class
- descriptive bound-distortion class
- descriptive preorganization score

These are molecular-mechanics conformational metrics.

They are not binding free energies.

The current isolated-ligand energies do not include an explicit solvent
free-energy contribution.

For very large and flexible molecules, particularly PROTACs, substantially
more conformational sampling may be required than the default.

## 7. Build the master design table

~~~bash
dock-master \
  --results ./analysis
~~~

The master table integrates:

- pose QC
- ligand strain
- anchor coverage
- typed interactions
- interaction-network richness

into:

~~~text
results/master_design_table.csv
~~~

The table provides a convenient single view for comparing compounds after the
individual analyses are complete.

`design_priority_score` is a descriptive multi-objective prioritization
heuristic.

It is not an affinity prediction.

Individual physical and structural metrics should remain available for
scientific interpretation rather than relying only on the composite score.

## Results layout

A completed project has approximately the following form:

~~~text
analysis/
├── receptor.pdb
├── ligand_manifest.csv
├── ligands/
│   ├── L0001.sdf
│   ├── L0002.sdf
│   └── ...
│
├── .dockpost/
│   └── workspace.json
│
└── results/
    ├── L0001/
    │   ├── complex_minimized.pdb
    │   └── ligand_minimized.sdf
    │
    ├── L0002/
    │   └── ...
    │
    ├── minimization_summary.csv
    ├── post_minimization_QC.csv
    ├── post_minimization_contacts.csv
    ├── post_minimization_contacts_enriched.csv
    ├── residue_feature_preferences.csv
    ├── focus_residue_QC.csv
    │
    ├── contact_landscape/
    │   └── ...
    │
    ├── ligand_strain/
    │   └── ...
    │
    └── master_design_table.csv
~~~

`focus_residue_QC.csv` is created only when focus residues are requested.

## Multichain and ternary complexes

All receptor chains are treated as part of the receptor.

There is no assumption that chain A or chain B is the primary protein.

This makes the workspace and minimization architecture suitable for:

- conventional protein-ligand complexes
- molecular-glue complexes
- multichain binding sites
- PROTAC ternary complexes

For a ternary system, both protein partners should be present in the receptor
PDB and the ligand should already be positioned in the intended Cartesian
coordinate frame.

Special receptor chemistry should still be supplied explicitly through the
receptor configuration when required.

## Coordinate-frame QC

`dock-init` checks whether each ligand appears spatially compatible with the
receptor coordinate frame.

This is designed to catch problems such as:

- ligand poses exported in a different coordinate frame
- receptor structures from a different alignment
- accidental use of unrelated docking files

A warning does not itself prove that a pose is invalid, but suspicious
coordinate-frame results should be inspected before downstream calculations
are interpreted.

## Scientific philosophy

The package is intended for physics-informed docking postprocessing rather
than replacing experimental affinity measurements.

Docking scores, minimization energies, ligand strain, contact counts, and
heuristic design scores represent different physical or descriptive
quantities and should not be interpreted interchangeably.

The workflow is useful for questions such as:

- Does the docked ligand remain geometrically stable after restrained
  minimization?
- Does the pose create obvious steric problems?
- Which receptor interactions recur across a ligand series?
- Which ligands preserve important interaction anchors?
- Which molecules adopt unusually strained bound conformations?
- Which compounds provide useful combinations of structural stability,
  interaction coverage, and conformational accessibility?

## Interpretation cautions

### OpenMM potential energy

The OpenMM complex potential energy is useful for monitoring minimization
behavior and detecting obvious problems.

It should not be interpreted directly as a ligand binding affinity.

### Ligand strain

The strain workflow evaluates intrinsic conformational energetics using the
selected molecular mechanics model.

It does not currently include a rigorous solvent free-energy term.

### Contact counts

More contacts are not automatically better.

Interaction geometry, interaction type, desolvation, strain, and receptor
environment should be considered together.

### Design priority score

`design_priority_score` is a descriptive heuristic designed to combine several
useful structural indicators.

It is not a thermodynamic observable and is not trained as an affinity model.

## Current implementation status

Version 0.1.0 provides a generalized public interface and standardized output
model.

The OpenMM minimizer is implemented natively in the generalized package.

Some downstream analyses currently call validated internal engines inherited
from the original workflow through temporary compatibility staging.

These implementation details are intentionally hidden from the public data
model and command-line interface.

Future releases may migrate those engines into native package modules while
preserving the same public workflow.

## Development status

Version 0.1.0 is an initial release candidate.

The generalized workflow has been regression-tested through:

~~~text
dock-init
    ->
dock-minimize
    ->
dock-qc
    ->
dock-enrich
    ->
dock-landscape
    ->
dock-strain
    ->
dock-master
~~~

The native generalized minimizer was validated against the original
target-specific implementation before the target-specific receptor chemistry
was moved into explicit configuration.

## License

MIT License.

See `LICENSE`.

## Citation

Citation metadata is provided in `CITATION.cff`.

If you use `dock-postprocess` in published work, please cite the software
release used for the analysis.
