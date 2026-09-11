# dock-postprocess v0.2.0 validation checklist

Run these checks in the same conda environment used for production OpenMM work.

## 1. Package tests

```bash
python -m pip install -e . --no-deps
python -m pytest -q
```

## 2. Dry minimization regression

Use a ligand that was previously validated with v0.1.0.

```bash
dock-minimize \
  --results ./analysis \
  --ligand L0001 \
  --receptor-config ./receptor-config.json \
  --solvent none \
  --protein-forcefield ff14SB \
  --ligand-forcefield openff-2.3.0 \
  --water-model tip3p \
  --platform CUDA
```

Confirm that the starting energy agrees with the prior dry model to numerical precision and that the minimized pose remains within expected GPU minimization variability.

## 3. Explicit-solvent minimization smoke test

```bash
dock-minimize \
  --results ./analysis \
  --ligand L0001 \
  --receptor-config ./receptor-config.json \
  --solvent explicit \
  --protein-forcefield ff14SB \
  --ligand-forcefield openff-2.3.0 \
  --water-model tip3p \
  --padding-nm 1.0 \
  --ionic-strength 0.15 \
  --platform CUDA
```

Confirm creation of `results/L0001/solvated/complex_minimized_solvated.pdb` and `system_metadata.json`.

## 4. Explicit-solvent NVT smoke test

```bash
dock-md \
  --results ./analysis \
  --ligand L0001 \
  --receptor-config ./receptor-config.json \
  --solvent explicit \
  --water-model tip3p \
  --ensemble nvt \
  --heat-ps 2 \
  --equilibration-ps 2 \
  --production-ps 5 \
  --report-interval-ps 1 \
  --trajectory-interval-ps 1 \
  --checkpoint-interval-ps 2 \
  --platform CUDA
```

## 5. Explicit-solvent NPT smoke test

```bash
dock-md \
  --results ./analysis \
  --ligand L0001 \
  --receptor-config ./receptor-config.json \
  --solvent explicit \
  --water-model tip3p \
  --ensemble npt \
  --heat-ps 2 \
  --equilibration-ps 2 \
  --production-ps 5 \
  --report-interval-ps 1 \
  --trajectory-interval-ps 1 \
  --checkpoint-interval-ps 2 \
  --platform CUDA
```

Confirm that the NPT state CSV contains periodic volume and density columns and that the trajectory, checkpoint, final structures, system XML, and JSON metadata are written.

The short MD calculations above are smoke tests only and are not intended for scientific interpretation.

## Recorded CUDA smoke-test result

The v0.2.0 release candidate was tested on 2026-09-11 with the previously validated p53 zinc workspace.

- Dry minimization: PASS. Starting potential energy -6795.886 kcal/mol; final -7077.209 kcal/mol.
- Explicit TIP3P minimization: PASS. PME, 1.0 nm dodecahedral box, 0.15 M NaCl; solvated minimized structure written.
- Explicit NPT MD: PASS. 50 to 300 K heating for 2 ps, 2 ps NPT equilibration, and 5 ps NPT production on CUDA mixed precision.

The OpenFF/Interchange warnings observed during system generation were non-fatal. They remain visible rather than being suppressed so users can audit parameterization messages.
