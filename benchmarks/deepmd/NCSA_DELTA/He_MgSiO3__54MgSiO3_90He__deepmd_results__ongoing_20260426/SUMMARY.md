# DeePMD Benchmark Snapshot: He/MgSiO3 `54MgSiO3_90He`

## Status

This folder is a **working benchmark snapshot**, not a finished benchmark campaign.
The original source tree mixed `DeePMD-kit` training/model-prep work with `LAMMPS`
inference work; this package keeps only the `deepmd` side.

High-priority reminder:

- Work from this benchmark family is still ongoing.
- We need to revisit these results before treating them as final guidance.
- See [HIGH_PRIORITY__ONGOING_WORK.md](./HIGH_PRIORITY__ONGOING_WORK.md).

## What This Folder Contains

- `SOURCE_SUMMARY__combined.md`
  The original combined source summary, which also discussed the related `lammps` work.
- `shared/`
  Training JSON files for the main `se_e2_a` and `DPA-2` variants.
- `model_prep/`
  Freeze/compress/test scripts plus distilled PT/DPA2 model-preparation summaries.
- `training_variants/`
  Per-variant DeePMD training inputs, submission scripts, and a distilled run summary.
- `intermediate_scaling_20260517/`
  Follow-up TensorFlow `se_e2_a` width/depth scaling work. This includes the
  `71MgSiO3_5He` held-out validation ranking for base, `big`, fitting-depth variants,
  and `balanced_10x`, plus in-flight `big2x`, `balanced_2x`, `big5x`, `balanced_5x`,
  and `fit_deep5x` jobs.
- `compressed_validation_20260518/`
  Curated `71MgSiO3_5He` validation comparison table with `parameter_count`,
  compressed `freeze` + `compress` + `dp test` results, and the noncompressed `big`
  reference row kept because compressed export failed.

This folder corresponds to the `DeePMD-kit` results only. The related `LAMMPS` inference
results were split into:

- [He_MgSiO3__54MgSiO3_90He__lammps_results__ongoing_20260426](/projects/bguf/akashgpt/run_scripts/helpful_scripts/benchmarks/lammps/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__lammps_results__ongoing_20260426)

## Scientific Summary

These DeepMD tests focus on training speed and model-preparation behavior on NCSA Delta for:

- `se_e2_a` with TensorFlow
- `se_e2_a` with PyTorch
- `DPA-2` with PyTorch
- several larger or corrected `DPA-2` follow-up variants

The main conclusions from the source summary are:

- For the small-batch training benchmark used here, `se_e2_a` with PyTorch is about
  `2.6x` slower per batch than `se_e2_a` with TensorFlow.
- The baseline `DPA-2` PyTorch run lands at nearly the same per-batch cost as
  `se_e2_a` PyTorch in this small-batch regime.
- That near-equality should **not** be overinterpreted as the final DPA-2 cost story.
  The source summary explicitly notes that a more realistic DPA-2 cost estimate still
  needs larger repformer settings and larger batch sizes.

In simple terms: the current DeepMD results are enough to say that PT training is slower
than TF for `se_e2_a` here, but they are **not** enough to close out the realistic cost
of larger `DPA-2` models. That follow-up is still pending.

## 2026-05-17 `se_e2_a` Intermediate Scaling Update

See [intermediate_scaling_20260517](./intermediate_scaling_20260517) for the current
energy-focused validation and the new in-flight intermediate architecture runs.

Current validation inference:

- On the held-out `71MgSiO3_5He` test split, `balanced_10x` gives the best completed
  energy RMSE/atom.
- Width-only `big` is the cheaper runner-up.
- Fitting-net depth-only scaling is not competitive for energy in this validation test.
- New intermediate `2x` and `5x` width/balanced/depth runs were submitted to determine
  whether a cheaper balanced architecture captures most of the `balanced_10x` gain.

## Packaging Notes

- Large training checkpoints were intentionally omitted so this benchmark copy stays
  focused on benchmark logic, setup files, summaries, and reproducibility metadata.
- The key scripts, JSON inputs, and compact markdown/TSV summaries were preserved.
- Raw Slurm outputs, full training logs, generated duplicate JSONs, and model artifacts
  such as `model_dpa2.pth` / `dpa2_ckpt-26000.pt` are intentionally not kept in this git
  benchmark copy. Their useful lessons should be distilled into markdown or TSV summaries.
