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
  Freeze/compress/test scripts and logs for the PT/DPA2 model-preparation path.
- `training_variants/`
  Per-variant DeePMD training inputs, logs, and Slurm outputs.

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

## Packaging Notes

- Large training checkpoints were intentionally omitted so this benchmark copy stays
  focused on benchmark logic, logs, and reproducibility metadata.
- The key scripts, JSON inputs, preparation logs, and Slurm outputs were preserved.
- The moderate-size `model_dpa2.pth` and `dpa2_ckpt-26000.pt` artifacts were kept because
  they directly document the DPA-2 freeze/test path discussed in this benchmark family.
