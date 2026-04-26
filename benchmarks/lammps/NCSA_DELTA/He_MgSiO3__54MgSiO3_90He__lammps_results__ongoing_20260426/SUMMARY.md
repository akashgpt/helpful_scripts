# LAMMPS Benchmark Snapshot: He/MgSiO3 `54MgSiO3_90He`

## Status

This folder is a **working benchmark snapshot**, not a final benchmark closeout.
The original source tree mixed `LAMMPS` inference tests and `DeePMD-kit` training/model
tests; this package keeps only the `LAMMPS` side.

High-priority reminder:

- Work from this benchmark family is still ongoing.
- We need to revisit these results before treating them as final guidance.
- See [HIGH_PRIORITY__ONGOING_WORK.md](./HIGH_PRIORITY__ONGOING_WORK.md).

## What This Folder Contains

- `SOURCE_SUMMARY__combined.md`
  The original combined source summary, which also discusses the related `deepmd` work.
- `shared/`
  Common LAMMPS inputs and configuration files used across the inference benchmarks.
- `variants/`
  Per-run submission scripts, input snapshots, LAMMPS logs, and Slurm outputs for the
  benchmark variants.

This folder corresponds to the `LAMMPS` results only. The related `deepmd` training and
model-preparation results were split into:

- [He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426](/projects/bguf/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426)

## Scientific Summary

These tests benchmark LAMMPS MD inference on NCSA Delta for a He/MgSiO3 system using:

- TensorFlow backend
- PyTorch backend
- PyTorch + KOKKOS

with size scaling across roughly:

- `360` atoms
- `2880` atoms
- `28800` atoms

The main conclusions from the source summary are:

- TensorFlow inference is consistently faster than PyTorch for this benchmark family.
- PyTorch is about `3.5x` to `4.5x` slower than TensorFlow across the tested sizes.
- KOKKOS is not helping this `pair_style deepmd` workload here; it is a regression relative
  to plain PyTorch in every tested size range.
- Per-atom throughput improves with system size, which is consistent with the GPU becoming
  better utilized at large `N`.

In simple terms: for this round, the important LAMMPS conclusion is not a fine-tuned KOKKOS
setting, but that the plain TensorFlow backend remains the fastest practical inference path
for this DeePMD workload on Delta.

## Packaging Notes

- `npt.dump` trajectory files were intentionally not copied because they are bulky and not
  needed to understand the benchmark outcome.
- The very large compressed PT model `model_comp.pth` was also intentionally not mirrored
  here. The copied PT scripts are preserved as benchmark records, but rerunning those exact
  jobs still requires access to the original source tree or an equivalent model artifact.
- The lighter `model_dpa2.pth` file was kept because it is small enough to preserve and is
  directly relevant to the `variant_PT_DPA2_N360` benchmark.
