# Intermediate `se_e2_a` Scaling Tests, 2026-05-17

This folder records the intermediate TensorFlow `se_e2_a` scaling tests run on
NCSA Delta for the `He_MgSiO3__54MgSiO3_90He` DeePMD training benchmark family.

The working run directory is:

`/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench`

The validation workspace used for the energy-focused comparison is:

`/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_71MgSiO3_5He__v1_i_train_test__20260517`

## Scope

Completed comparison before the new intermediate jobs:

- Base `se_e2_a`.
- Width-only `big`.
- Fitting-depth-only `fit_deep2x` and `fit_deep10x`.
- Balanced width+depth `balanced_10x`.
- Held-out validation on `CONFIG=71MgSiO3_5He`, using copied local data from
  `pre/recal/{set_train,set_test}`. The original `qmd_data/v1_i*` folders were not
  modified.

New in-flight intermediate cases:

- `big2x`: width-only intermediate.
- `balanced_2x`: width plus modest depth intermediate.
- `big5x`: width-only intermediate.
- `balanced_5x`: width plus depth intermediate.
- `fit_deep5x`: fitting-net depth-only intermediate.

The 5x jobs intentionally use Slurm `after` rather than `afterok`, so they become
eligible once both 2x jobs have started. This enforces launch ordering without waiting
for the 2x trainings to complete.

## Files

- `RESULTS_SO_FAR.md`
  Current interpretation, validation ranking, and live job-state notes.
- `reference_results/ENERGY_VALIDATION_RANKING__71MgSiO3_5He.tsv`
  Energy-focused validation ranking with asymmetric bootstrap 1-sigma and 3-sigma
  intervals.
- `reference_results/INTERMEDIATE_JOB_CHAIN_20260517.tsv`
  Submitted job ids and dependency semantics for the intermediate runs.
- `reference_results/INTERMEDIATE_VARIANTS_20260517.tsv`
  Variant architecture manifest copied from the live run directory.
- `reference_results/VALIDATION_SUMMARY__71MgSiO3_5He.tsv`
  Full validation summary copied from the validation workspace.
- `reference_inputs/shared/`
  Shared DeePMD JSON input files for the new intermediate variants.
- `reference_scripts/setup_submit_intermediate_se_e2_a.py`
  Script used to generate and submit the intermediate jobs.
- `reference_scripts/calc_energy_rmse_sigma.py`
  Bootstrap script used for asymmetric energy RMSE error bars.
- `reference_scripts/submission_scripts/`
  Submitted Slurm scripts for the five intermediate runs.

## Current Scientific Read

For the `71MgSiO3_5He` held-out validation set, the main signal is energy. The
completed models show a clear ordering:

`balanced_10x` is best, followed by width-only `big`, then the fitting-depth-only
models, then base.

The new intermediate jobs are intended to answer whether balanced width+depth scaling
improves smoothly between base and `balanced_10x`, and whether a cheaper balanced model
can retain most of the energy advantage.
