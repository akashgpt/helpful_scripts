# Model-Prep Summary

This note replaces raw `slurm-*`, `log.*`, `lcurve.out`, and `dp_test_results.txt`
files that were previously stored in this benchmark folder.

## PT 100-Step Smoke Train

- Input: `pt_train_input.json`.
- DeePMD build: CUDA, device `CUDA:0`.
- DeePMD version reported during freeze/compress: `3.1.3`.
- Model size: `2.666 M` parameters, all trainable.
- Training length: 100 steps.
- Average training time: `0.1053 s/batch`, excluding 10 warmup batches.
- Final logged training point at step 100:
  - total RMSE: `3.62e+01`
  - energy RMSE: `1.79e+00`
  - force RMSE: `3.16e+00`
  - virial RMSE: `7.74e-01`
  - learning rate: `7.1e-06`
- The trained checkpoint was saved to `model-compression/model.ckpt`.

## Freeze / Compress Notes

- Initial model-prep submission `slurm-17777635.out` failed before useful work:
  `mkdir: cannot create directory 'model-compression': Permission denied`.
- PT freeze succeeded and saved `model.pth`.
- PT compress ran after freeze and reported tabulation lower/upper boundaries.
- TF `dp compress` can fail if the training JSON includes the legacy generated
  key `loss.loss_func`; see `DEEPMD_COMPRESS_LOSS_FUNC_NOTE.md` for the
  sanitized-input workaround used in the 2026-05-18 compressed validation pass.
- First DPA-2 freeze attempt failed with `FileNotFoundError: checkpoint`.
- DPA-2 freeze v2 succeeded and saved a frozen model to the live working tree:
  `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/shared/model_dpa2.pth`.

## `dp test` Smoke Results

See `DP_TEST_SUMMARY.tsv` for the compact table distilled from the raw output.

The smoke test used:

`/work/nvme/bguf/akashgpt/qmd_data/MgSiOH__R2SCAN/deepmd_collection_TRAIN/u.h.j.j.pro-l.liquid_vapor.water1.0g.merge.recal/deepmd`

Main read:

- The PT `se_e2_a` smoke model had much smaller error on this single test system than
  the DPA-2 variants in this short model-prep smoke test.
- These numbers are not a production validation result; they are a quick model-prep
  sanity check on one system.

## Archival Policy

This git benchmark folder should store summaries and reproducibility handles, not raw
scheduler/model logs. Raw logs, checkpoints, frozen models, generated input duplicates,
and full Slurm outputs should stay in the live run directory unless there is a specific
reason to preserve a small excerpt.
