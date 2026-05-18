# `variant_train_se_e2_a_TF_big2x`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `partial`; reason: has lcurve but did not reach configured steps.
- Steps: `341000/1000000 (34.1%)`.
- Average training time: `not available` s/batch.
- Wall time: `not available` s.
- Model: backend: `TF`; descriptor: `se_e2_a`; descriptor neurons: `[35,70,140]`; fitting net: `[340,340,340]`; systems: `255`.
- L-curve rows: `3411`.
- Force RMSE signal: final `0.338`, best `0.111`, last-100 median `0.369`, last-100 p10/p90 `0.25` / `0.53`.

## Lesson

This intermediate scaling run was still incomplete at the snapshot time; use it as an in-flight setup/result record until final validation is added.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
