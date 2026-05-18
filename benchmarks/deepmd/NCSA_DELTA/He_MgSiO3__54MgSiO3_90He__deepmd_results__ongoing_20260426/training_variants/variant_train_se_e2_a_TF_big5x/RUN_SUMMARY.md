# `variant_train_se_e2_a_TF_big5x`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `partial`; reason: has lcurve but did not reach configured steps.
- Steps: `262800/1000000 (26.28%)`.
- Average training time: `not available` s/batch.
- Wall time: `not available` s.
- Model: backend: `TF`; descriptor: `se_e2_a`; descriptor neurons: `[56,112,224]`; fitting net: `[540,540,540]`; systems: `255`.
- L-curve rows: `2629`.
- Force RMSE signal: final `0.407`, best `0.135`, last-100 median `0.3605`, last-100 p10/p90 `0.226` / `0.61`.

## Lesson

This intermediate scaling run was still incomplete at the snapshot time; use it as an in-flight setup/result record until final validation is added.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
