# `variant_train_dpa2_PT_big_strip_v2`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `failed`; reason: filesystem input/output error while reading data.
- Steps: `499900/1000000 (49.99%)`.
- Average training time: `not available` s/batch.
- Wall time: `not available` s.
- Model: backend: `PT`; descriptor: `dpa2`; descriptor neurons: `repinit=[75,150,300];nsel=360/120;layers=6;g=192/96`; fitting net: `[720,720,720]`; systems: `255`.
- L-curve rows: `5000`.
- Force RMSE signal: final `3.76`, best `1.05`, last-100 median `3.6799999999999997`, last-100 p10/p90 `2.52` / `4.86`.

## Lesson

This DPA-2 case is primarily a setup/runtime or failure-mode record, not evidence of improved predictive quality.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
