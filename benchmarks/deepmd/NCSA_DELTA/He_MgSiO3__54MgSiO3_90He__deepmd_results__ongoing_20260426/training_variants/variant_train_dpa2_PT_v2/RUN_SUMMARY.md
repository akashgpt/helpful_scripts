# `variant_train_dpa2_PT_v2`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `failed`; reason: PyTorch dataloader bus error / shm pressure.
- Steps: `972800/1000000 (97.28%)`.
- Average training time: `not available` s/batch.
- Wall time: `not available` s.
- Model: backend: `PT`; descriptor: `dpa2`; descriptor neurons: `repinit=[25,50,100];nsel=360/120;layers=6;g=64/32`; fitting net: `[240,240,240]`; systems: `255`.
- L-curve rows: `9729`.
- Force RMSE signal: final `3.29`, best `0.504`, last-100 median `3.5549999999999997`, last-100 p10/p90 `2.23` / `4.6`.

## Lesson

This DPA-2 case is primarily a setup/runtime or failure-mode record, not evidence of improved predictive quality.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
