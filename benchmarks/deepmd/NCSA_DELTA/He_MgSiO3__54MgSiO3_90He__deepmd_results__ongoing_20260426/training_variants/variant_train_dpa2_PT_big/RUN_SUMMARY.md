# `variant_train_dpa2_PT_big`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `partial`; reason: has lcurve but no finished-training marker.
- Steps: `1000000/1000000 (100.0%)`.
- Average training time: `0.0991` s/batch.
- Wall time: `not available` s.
- Model: backend: `PT`; descriptor: `dpa2`; descriptor neurons: `repinit=[75,150,300];nsel=120/30;layers=6;g=192/96`; fitting net: `[720,720,720]`; systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `3.49`, best `0.504`, last-100 median `3.71`, last-100 p10/p90 `2.2` / `4.68`.

## Lesson

This DPA-2 case is primarily a setup/runtime or failure-mode record, not evidence of improved predictive quality.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
