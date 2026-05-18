# `variant_train_se_e2_a_PT`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `partial`; reason: has lcurve but no finished-training marker.
- Steps: `1000000/1000000 (100.0%)`.
- Average training time: `0.1024` s/batch.
- Wall time: `not available` s.
- Model: backend: `PT`; descriptor: `se_e2_a`; params: `2.665665 M`; descriptor neurons: `[25,50,100]`; fitting net: `[240,240,240]`; systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `0.598`, best `0.088`, last-100 median `0.352`, last-100 p10/p90 `0.259` / `0.598`.

## Lesson

Run did not complete cleanly: has lcurve but no finished-training marker.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
