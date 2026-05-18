# `variant_train_se_e2_a_TF_both_deep2x`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `complete`; reason: finished training.
- Steps: `1000000/1000000 (100.0%)`.
- Average training time: `0.0994` s/batch.
- Wall time: `100974.538` s.
- Model: backend: `TF`; descriptor: `se_e2_a`; params: `5.380165 M`; descriptor neurons: `[25,50,100,100,100,100,100,100]`; fitting net: `[240,240,240,240,240,240,240,240]`; systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `0.459`, best `0.151`, last-100 median `0.4505`, last-100 p10/p90 `0.269` / `0.708`.

## Lesson

Completed run; keep this as a reproducible setup plus compact timing/error record.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
