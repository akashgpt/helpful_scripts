# `variant_train_se_e2_a_TF_fit_deep10x`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `complete`; reason: finished training.
- Steps: `1000000/1000000 (100.0%)`.
- Average training time: `0.1003` s/batch.
- Wall time: `103929.023` s.
- Model: backend: `TF`; descriptor: `se_e2_a`; params: `26.768865 M`; descriptor neurons: `[25,50,100]`; fitting net: `[240,240,240,240,240,240,240,240,...;n=86]`; systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `0.375`, best `0.116`, last-100 median `0.356`, last-100 p10/p90 `0.218` / `0.601`.

## Lesson

Fitting-depth-only scaling completed, but later energy validation showed it was not the most competitive direction.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
