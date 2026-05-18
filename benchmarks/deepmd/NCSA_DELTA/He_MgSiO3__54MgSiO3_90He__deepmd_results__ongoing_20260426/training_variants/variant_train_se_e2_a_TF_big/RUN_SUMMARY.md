# `variant_train_se_e2_a_TF_big`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `complete`; reason: finished training.
- Steps: `1000000/1000000 (100.0%)`.
- Average training time: `0.0805` s/batch.
- Wall time: `82379.072` s.
- Model: backend: `TF`; descriptor: `se_e2_a`; params: `23.91 M`; descriptor neurons: `[75,150,300]`; fitting net: `[720,720,720]`; systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `0.391`, best `0.117`, last-100 median `0.3815`, last-100 p10/p90 `0.237` / `0.672`.

## Lesson

Width-only TensorFlow scaling completed and improved the energy-validation picture relative to base, but less than balanced scaling.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
