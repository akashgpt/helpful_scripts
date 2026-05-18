# `variant_train_se_e2_a_TF_balanced_10x`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `complete`; reason: finished training.
- Steps: `1000000/1000000 (100.0%)`.
- Average training time: `0.163` s/batch.
- Wall time: `166047.392` s.
- Model: backend: `TF`; descriptor: `se_e2_a`; params: `26.804305 M`; descriptor neurons: `[60,120,240,240,240,240]`; fitting net: `[620,620,620,620,620,620]`; systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `0.371`, best `0.113`, last-100 median `0.373`, last-100 p10/p90 `0.239` / `0.663`.

## Lesson

This completed balanced width+depth model is the strongest completed energy-validation candidate from the current comparison set.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
