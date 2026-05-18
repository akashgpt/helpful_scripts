# `variant_train_se_e2_a_TF_desc_deep2x`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `failed`; reason: NaN detected during training.
- Steps: `527900/1000000 (52.79%)`.
- Average training time: `not available` s/batch.
- Wall time: `not available` s.
- Model: backend: `TF`; descriptor: `se_e2_a`; params: `5.443165 M`; descriptor neurons: `[25,50,100,100,100,100,100,100,...;n=14]`; fitting net: `[240,240,240]`; systems: `255`.
- L-curve rows: `5280`.
- Force RMSE signal: final `0.262`, best `0.109`, last-100 median `0.3665`, last-100 p10/p90 `0.257` / `0.595`.

## Lesson

Descriptor-depth-only scaling was unstable or too memory-heavy in this benchmark family.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
