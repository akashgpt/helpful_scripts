# `variant_train_se_e2_a_TF_desc_deep10x`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `failed`; reason: TensorFlow/DeepMD out-of-memory.
- Steps: `301500/1000000 (30.15%)`.
- Average training time: `not available` s/batch.
- Wall time: `not available` s.
- Model: backend: `TF`; descriptor: `se_e2_a`; params: `26.653165 M`; descriptor neurons: `[25,50,100,100,100,100,100,100,...;n=98]`; fitting net: `[240,240,240]`; systems: `255`.
- L-curve rows: `3016`.
- Force RMSE signal: final `3.07`, best `1.04`, last-100 median `3.72`, last-100 p10/p90 `2.35` / `5.22`.

## Lesson

Descriptor-depth-only scaling was unstable or too memory-heavy in this benchmark family.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
