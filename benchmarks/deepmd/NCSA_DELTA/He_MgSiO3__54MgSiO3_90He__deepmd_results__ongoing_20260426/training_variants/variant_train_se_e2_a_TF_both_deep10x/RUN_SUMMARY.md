# `variant_train_se_e2_a_TF_both_deep10x`

## Curated Result

- Snapshot: `2026-05-17 23:58 CDT` from the live `training_bench` directory.
- Status: `failed`; reason: TensorFlow/DeepMD out-of-memory.
- Steps: `563200/1000000 (56.32%)`.
- Average training time: `not available` s/batch.
- Wall time: `not available` s.
- Model: backend: `TF`; descriptor: `se_e2_a`; params: `26.553265 M`; descriptor neurons: `[25,50,100,100,100,100,100,100,...;n=47]`; fitting net: `[240,240,240,240,240,240,240,240,...;n=47]`; systems: `255`.
- L-curve rows: `5633`.
- Force RMSE signal: final `1.28`, best `0.39`, last-100 median `0.8985000000000001`, last-100 p10/p90 `0.627` / `1.27`.

## Lesson

Run did not complete cleanly: TensorFlow/DeepMD out-of-memory.

Only curated setup files and this summary are kept here. Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, generated `out.json`, checkpoints, and model artifacts remain out of the git benchmark folder.
