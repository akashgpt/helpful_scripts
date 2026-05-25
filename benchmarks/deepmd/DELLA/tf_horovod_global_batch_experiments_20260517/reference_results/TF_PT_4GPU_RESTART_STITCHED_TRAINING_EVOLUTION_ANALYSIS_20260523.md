# 4GPU Checkpoint Restart Stitched Training Curves

Date: 2026-05-23

Plot:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_PT_4GPU_RESTART_STITCHED_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

Summary table:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_PT_4GPU_RESTART_STITCHED_TRAINING_EVOLUTION_SUMMARY_20260523.tsv
```

Recipe:

- Use the cumulative ``lcurve.out`` in each restart-chain run directory.
- Smooth each metric with both rolling median and rolling mean.
- Draw restart-boundary markers from ``CHAIN_HISTORY.tsv`` when present.
- Read this as an optimizer-health/restart-continuity diagnostic; still use
  validation RMSE before ranking continuation checkpoints.

## 2026-05-24 Follow-up Restart Diagnostics

- `tf4g_100k_none_10min_chain_healthgate_second_latest_ckpt` reached
  100000 steps cleanly. Final training metrics were total `0.898`, E
  `0.0148`, F `0.489`, V `0.0281`, LR `1.0e-08`; the last-10%-of-rows
  median was total `0.9165`, E `0.01515`, F `0.519`, V `0.0233`. This
  scratch diagnostic did not run freeze/compress.
- `tf4g_100k_none_final_template_15min_hvd` reached 100000 steps and did run
  freeze/compress, producing `model-compression/pv.pb` and
  `model-compression/pv_comp.pb`. Final training metrics were total `0.877`,
  E `0.00155`, F `0.484`, V `0.0307`, LR `1.0e-08`; last-10%-of-rows median
  was total `0.903`, E `0.0148`, F `0.507`, V `0.0233`.
- `pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2` reached
  100000 steps and ran freeze/compress, producing `model-compression/pv.pth`
  and `model-compression/pv_comp.pth`. Final training metrics were total
  `7.51`, E `0.0274`, F `0.524`, V `0.0374`, LR `2.0e-04`; last-10%-of-rows
  median was total `7.15`, E `0.03525`, F `0.493`, V `0.0347`.
- Relative to the clean 4GPU PT baseline, all PT restart-chain variants should
  be treated as failed or unresolved. The fixed v2 run improved the mechanics
  and finalized successfully, but the total-loss behavior still drifted high.
- The old PT healthgate attempt selected `model.ckpt-9800.pt` after reaching
  about step `19980`; use the `v2` directory for the clean selector-guarded
  comparison.
- The PT v2 run used `save_freq = 100` and produced many checkpoints, not one:
  `model.ckpt-100.pt` through `model.ckpt-100000.pt`.

