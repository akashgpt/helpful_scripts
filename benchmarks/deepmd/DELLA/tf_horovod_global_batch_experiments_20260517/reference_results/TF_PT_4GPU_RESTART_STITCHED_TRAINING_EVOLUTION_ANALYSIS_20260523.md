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
