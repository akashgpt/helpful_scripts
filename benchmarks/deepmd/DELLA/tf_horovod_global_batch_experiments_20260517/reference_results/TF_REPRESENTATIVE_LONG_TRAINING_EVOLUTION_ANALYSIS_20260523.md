# TF Representative Long-Training Evolution

Date: 2026-05-23

Plot:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

Summary table:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_SUMMARY_20260523.tsv
```

Important comparability note:

This is the closest available 1/4/8/16-GPU TF long-training comparison. It is
not exactly the same design as the PT fixed-100k-none plot because this
benchmark tree only has true TF `none` 100k cases for 4GPU and 8GPU. The
1/4/8/16 curves here use the `reuse_*_10k_10x` linear-scaling family, with the
16GPU curve taken from the continuation run.

Interpretation:

- By late-window median total loss, the best TF representative curve is
  `4gpu`.
- By late-window median energy RMSE, the best TF representative curve is
  `4gpu`.
- By late-window median force RMSE, the best TF representative curve is
  `4gpu`.
- The TF curves are much less uniformly healthy than the PT fixed-100k curves:
  the 2/8/16-style large-batch TF history seen elsewhere in the validation
  tables is consistent with optimization instability rather than a simple data
  limitation.
- Use this plot as an optimizer-health diagnostic; validation RMSE remains the
  production-selection metric.
