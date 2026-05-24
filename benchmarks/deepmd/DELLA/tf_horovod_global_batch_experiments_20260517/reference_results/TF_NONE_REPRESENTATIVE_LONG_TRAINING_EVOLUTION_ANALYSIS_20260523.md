# TF Representative None-Scaling Training Evolution

Date: 2026-05-23

Plot:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_NONE_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

Summary table:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_NONE_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_SUMMARY_20260523.tsv
```

Comparability note:

This is the TF-only counterpart to the representative long-training plot, but
restricted to available `none`-schedule curves. For 1GPU, worker-scaling
choices are equivalent because the worker-count factor is 1, so the
`reuse_1gpu_10k_10x` curve is used as the shared 1GPU baseline. The available
TF `none` 100k curves with lcurve data are 4GPU and 8GPU; a 16GPU `none`
folder exists in the scratch tree, but no `lcurve.out` was available for this
plot.

Interpretation:

- The 4GPU `none` curve is the stable-looking TF production-style curve used
  in the explicit TF-vs-PT none comparison.
- The 8GPU `none` curve is included to show that `none` is not a universal
  cure at higher GPU count; the present stability observation is mainly about
  the 4GPU TF case.
- Validation RMSE remains the model-selection criterion; this plot is an
  optimizer-health diagnostic.
