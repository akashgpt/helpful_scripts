# PT First-10k Training Evolution

Date: 2026-05-23

This is the first-10k-step version of the PT fixed-100k training-evolution plot.
It uses the same four PT `none` runs as the 100k figure, filtered to lcurve rows
with `step <= 10000`.

Plot:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/PT_10K_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

Summary table:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/PT_10K_TRAINING_EVOLUTION_SUMMARY_20260523.tsv
```

Interpretation:

- By late-window median total loss within the first 10k steps, the lowest curve is `16gpu`.
- By late-window median energy RMSE within the first 10k steps, the lowest curve is `16gpu`.
- By late-window median force RMSE within the first 10k steps, the lowest curve is `16gpu`.
- This plot is an early-training convergence diagnostic only; validation RMSE and the full 100k curves remain the model-selection references.
