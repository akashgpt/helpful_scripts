# PT Fixed-100k Training Evolution

Date: 2026-05-23

Plot:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/PT_100K_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

Summary table:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/PT_100K_TRAINING_EVOLUTION_SUMMARY_20260523.tsv
```

Interpretation:

- The four PT fixed-100k runs all show a broadly decreasing smoothed training
  total loss, energy RMSE, force RMSE, and virial/stress proxy, with noisy
  minibatch-scale oscillations suppressed by rolling mean/median lines.
- By late-window median total loss, the best curve is `16gpu`.
- By late-window median energy RMSE, the best curve is `16gpu`.
- By late-window median force RMSE, the best curve is `16gpu`.
- The 4/8/16 GPU curves converge to similar late-training quality, consistent
  with the validation table where these PT 100k-none models occupy a tight
  validation-energy band.
- Training metrics alone still do not fully determine validation ranking; use
  this plot as an optimizer-health view, not as a replacement for validation
  RMSE.
