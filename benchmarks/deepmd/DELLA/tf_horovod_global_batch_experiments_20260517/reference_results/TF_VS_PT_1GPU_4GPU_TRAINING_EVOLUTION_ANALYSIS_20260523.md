# TF vs PT 1GPU/4GPU Training Evolution

Date: 2026-05-23

Plot:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_VS_PT_1GPU_4GPU_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

Summary table:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_VS_PT_1GPU_4GPU_TRAINING_EVOLUTION_SUMMARY_20260523.tsv
```

Comparability note:

The PT 1GPU/4GPU curves are fixed-100k `none` runs. The TF 4GPU curve is the
fixed-100k `none` run, but the available TF 1GPU long curve is the
`reuse_1gpu_10k_10x` linear-scaling run. This plot is therefore best read as a
framework optimizer-health comparison, not as a perfectly controlled schedule
comparison.

Interpretation:

- Best late-window median total loss: `TF 4gpu`.
- Best late-window median energy RMSE: `TF 4gpu`.
- Best late-window median force RMSE: `PT 4gpu`.
- TF 4GPU and PT 4GPU are the cleanest practical comparison here; both are
  100k-step `none` cases and both train smoothly, with PT 4GPU having a lower
  late-window median force while TF 4GPU has the best validation force/energy
  among these particular rows.
- The TF 1GPU curve is reasonable but schedule-mismatched relative to PT 1GPU,
  so avoid over-reading TF-vs-PT 1GPU differences from this plot alone.
