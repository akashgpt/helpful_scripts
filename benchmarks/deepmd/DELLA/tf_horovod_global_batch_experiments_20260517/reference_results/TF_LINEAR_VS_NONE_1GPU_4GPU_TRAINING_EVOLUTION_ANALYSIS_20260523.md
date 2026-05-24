# TF Linear vs None 1/4GPU Training Evolution

Date: 2026-05-23

Plot:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_LINEAR_VS_NONE_1GPU_4GPU_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

Summary table:

```text
/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/DELLA/tf_horovod_global_batch_experiments_20260517/reference_results/TF_LINEAR_VS_NONE_1GPU_4GPU_TRAINING_EVOLUTION_SUMMARY_20260523.tsv
```

Comparability note:

This plot compares the available TF 1GPU/4GPU long curves for linear-vs-none
schedule behavior. For 1GPU, `linear`, `sqrt`, and `none` worker scaling are equivalent because the worker-count factor is 1. The comparison therefore uses one `1gpu baseline` curve plus distinct `4gpu linear` and `4gpu none` curves.

Interpretation:

- The `4gpu linear` curve is the one that shows the mild late-training
  deterioration visible in the TF representative plot.
- The `4gpu none` curve is the cleaner 4GPU TF curve used in the TF-vs-PT
  comparison.
- Given the evidence so far, `none` is clearly more stable than `linear` and
  `sqrt` for the TF multi-GPU cases tested in this benchmark family. Use this
  as the default for production-style TF runs unless the goal is explicitly to
  test a different worker-scaling schedule.
