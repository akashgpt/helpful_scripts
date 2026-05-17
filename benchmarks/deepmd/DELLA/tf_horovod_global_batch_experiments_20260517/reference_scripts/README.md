# Reference Scripts

These are archived reference copies from the scratch experiment root:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517
```

They are included here to make the benchmark record reproducible. Do not run
submission scripts directly from this benchmark folder. Copy/adapt them into a
scratch working directory first, then submit from the intended run root.

| Script | Purpose |
|---|---|
| `materialize_experiments.py` | Generated the first short-run matrix from reused baseline runs and missing experiment definitions. |
| `materialize_10x_steps.py` | Generated the 10x-step follow-up matrix and run directories. |
| `summarize_training_runs.py` | Parsed `lcurve.out` and timing information into `TRAINING_SUMMARY.tsv`. |
| `summarize_pseudo_validation.py` | Parsed `dp test` logs and now reports asymmetric 1-sigma and 2-sigma bootstrap errors. |
| `run_pseudo_validation.sbatch` | Slurm array used for freeze/compress/test pseudo-validation. |
| `submit_*.sh` | Historical submission helpers used from the scratch experiment root. |
