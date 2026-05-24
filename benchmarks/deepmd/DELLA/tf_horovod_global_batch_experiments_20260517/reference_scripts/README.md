# Reference Scripts

These are archived reference copies from the scratch experiment root:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517
```

They are included here to make the benchmark record reproducible. Do not run
submission scripts directly from this benchmark folder. Copy/adapt them into a
scratch working directory first, then submit from the intended run root.

Pseudo-validation summarizers parse DeePMD `dp test` logs. Their metric units
are `Energy RMSE/Natoms` in eV/atom, `Force RMSE` in eV/A, and
`Virial RMSE/Natoms` in eV per atom. They do not convert virial to stress in
GPa. Aggregate uncertainty is reported as explicit bootstrap percentiles:
p0.135, p15.865, p50, p84.135, and p99.865.

| Script | Purpose |
|---|---|
| `materialize_experiments.py` | Generated the first short-run matrix from reused baseline runs and missing experiment definitions. |
| `materialize_10x_steps.py` | Generated the 10x-step follow-up matrix and run directories. |
| `summarize_training_runs.py` | Parsed `lcurve.out` and timing information into `TRAINING_SUMMARY.tsv`. |
| `summarize_pseudo_validation.py` | Parsed `dp test` logs and reports bootstrap percentile columns. |
| `print_all_tf_val_sorted_steps.py` | Prints the TF validation table with checkpoint/final step columns for the 20260517 NH3/H2 benchmark analysis. |
| `organize_tf_pt_training_plots.py` | Organizes generated TF/PT training-curve plots into the benchmark plot subfolders used for the 20260517 analysis. |
| `plot_*training_evolution.py` | Plain lcurve plot templates: load one `lcurve.out` per run, plot total/energy/force/virial-stress columns, and use rolling median/mean smoothing. See `TRAINING_CURVE_PLOT_TEMPLATES_20260524.md` for which script maps to each figure. |
| `plot_checkpoint_restart_stitched_training_evolution.py` | Stitched checkpoint-restart plot template: uses cumulative `lcurve.out`, restart markers from `CHAIN_HISTORY.tsv`, rolling median/mean smoothing, and writes PNG/TSV/MD outputs. |
| `run_pseudo_validation.sbatch` | Slurm array used for freeze/compress/test pseudo-validation. |
| `DELLA_TIGER_train_1h.apptr.Ngpu.TF.sh` | Della/Tiger TF production 1-to-N-GPU single-slice training template; Level 2 owns chained resubmission, while freeze/compress is gated by DeePMD's `finished training` log signal. |
| `DELLA_TIGER_train_1h.apptr.Ngpu.PT.sh` | Della/Tiger PT production 1-to-N-GPU single-slice training template; Level 2 owns chained resubmission, while freeze/compress is gated by DeePMD's `finished training` log signal. |
| `ALCF_POLARIS_train_1h.apptr.Ngpu.TF.sh` | Polaris TF production multi-GPU single-slice training template with PBS resource directives; freeze/compress is gated by DeePMD's `finished training` log signal. |
| `ALCF_POLARIS_train_1h.apptr.Ngpu.PT.sh` | Polaris PT production multi-GPU single-slice training template with PBS resource directives; freeze/compress is gated by DeePMD's `finished training` log signal. |
| `DELLA_TIGER_train_1h.apptr.Ngpu.restart.TF.sh` | Self-resubmitting Della/Tiger TF checkpoint-restart reference: `sbatch` self-chain, `CHAIN_HISTORY.tsv`/`CHAIN_ATTEMPTS.txt` logs, duplicate-submit marker, max-chain cap, health gate, numeric checkpoint selection, rollback guard, no `--skip-neighbor-stat`. |
| `DELLA_TIGER_train_1h.apptr.Ngpu.restart.PT.sh` | Self-resubmitting Della/Tiger PT checkpoint-restart reference: `sbatch` self-chain, `CHAIN_HISTORY.tsv`/`CHAIN_ATTEMPTS.txt` logs, duplicate-submit marker, max-chain cap, health gate, numeric checkpoint selection, rollback guard, no `--skip-neighbor-stat`. |
| `ALCF_POLARIS_train_1h.apptr.Ngpu.restart.TF.sh` | Self-resubmitting Polaris TF checkpoint-restart reference using `qsub` self-chain, `CHAIN_HISTORY.tsv`/`CHAIN_ATTEMPTS.txt` logs, and the same safeguards. |
| `ALCF_POLARIS_train_1h.apptr.Ngpu.restart.PT.sh` | Self-resubmitting Polaris PT checkpoint-restart reference using `qsub` self-chain, `CHAIN_HISTORY.tsv`/`CHAIN_ATTEMPTS.txt` logs, and the same safeguards. |
| `submit_*.sh` | Historical submission helpers used from the scratch experiment root. |
