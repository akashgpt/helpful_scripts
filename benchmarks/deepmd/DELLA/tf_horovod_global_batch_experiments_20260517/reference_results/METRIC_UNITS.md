# Metric Units

This benchmark uses two different metric sources.

## Pseudo-Validation TSVs

Files:

- `PSEUDO_VALIDATION_SUMMARY.tsv`
- `PSEUDO_VALIDATION_SUMMARY_10x_COMPLETED.tsv`

These values are parsed from DeePMD `dp test` logs and aggregated as a
frame-count-weighted mean of per-system RMSE values.

| Column | Source log label | Unit | Meaning |
|---|---|---|---|
| `energy_rmse_per_atom` | `Energy RMSE/Natoms` | eV/atom | Per-atom energy RMSE from `dp test`. |
| `force_rmse` | `Force RMSE` | eV/A | Cartesian force RMSE from `dp test`. |
| `virial_rmse_per_atom` | `Virial RMSE/Natoms` | eV/atom | Per-atom-normalized virial RMSE from `dp test`; not stress in GPa. |
| `*_p0p135` | bootstrap over systems | same as metric | 0.135th percentile, approximately the lower 3-sigma normal-equivalent percentile. |
| `*_p15p865` | bootstrap over systems | same as metric | 15.865th percentile, approximately the lower 1-sigma normal-equivalent percentile. |
| `*_p50` | bootstrap over systems | same as metric | 50th percentile, the bootstrap median. |
| `*_p84p135` | bootstrap over systems | same as metric | 84.135th percentile, approximately the upper 1-sigma normal-equivalent percentile. |
| `*_p99p865` | bootstrap over systems | same as metric | 99.865th percentile, approximately the upper 3-sigma normal-equivalent percentile. |

## Training TSVs

Files:

- `TRAINING_SUMMARY.tsv`
- `TRAINING_SUMMARY_10x_COMPLETED.tsv`

These values are parsed from DeePMD `lcurve.out`. They are training-curve
metrics, not pseudo-validation metrics.

| Column | Source | Unit / convention |
|---|---|---|
| `final_rmse` | final `lcurve.out` total RMSE | DeePMD training loss/RMSE convention. |
| `final_rmse_e` | final `lcurve.out` energy RMSE | DeePMD training-curve energy convention. |
| `final_rmse_f` | final `lcurve.out` force RMSE | DeePMD training-curve force convention. |
| `final_rmse_v` | final `lcurve.out` virial RMSE | DeePMD training-curve virial convention. |
| `late_mean_rmse_f`, `late_best_rmse_f` | late-window `lcurve.out` force RMSE | DeePMD training-curve force convention. |
| `avg_train_s_per_batch`, `wall_time_s`, `gpu_seconds` | Slurm / training logs | seconds. |
