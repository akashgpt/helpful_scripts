# Global-Batch DeePMD Experiment Analysis, 2026-05-17

Working root:

`/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517`

## Scope

This reference analysis combines:

- reused 10k GPU-scaling runs copied into the scratch experiment tree;
- new 15-minute tests in `sample_matched`, `lr_sensitivity`,
  `walltime_matched`, and `walltime_and_long_baseline`;
- completed 10x-step follow-up runs in `runs/long_steps_10x`;
- the 16GPU/100k continuation from the 50k checkpoint in
  `runs/long_steps_10x_continuations`;
- pseudo-validation by freezing/compressing completed checkpoints and running
  `dp test` against the shared DFT-labeled pseudo-validation systems.

The pseudo-validation systems are:

- `v7_i34/md/*/pre/recal/set_train/deepmd` and `set_test/deepmd`;
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i16/md/*/pre/recal/set_train/deepmd` and `set_test/deepmd`.

This is a pseudo-validation set, not a fully independent validation set,
because the configurations trace back to active-learning trajectories generated
by a related `pv_comp.pb` model before VASP DFT recalculation. It is still a
useful DFT-labeled challenge set for this model family.

## Output Files

Original short/reused matrix:

- `TRAINING_SUMMARY.tsv`
- `PSEUDO_VALIDATION_SUMMARY.tsv`
- `PSEUDO_VALIDATION_SYSTEMS.tsv`

Completed 10x matrix:

- `TRAINING_SUMMARY_10x_COMPLETED.tsv`
- `EXPERIMENT_MATRIX_10x_COMPLETED_FOR_PSEUDOVAL.tsv`
- `PSEUDO_VALIDATION_SUMMARY_10x_COMPLETED.tsv`
- `METRIC_UNITS.md`

Scripts archived for reproducibility:

- `reference_scripts/build_10x_completed_analysis.py`
- `reference_scripts/run_pseudo_validation_10x.sbatch`
- `reference_scripts/summarize_pseudo_validation_any.py`

The pseudo-validation scripts use local copies/adaptations of ALCHEMY patterns
only. Nothing in `ALCHEMY__dev` was modified.

## Metric Conventions

Pseudo-validation E/F/virial values are parsed from DeePMD `dp test` logs and
aggregated as frame-count-weighted means of per-system RMSE values.

- `Energy RMSE/atom` means DeePMD `Energy RMSE/Natoms`, in eV/atom.
- `Force RMSE` means DeePMD `Force RMSE`, in eV/A.
- `Virial RMSE/Natoms` means DeePMD `Virial RMSE/Natoms`, in eV per atom; it
  is not stress converted to GPa.
- `Total RMSE` is the final training total from `lcurve.out`, not a
  pseudo-validation metric.
- Percentile tuples are ordered as p15.865/p50/p84.135 unless otherwise noted.

## Completed 10x Ranking

The 10x completed analysis is sorted by frame-weighted pseudo-validation
energy RMSE/atom over all 2547 pseudo-validation frames. Energy is the primary
ranking metric here, but the table keeps final training total RMSE, force RMSE,
and virial RMSE visible. Energy percentiles are bootstrap percentiles over
validation systems, preserving frame weighting.

| Case | GPUs | Steps | Wall s | Total RMSE | Energy RMSE/atom (eV/atom) | Energy p15.865/p50/p84.135 | Force RMSE (eV/A) | Virial RMSE/Natoms (eV/atom) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reuse_4gpu_10k_10x` | 4 | 100000 | 2109.329 | 0.997 | 0.177581 | 0.171740/0.177876/0.183535 | 0.655888 | 0.075694 |
| `2gpu_5k_linear_10x` | 2 | 50000 | 1079.118 | 1.03 | 0.181000 | 0.174224/0.181103/0.187784 | 0.686653 | 0.051985 |
| `reuse_1gpu_10k_10x` | 1 | 100000 | 2076.592 | 1.3 | 0.183390 | 0.176372/0.183571/0.190311 | 0.692176 | 0.054289 |
| `4gpu_2500_linear_10x` | 4 | 25000 | 567.368 | 1.32 | 0.184682 | 0.177629/0.184766/0.191526 | 0.701653 | 0.052845 |
| `8gpu_1250_sqrt_10x` | 8 | 12500 | 347.398 | 0.86 | 0.189304 | 0.182505/0.189484/0.196028 | 0.791522 | 0.062366 |
| `16gpu_625_linear_10x` | 16 | 6250 | 275.955 | 1.51 | 0.197025 | 0.190047/0.197057/0.203656 | 0.780122 | 0.083433 |
| `4gpu_2500_none_10x` | 4 | 25000 | 562.884 | 1.89 | 0.197189 | 0.189677/0.197146/0.204454 | 0.850935 | 0.063014 |
| `8gpu_1250_none_10x` | 8 | 12500 | 343.402 | 1.34 | 0.199872 | 0.192313/0.199796/0.207358 | 0.852042 | 0.064143 |
| `reuse_8gpu_10k_10x` | 8 | 100000 | 3215.042 | 11.4 | 0.294202 | 0.282922/0.294551/0.305806 | 3.035107 | 0.545935 |
| `reuse_16gpu_10k_10x_from50k` | 16 | 100000 | 2337.833 | 21 | 2.276191 | 2.061689/2.272114/2.481947 | 2.070921 | 0.864581 |
| `8gpu_1250_linear_10x` | 8 | 12500 | 350.347 | 75.1 | 2.584377 | 2.516315/2.585582/2.653173 | 0.754503 | 0.505609 |
| `16gpu_5k_linear_10x` | 16 | 50000 | 2112.933 | 98.1 | 4.295180 | 4.198841/4.297243/4.392560 | 1.133693 | 0.535195 |
| `4gpu_2500_sqrt_10x` | 4 | 25000 | 563.580 | 171 | 4.303055 | 4.078259/4.299717/4.531679 | 3.767129 | 5.121701 |
| `reuse_2gpu_10k_10x` | 2 | 100000 | 2104.448 | 491 | 12.611698 | 11.930820/12.603242/13.307674 | 13.010718 | 16.416686 |
| `8gpu_7k_linear_10x` | 8 | 70000 | 1893.203 | 396 | 17.579198 | 17.253399/17.570189/17.896970 | 4.923031 | 3.791055 |

## Interpretation

The leading energy group is practically tight:

- `reuse_4gpu_10k_10x`
- `2gpu_5k_linear_10x`
- `reuse_1gpu_10k_10x`
- `4gpu_2500_linear_10x`
- `8gpu_1250_sqrt_10x`
- `16gpu_625_linear_10x`

The first four are nearly indistinguishable by energy RMSE/atom when comparing
their bootstrap percentile ranges. The 8GPU/1250 sqrt and 16GPU/625 linear
cases have slightly higher energy RMSE, but remain close enough that wall-time
constraints can justify considering them.

`16gpu_625_linear_10x` is especially important for wall-clock constrained
training: it reaches a competitive model in 275.955 s, which is about 7.5x
faster than the 1GPU/100k case, 3.9x faster than the 2GPU/50k case, 2.1x
faster than the 4GPU/25k case, and 1.26x faster than the 8GPU/1250 sqrt case.
Its energy RMSE is a little worse than the 1/2/4GPU group, but not
catastrophically. Its force RMSE is similar to the 8GPU case and not far from
the 4GPU case. Its Virial RMSE/Natoms is somewhat worse than the 2/4/8GPU
short-step cases, but still nowhere near the failure cases. If node/GPU
allocation is plentiful and wall time dominates, this is a reasonable
fast-training option.

If GPU-hours, queue friendliness, and robustness matter, the 2GPU/50k and
4GPU/25k cases are better balanced. They give nearly the same energy as the
1GPU/100k baseline with much shorter wall time and modest GPU-second cost.
If one wants the most balanced "fast but not wasteful" setting,
`4gpu_2500_linear_10x` is probably the sweet spot.

The large-GPU long-step cases do not support the simple rule that many GPUs for
one hour equals fewer GPUs for many hours. Several are clear failures by energy,
force, virial, and final training total RMSE:

- `reuse_2gpu_10k_10x`
- `8gpu_7k_linear_10x`
- `4gpu_2500_sqrt_10x`
- `16gpu_5k_linear_10x`
- `reuse_16gpu_10k_10x_from50k`
- `reuse_8gpu_10k_10x`

## 10k Versus 10x Trends

The 10x follow-up changed the interpretation in a useful way. The original
10k/short matrix was good for throughput and early-training diagnostics, but it
was not enough to identify which schedules remain stable when allowed to train
longer.

| Case family | GPUs | Short steps | 10x steps | Energy short -> 10x | Force short -> 10x | Main read |
|---|---:|---:|---:|---:|---:|---|
| `reuse_1gpu_10k` -> `reuse_1gpu_10k_10x` | 1 | 10000 | 100000 | 0.216 -> 0.183 | 1.183 -> 0.692 | improved; standard baseline becomes much stronger |
| `reuse_2gpu_10k` -> `reuse_2gpu_10k_10x` | 2 | 10000 | 100000 | 0.203 -> 12.612 | 0.875 -> 13.011 | failed with long update-matched training |
| `reuse_4gpu_10k` -> `reuse_4gpu_10k_10x` | 4 | 10000 | 100000 | 0.146 -> 0.178 | 0.762 -> 0.656 | force improves; energy remains competitive but not monotonic |
| `reuse_8gpu_10k` -> `reuse_8gpu_10k_10x` | 8 | 10000 | 100000 | 0.194 -> 0.294 | 0.766 -> 3.035 | degraded under long update-matched training |
| `2gpu_5k_linear` -> `2gpu_5k_linear_10x` | 2 | 5000 | 50000 | 1.665 -> 0.181 | 3.534 -> 0.687 | short-run failure was an early-training artifact |
| `4gpu_2500_linear` -> `4gpu_2500_linear_10x` | 4 | 2500 | 25000 | 0.194 -> 0.185 | 0.954 -> 0.702 | improved; balanced fast option |
| `4gpu_2500_sqrt` -> `4gpu_2500_sqrt_10x` | 4 | 2500 | 25000 | 0.221 -> 4.303 | 1.178 -> 3.767 | failed when trained longer |
| `4gpu_2500_none` -> `4gpu_2500_none_10x` | 4 | 2500 | 25000 | 0.263 -> 0.197 | 1.602 -> 0.851 | improved, but still behind linear 2/4GPU |
| `8gpu_1250_linear` -> `8gpu_1250_linear_10x` | 8 | 1250 | 12500 | 0.200 -> 2.584 | 0.911 -> 0.755 | energy failed despite acceptable force |
| `8gpu_1250_sqrt` -> `8gpu_1250_sqrt_10x` | 8 | 1250 | 12500 | 0.209 -> 0.189 | 1.017 -> 0.792 | stable and competitive for wall time |
| `8gpu_1250_none` -> `8gpu_1250_none_10x` | 8 | 1250 | 12500 | 0.255 -> 0.200 | 1.672 -> 0.852 | improved, but not a leading choice |
| `16gpu_625_linear` -> `16gpu_625_linear_10x` | 16 | 625 | 6250 | 0.209 -> 0.197 | 1.057 -> 0.780 | improved; best wall-clock-only option |
| `8gpu_7k_linear` -> `8gpu_7k_linear_10x` | 8 | 7000 | 70000 | 0.192 -> 17.579 | 0.789 -> 4.923 | failed badly when trained longer |
| `16gpu_5k_linear` -> `16gpu_5k_linear_10x` | 16 | 5000 | 50000 | 0.221 -> 4.295 | 0.862 -> 1.134 | failed by energy |

Observed trends:

1. The standard 1GPU case benefits from longer training: energy improves from
   0.216 to 0.183 eV/atom and force from 1.183 to 0.692 eV/A.
2. The 4GPU update-matched case remains excellent, but energy is not monotonic:
   10k had lower pseudo-validation energy than 100k, while force improved. This
   reinforces that final checkpoint selection should use validation, not only
   scripted step count.
3. Scaled-step schedules are the reliable large-batch path. `2gpu_5k_linear`,
   `4gpu_2500_linear`, `8gpu_1250_sqrt`, and `16gpu_625_linear` all become
   reasonable after the 10x follow-up.
4. Long update-matched or walltime-matched large-GPU runs are risky. Several
   pass through better training-loss regions, then end at much worse final
   checkpoints. The lcurve evidence showed unstable trajectories rather than
   validation error smoothly improving to the end.
5. Learning-rate scaling is not one-size-fits-all. Linear scaling works well
   for 2GPU/50k and 4GPU/25k, but 8GPU/12.5k linear fails in energy; 8GPU sqrt
   is the more stable 8GPU scaled-step option in this matrix.
6. More data throughput is not equivalent to proportionally more useful
   optimization. A many-GPU one-hour run can be attractive when wall time is the
   only constraint, but it is not automatically comparable to fewer GPUs for
   many hours unless the schedule and checkpoint selection are redesigned around
   global batch size.

## Recommendation

Use the choice of GPU count according to the actual bottleneck:

1. Use 1GPU for cheap debugging and canonical serial baselines.
2. Use 2GPU/50k or 4GPU/25k when balancing wall time and GPU-hours.
3. Use 8GPU/1250 sqrt or 16GPU/625 linear only when wall time is the dominant
   constraint and GPU allocation is available.
4. Do not routinely use 8GPU or 16GPU long-step/update-matched schedules
   without redesigning the schedule around global batch size, learning-rate
   behavior, validation stopping, and downstream MD stability.

## Remaining Follow-Up

Two jobs were submitted separately after this completed 10x matrix and were
still pending at this snapshot:

| Job ID | Case | Purpose |
|---:|---|---|
| 8375105 | `dpgb16g_100k_90m` | Fresh 16GPU/100k/90m diagnostic. |
| 8381309 | `8node_32gpu_3125_linear_10x` | Extra 8-node/32GPU wall-time diagnostic, linear scaling, 3125 steps = 100000/32. |

They should be analyzed separately when they finish, then compared against this
completed 10x matrix using the same freeze/compress/`dp test`
pseudo-validation protocol.
