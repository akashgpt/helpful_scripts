# DeePMD TF/Horovod Global-Batch Benchmark

Date: 2026-05-17

Status: reference snapshot. The first short-run/reused-run matrix and the
completed 10x-step follow-up matrix have both been pseudo-validated. Separate
fresh 16GPU/100k/90m and 8-node/32GPU/3125-step diagnostic jobs were still
pending at the latest snapshot, but they are not part of the completed 10x
matrix archived here.

Current result snapshot:

```text
RESULTS_SO_FAR.md
```

Archived reference outputs and scripts:

```text
reference_results/
reference_scripts/
```

This experiment bundle tests whether larger Horovod global batch sizes improve
NH3/H2 DeePMD training quality per wall time or GPU-hour. It deliberately reuses
the completed 10k scaling runs under:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422
```

The runnable helper scripts and generated run directories live under the
existing scratch benchmark root:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/
```

The generator creates only the missing runs in a separate scratch tree:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517
```

## Scientific Questions

1. Do larger global batches improve validation/test RMSE at fixed optimizer-step
   count?
2. If `numb_steps` and `decay_steps` are reduced by worker count, does
   multi-GPU training reach similar quality faster?
3. At fixed wall time, do 8/16 GPU runs produce better models than the 1-GPU
   standard?
4. Which configuration is best by GPU-hour, not just wall time?

## Reused Runs

These completed runs should not be repeated:

| Role | GPUs | Steps | Path |
|---|---:|---:|---|
| update-matched baseline | 1 | 10000 | `10k_gpu_scaling_loss_rerun/1gpu` |
| update-matched | 2 | 10000 | `10k_gpu_scaling_loss_rerun/2gpu` |
| update-matched | 4 | 10000 | `10k_gpu_scaling_loss_rerun/4gpu` |
| update-matched | 8 | 10000 | `multinode_scaling_10k_20260516/2node_8gpu` |
| update-matched | 16 | 10000 | `multinode_scaling_10k_20260516/4node_16gpu` |

## Experiment Groups

The new runs were split into these groups:

| Group | Purpose |
|---|---|
| `sample_matched` | Keep approximate rank-batches comparable to 1 GPU x 10k. |
| `lr_sensitivity` | Test DeePMD `scale_by_worker` choices for promising 4/8 GPU cases. |
| `walltime_matched` | Spend about the same wall time as the 1-GPU 10k baseline. |
| `long_steps_10x` | Repeat the matrix with 10x more optimizer steps and 1-hour caps. |
| `long_steps_10x_continuations` | Continue the cancelled 16GPU/100k case from its 50k checkpoint. |
| `long_steps_10x_extra` | Extra post-analysis diagnostics such as the pending 8-node/32GPU/3125-step run. |

The generator marked every case in the scratch matrices and wrote per-run
`myinput.json`, Slurm scripts, and `RUN_INFO.md` files. This benchmark folder
keeps only reference copies of matrices, summaries, and scripts.

## Usage

Preview what will be created:

```bash
python scripts/materialize_experiments.py --dry-run
```

Create the scratch tree:

```bash
python scripts/materialize_experiments.py
```

Then submit selectively from the generated scratch root:

```bash
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517
bash submit_sample_matched.sh
```

The runs in this benchmark snapshot have already been submitted and analyzed;
use these commands only as historical reference or as templates for a new
scratch-side experiment.

## Evaluation

Training `lcurve.out` is useful but not sufficient. For scientific decisions,
run the same `dp test` validation/test protocol on all completed checkpoints and
compare:

```text
best validation/test RMSE_f
best validation/test RMSE_e
best validation/test RMSE_v
wall time to best checkpoint
GPU-hours to best checkpoint
rank-batches ~= GPUs * steps * local batch size
```

Pseudo-validation aggregate RMSEs for the 10x completed matrix are reported
with explicit bootstrap percentiles. The bootstrap resamples validation systems
while keeping each system's frame count as the weighting factor for the
aggregate RMSE. The archived percentile columns are p0.135, p15.865, p50,
p84.135, and p99.865, so the median and both 1-sigma/3-sigma
normal-equivalent percentile bounds are visible directly.

Metric units follow the DeePMD `dp test` log output. Energy is
`Energy RMSE/Natoms` in eV/atom, force is `Force RMSE` in eV/A, and the third
reported pseudo-validation metric is `Virial RMSE/Natoms` in eV per atom, not
stress converted to GPa.

For this TF/Horovod setup, larger GPU counts are data-parallel large-batch
experiments. They are not expected to reduce per-step time automatically.

## Current Inference

Energy RMSE/atom (eV/atom) is the primary ranking metric in the completed 10x analysis.
The leading energy group is practically tight:

| Case | GPUs | Steps | Wall s | Energy RMSE/atom (eV/atom) | Force RMSE (eV/A) | Virial RMSE/Natoms (eV/atom) |
|---|---:|---:|---:|---:|---:|---:|
| `reuse_4gpu_10k_10x` | 4 | 100000 | 2109.329 | 0.177581 | 0.655888 | 0.075694 |
| `2gpu_5k_linear_10x` | 2 | 50000 | 1079.118 | 0.181000 | 0.686653 | 0.051985 |
| `reuse_1gpu_10k_10x` | 1 | 100000 | 2076.592 | 0.183390 | 0.692176 | 0.054289 |
| `4gpu_2500_linear_10x` | 4 | 25000 | 567.368 | 0.184682 | 0.701653 | 0.052845 |
| `8gpu_1250_sqrt_10x` | 8 | 12500 | 347.398 | 0.189304 | 0.791522 | 0.062366 |
| `16gpu_625_linear_10x` | 16 | 6250 | 275.955 | 0.197025 | 0.780122 | 0.083433 |

If wall time is the dominant constraint and GPU allocation is available,
`16gpu_625_linear_10x` is a reasonable fast option: about 7.5x faster than the
1GPU/100k case, 3.9x faster than the 2GPU/50k case, 2.1x faster than the
4GPU/25k case, and 1.26x faster than the 8GPU/1250 sqrt case. If GPU-hours and
robustness matter, the 2GPU/4GPU cases are better balanced. The conservative
default remains 1GPU for debugging/baselines and 4GPU for practical
acceleration, with `4gpu_2500_linear_10x` as the best "fast but not wasteful"
setting from the completed 10x matrix.

The 10x follow-up changed the early-read conclusions in two ways. First,
longer training rescued some short-step cases: the original `2gpu_5k_linear`
looked like a clear failure at 5k steps, but `2gpu_5k_linear_10x` is in the
leading energy group at 50k steps. Second, longer training exposed instability
in several large-GPU long-step schedules: `reuse_2gpu_10k_10x`,
`reuse_8gpu_10k_10x`, `8gpu_7k_linear_10x`, `16gpu_5k_linear_10x`, and
`4gpu_2500_sqrt_10x` all end as poor final checkpoints.

The practical trend is therefore not "more GPUs plus more steps is always
better." Scaled-step schedules such as 2GPU/50k, 4GPU/25k, 8GPU/12.5k sqrt,
and 16GPU/6.25k linear are the useful large-batch experiments. Update-matched
or walltime-matched large-GPU long-step schedules need validation-based
checkpoint selection and probably a redesigned learning-rate schedule before
they should be treated as production candidates.
