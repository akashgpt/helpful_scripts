# DeePMD TF/Horovod Global-Batch Experiment Plan

Date: 2026-05-17

Status: ongoing. The first short-run and reused-run matrix is complete and has
been pseudo-validated. A 10x-step follow-up matrix was submitted on
2026-05-17 with 1-hour walltime caps and is still running/pending at the latest
benchmark snapshot.

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

## New Runs

The planned new runs are split into three groups:

| Group | Purpose |
|---|---|
| `sample_matched` | Keep approximate rank-batches comparable to 1 GPU x 10k. |
| `lr_sensitivity` | Test DeePMD `scale_by_worker` choices for promising 4/8 GPU cases. |
| `walltime_matched` | Spend about the same wall time as the 1-GPU 10k baseline. |

The generator marks every case in `EXPERIMENT_MATRIX.tsv` and writes per-run
`myinput.json`, `run_srun_train_mem.sbatch`, and `RUN_INFO.md` files.

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

Do not submit the whole matrix blindly. Start with `sample_matched`, inspect
validation/test behavior, then decide whether `walltime_matched` and 16-GPU
cases are worth the allocation.

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

Pseudo-validation aggregate RMSEs are now reported with asymmetric 1-sigma and
2-sigma bootstrap intervals. The bootstrap resamples validation systems while
keeping each system's frame count as the weighting factor for the aggregate
RMSE.

For this TF/Horovod setup, larger GPU counts are data-parallel large-batch
experiments. They are not expected to reduce per-step time automatically.
