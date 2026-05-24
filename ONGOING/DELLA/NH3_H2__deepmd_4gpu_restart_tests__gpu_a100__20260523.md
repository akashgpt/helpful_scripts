# NH3/H2 DeePMD 4GPU Restart Tests

Updated: 2026-05-23
Cluster: Della (`della-gpu.princeton.edu`)
Scratch root: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517`
Experiment tree: `runs/restart_tests_20260523`

## Purpose

Explore checkpoint-based restart behavior for 4GPU DeePMD TensorFlow/Horovod
and PyTorch training. Each submitted job restarts from an existing checkpoint
and targets 100000 additional optimizer steps with a 1 hour wall-time cap.

DeePMD restart uses `numb_steps` as the absolute final optimizer step, not as
an increment. Therefore the 10k source uses `numb_steps = 110000`, and the
100k sources use `numb_steps = 200000`.

## Jobs

| Case | Job ID | Backend | Work dir | Start checkpoint | Target final step | Script |
|---|---:|---|---|---|---:|---|
| `tf4g_10k_restart_plus100k` | 8641436 | TF/Horovod | `runs/restart_tests_20260523/tf4g_10k_restart_plus100k` | `runs/update_matched/4gpu/model-compression/model.ckpt-10000` | 110000 | `run_srun_train_mem.sbatch` |
| `tf4g_100k_restart_plus100k` | 8641437 | TF/Horovod | `runs/restart_tests_20260523/tf4g_100k_restart_plus100k` | `runs/none_100k/4gpu_100k_none/model-compression/model.ckpt-100000` | 200000 | `run_srun_train_mem.sbatch` |
| `pt4g_100k_restart_plus100k` | 8641438 | PyTorch | `runs/restart_tests_20260523/pt4g_100k_restart_plus100k` | `runs/pt_none_100k/4gpu_100k_none_pt/model-compression/model.ckpt-100000.pt` | 200000 | `run.sbatch` |

Status snapshot, 2026-05-23 16:10 EDT:

```text
8641436  RUNNING  della-l06g12  elapsed ~00:20
8641437  RUNNING  della-l06g12  elapsed ~00:19
8641438  RUNNING  della-l08g2   elapsed ~00:14
```

Early log check: `8641436` started DeePMD/Horovod on 4 visible A100 GPUs and
entered DeePMD initialization. The repeated PMIx `psec/munge` messages match
previous Della/Horovod noise; no fatal restart error was visible in the first
log chunk. No `lcurve.out` existed yet at this snapshot.

## Restart Commands

TF jobs use the locally validated restart syntax from the earlier
`reuse_16gpu_10k_10x_from50k` continuation:

```bash
dp train --mpi-log=workers --skip-neighbor-stat myinput.json --restart <tf_checkpoint_prefix>
```

PT syntax was confirmed from the installed DeePMD CLI help on Della:

```bash
dp --pt train myinput.json --restart <pt_checkpoint_file.pt>
```

The PT launcher wraps that command with `python -m torch.distributed.run
--nproc_per_node=4 --no-python`, matching the completed PT 4GPU 100k run.

## Check Status

```bash
squeue -j 8641436,8641437,8641438
sacct -j 8641436,8641437,8641438 --format=JobID,JobName%24,State,Elapsed,Timelimit,ExitCode
```

Useful live logs:

```bash
tail -f runs/restart_tests_20260523/tf4g_10k_restart_plus100k/slurm-8641436.out
tail -f runs/restart_tests_20260523/tf4g_100k_restart_plus100k/slurm-8641437.out
tail -f runs/restart_tests_20260523/pt4g_100k_restart_plus100k/slurm-8641438.out
```

## Next Step

After the jobs finish or hit wall time, inspect each `slurm-*.out`,
`lcurve.out`, and `model-compression/checkpoint` to confirm whether restart
advanced from the starting checkpoint and how many additional steps completed.
Do not run validation yet unless explicitly requested.

## Completion Snapshot, 2026-05-23 17:05 EDT

All three restart jobs completed with Slurm state `COMPLETED` and exit code `0:0`:

| Case | Job ID | Elapsed | Final lcurve step | Notes |
|---|---:|---:|---:|---|
| `tf4g_10k_restart_plus100k` | 8641436 | 00:41:20 | 110000 | Restarted from step 10000 and reached the absolute target step 110000. |
| `tf4g_100k_restart_plus100k` | 8641437 | 00:47:19 | 200000 | Restarted from step 100000 and reached the absolute target step 200000. |
| `pt4g_100k_restart_plus100k` | 8641438 | 01:04:23 | 200000 | Restarted from step 100000 and reached the absolute target step 200000. |

Stitched training-curve outputs were written in a separate plot folder:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/training_loss_plots_10x_20260521/CONTINUATIONS/
```

Files:

- `TF_PT_4GPU_RESTART_STITCHED_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png`
- `TF_PT_4GPU_RESTART_STITCHED_TRAINING_EVOLUTION_SUMMARY_20260523.tsv`
- `TF_PT_4GPU_RESTART_STITCHED_TRAINING_EVOLUTION_ANALYSIS_20260523.md`

Quick read from final continuation rows:

| Case | Final TRAIN total | Final TRAIN F RMSE (eV/A) |
|---|---:|---:|
| `tf4g_10k_restart_plus100k` | 1.96 | 0.968 |
| `tf4g_100k_restart_plus100k` | 1.25 | 0.614 |
| `pt4g_100k_restart_plus100k` | 2.09 | 0.692 |

Next scientific step, if requested: freeze/compress and pseudo-validate final and best continuation checkpoints before ranking them against the original 100k models.

