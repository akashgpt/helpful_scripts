# NH3/H2 DeePMD TF 8GPU `none` Decay10k Diagnostic

## Status Snapshot

- Date: 2026-05-23
- Cluster: Della
- Slurm partition/QOS: `gputest` / `gpu-test`
- Job ID: `8644423`
- Final scheduler state: `COMPLETED`, elapsed `00:50:12`, exit `0:0`
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/none_100k/8gpu_100k_none_decay10k`

## Purpose

Rerun the problematic TF/Horovod `8gpu_100k_none` benchmark with a more
production-like LR decay interval. This tests whether the bad 8GPU `none`
result was partly due to keeping the LR too high/coarse for the whole 100k run.

## One-Variable Change

Source case:

`/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/none_100k/8gpu_100k_none`

Changed:

- `learning_rate.decay_steps`: `100000` -> `10000`

Unchanged:

- Framework: TF/Horovod
- GPUs: 8 total = 2 nodes x 4 A100/node
- `training.numb_steps = 100000`
- `learning_rate.scale_by_worker = none`
- `learning_rate.start_lr = 0.001`
- `training.save_freq = 100`
- Wall time: 1 hour

## How To Check

```bash
squeue -j 8644423
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/none_100k/8gpu_100k_none_decay10k
tail -n 5 lcurve.out
ls -lh slurm-*.out gpu_mem_util_*.csv
```

## Next Step Once Done

Compare against:

- `runs/none_100k/8gpu_100k_none`
- `runs/none_100k/4gpu_100k_none`
- PT 8GPU `none` result

Main question: does `decay_steps = 10000` remove or reduce the 8GPU TF loss
blowup while keeping `scale_by_worker = none`?

Do not remove this ONGOING note without explicit user confirmation.

## Completion / Analysis Snapshot

Updated: 2026-05-24

This run finished cleanly at step 100000 with Slurm state COMPLETED. Final lcurve row: 100000  1.09e+00  2.38e-03  6.22e-01  3.55e-02  1.0e-08. Compared with the older 8gpu_100k_none run, decay_steps = 10000 removed the catastrophic loss blowup, supporting the interpretation that the old 8GPU TF failure was strongly influenced by the learning-rate schedule rather than simply 8 GPUs with scale_by_worker = none.
