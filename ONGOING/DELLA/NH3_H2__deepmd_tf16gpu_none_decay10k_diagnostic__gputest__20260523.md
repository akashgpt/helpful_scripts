# NH3/H2 DeePMD TF 16GPU `none` Decay10k Diagnostic

## Status Snapshot

- Date: 2026-05-23
- Cluster: Della
- Slurm partition/QOS: `gputest` / `gpu-test`
- Job ID: `8647911`
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/none_100k/16gpu_100k_none_decay10k`

## Purpose

Companion to the completed `8gpu_100k_none_decay10k` diagnostic. Test whether a
production-like 10k LR decay interval also keeps TF/Horovod `scale_by_worker =
none` stable at 16 GPUs.

## One-Variable Change From 16GPU Baseline

Source case:

`/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/none_100k/16gpu_100k_none`

Changed:

- `learning_rate.decay_steps`: `100000` -> `10000`
- Slurm walltime: `02:00:00` -> `01:00:00`

Unchanged:

- Framework: TF/Horovod
- GPUs: 16 total = 4 nodes x 4 A100/node
- `training.numb_steps = 100000`
- `learning_rate.scale_by_worker = none`
- `learning_rate.start_lr = 0.001`
- `training.save_freq = 100`

## Related Existing Job

Job `8375105` was an older 16GPU 100k none submission on the regular `gpu` partition. It was cancelled at user request on 2026-05-23 and should be ignored; this diagnostic is the active 16GPU test.

## How To Check

```bash
squeue -j 8647911
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/none_100k/16gpu_100k_none_decay10k
tail -n 5 lcurve.out
ls -lh slurm-*.out gpu_mem_util_*.csv
```

## Next Step Once Done

Compare training stability against:

- `8gpu_100k_none_decay10k`
- old `8gpu_100k_none`
- original/pending `16gpu_100k_none`
- PT 16GPU `none`

If stable, freeze/compress/pseudo-validate final and best available checkpoints.

Do not remove this ONGOING note without explicit user confirmation.

## Completion / Analysis Snapshot

Updated: 2026-05-24

Job 8647911 did not complete the 100k target. It timed out at the 1 hour wall limit after reaching step 77270. Final lcurve row: 77270  1.65e+03  1.42e+01  1.56e+01  7.38e+01  3.2e-07. Treat this as an unsuccessful diagnostic rather than a finished 100k result; the training loss had already blown up badly, so no final validation was launched for this case.
