# NH3/H2 DeePMD TF 4GPU `none` Seed Stability

Last updated: 2026-05-23 19:25 EDT

## Purpose

Run additional TF/Horovod 4GPU/100k `scale_by_worker = none` training
replicates to test whether the stable `4gpu_100k_none` behavior holds across
different random seeds.

Originally seeds 02-09 were queued. On 2026-05-23, the never-started pending
jobs for seeds 06-09 were cancelled and their run folders deleted at user
request. Keep/analyze only seeds that completed or had already started.

## Working Directory

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/seed_replicates_4gpu_none_100k
```

Source template:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/none_100k/4gpu_100k_none
```

## Shared Settings

- Framework: TF/Horovod DeePMD
- Partition: `gputest`
- GPUs: 4 A100 on 1 node
- Wall time: 1 hour
- `training.numb_steps = 100000`
- `learning_rate.decay_steps = 100000`
- `learning_rate.scale_by_worker = none`
- `training.save_freq = 100`
- Changed per replicate: `model.descriptor.seed`, `model.fitting_net.seed`, and `training.seed`

## Jobs

| Job ID | Name | Run directory | Seed | Status snapshot |
|---:|---|---|---:|---|
| 8642131 | `tf4g_none_s02` | `4gpu_100k_none_seed02` | 2 | completed |
| 8642132 | `tf4g_none_s03` | `4gpu_100k_none_seed03` | 3 | completed |
| 8642133 | `tf4g_none_s04` | `4gpu_100k_none_seed04` | 4 | completed |
| 8642134 | `tf4g_none_s05` | `4gpu_100k_none_seed05` | 5 | completed |
| 8642135 | `tf4g_none_s06` | `4gpu_100k_none_seed06` | 6 | cancelled before start; folder deleted |
| 8642136 | `tf4g_none_s07` | `4gpu_100k_none_seed07` | 7 | cancelled before start; folder deleted |
| 8642137 | `tf4g_none_s08` | `4gpu_100k_none_seed08` | 8 | cancelled before start; folder deleted |
| 8642138 | `tf4g_none_s09` | `4gpu_100k_none_seed09` | 9 | cancelled before start; folder deleted |

## Check Commands

```bash
squeue -j 8642133,8642134 -o '%i %.18j %.8T %.10M %.9P %R'
sacct -j 8642131,8642132,8642133,8642134,8642135,8642136,8642137,8642138 --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

## Next Step After Completion

1. Check each `slurm-*.out` for `JOB_END` and any DeePMD/Horovod errors.
2. Summarize each `lcurve.out`: final step, final train total, best train
   total, late rolling median/mean, and whether any blow-up occurred.
3. Plot all seed curves against the original `4gpu_100k_none`.
4. If stable, update the benchmark conclusion that TF 4GPU `none` is robust
   across seeds for this NH3/H2 dataset. If any seed fails, freeze/compress and
   pseudo-validate final and best checkpoints before deciding whether it is an
   optimizer instability or just training-loss noise.


## Completion / Analysis Snapshot

Updated: 2026-05-24

Seeds 02-05 all completed the 100k-step TF/Horovod 4GPU none training. Seeds 06-09 were cancelled before starting and their folders were deleted at user request. The completed seed runs look stable by training curve: no seed shows the catastrophic blowup observed in the older 8GPU TF case. Final-checkpoint pseudo-validation completed separately under job array 8647865 for seeds 02-05.
