# NH3/H2 ML v4 MD and DeepMD global-batch background simulations

Created: 2026-05-18 17:28 EDT

Cluster: Della (`della9.princeton.edu`)

Requested note location: `/projects/bguf/akashgpt/run_scripts/helpful_scripts/ONGOING`

Actual note location: `/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/ONGOING/DELLA`

Note: `/projects/bguf/akashgpt/run_scripts/helpful_scripts` did not exist on `della9` when checked, so this note was placed in the available HELPFUL_SCRIPTS tree under `/projects/BURROWS`.

## Purpose

Track currently queued background simulations under these two scratch trees so the work can be resumed after a session crash or context loss:

- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422`

## Current queue snapshot

Captured with:

```bash
squeue -j 8205299,8205302,8205305,8205314,8205319,8205321,8205322,8205326,8205331,8205332,8375105,8381309 -o "%.18i %.9P %.32j %.8u %.2t %.10M %.10l %.6D %.20R"
```

At snapshot time all tracked jobs were pending. No matching local non-Slurm background processes were found with `pgrep -af sim_data_ML_v`.

| Job ID | Group | Partition | State | Time limit | Nodes | Reason | Working directory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `8205299` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_8/108H2_1NH3` |
| `8205302` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_9/90H2_10NH3` |
| `8205305` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_6/72H2_24NH3` |
| `8205314` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_3/2H2_71NH3` |
| `8205319` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_3/72H2_24NH3` |
| `8205321` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_6/6H2_54NH3` |
| `8205322` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_2/40H2_40NH3` |
| `8205326` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_4/2H2_71NH3` |
| `8205331` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_6/90H2_10NH3` |
| `8205332` | ML v4 MD | `gpu` | `PD` | `04:00:00` | 1 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_7/90H2_10NH3` |
| `8375105` | DeepMD global batch | `gpu` | `PD` | `01:30:00` | 4 | `Priority` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/long_steps_10x_fresh_90min/reuse_16gpu_10k_10x_90min` |
| `8381309` | DeepMD global batch | `gputest` | `PD` | `01:00:00` | 8 | `Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/long_steps_10x/32gpu_3125_linear_10x` |

## ML v4 MD details

Active path family:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md
```

The queued jobs are LAMMPS/PLUMED GPU MD continuation or sampling jobs for selected zones and compositions. Each uses:

```bash
sbatch --job-name=<MD_ZONE...> sub.lmp_plumed.gpu.4h.sh
```

Resource shape from `scontrol show job`:

- `Partition=gpu`
- `QOS=gpu-short`
- `TimeLimit=04:00:00`
- `NumNodes=1`
- `NumCPUs=2`
- `ReqTRES=cpu=2,mem=32G,node=1,billing=64,gres/gpu=1`
- `TresPerNode=gres/gpu:1`

Representative check command:

```bash
squeue -j 8205299,8205302,8205305,8205314,8205319,8205321,8205322,8205326,8205331,8205332
```

Representative resume pattern, from the relevant working directory:

```bash
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i22/md/ZONE_8/108H2_1NH3
sbatch --job-name=MD_ZONE_8_108H2_1NH3_$(date +%s) sub.lmp_plumed.gpu.4h.sh
```

Before resubmitting, inspect each zone/composition output directory for existing `slurm-<jobid>.out`, LAMMPS trajectory/log files, and restart files so duplicate MD segments are not launched accidentally.

## DeepMD global-batch details

Active path family:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517
```

Tracked queued jobs:

- `8375105`: `long_steps_10x_fresh_90min/reuse_16gpu_10k_10x_90min`, job name `dpgb16g_100k_90m`, `gpu`, 4 nodes, 16 A100 GPUs, 90 minutes.
- `8381309`: `long_steps_10x/32gpu_3125_linear_10x`, job name `dpgb10x_32g_3125_lin`, `gputest`, 8 nodes, 32 A100 GPUs, 60 minutes.

Both use:

```bash
sbatch run_srun_train_mem.sbatch
```

Representative check command:

```bash
squeue -j 8375105,8381309
```

Representative resume pattern:

```bash
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/long_steps_10x_fresh_90min/reuse_16gpu_10k_10x_90min
sbatch run_srun_train_mem.sbatch
```

For `8381309`, note the pending reason points to unavailable or reserved requested nodes in `gputest`. If it remains stuck, consider checking partition/node availability before resubmitting unchanged:

```bash
sinfo -p gputest
scontrol show jobid=8381309
```

## Next steps

1. Recheck the queue after the estimated starts pass:

```bash
squeue -u ag5805
```

2. For completed jobs, inspect Slurm outputs and generated data in the working directories above.

3. For failed or timed-out jobs, use `scontrol show jobid=<jobid>` and the relevant `slurm-<jobid>.out` file to decide whether to resubmit, change partition, shorten/extend walltime, or alter node count.

4. Keep this note in `ONGOING/DELLA` until the jobs are confirmed complete. Do not remove it without explicit confirmation.

## Reassessment - 2026-05-21 07:10 EDT

Current Della queue snapshot from `squeue -u ag5805`:

| Job ID | Name | State | Partition | Runtime / limit | Nodes | Working directory | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `8375105` | `dpgb16g_100k_90m` | `PENDING`, reason `Priority` | `gpu` | `0:00 / 01:30:00` | 4 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/long_steps_10x_fresh_90min/reuse_16gpu_10k_10x_90min` | Still queued; no output files beyond setup files yet. |
| `8533968` | `ML_v8_i24` | `RUNNING` | `gpu` | `~3:17 / 12:00:00` | 1 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i24/train` | Active DeepMD training; `lcurve.out` and `slurm-8533968.out` are growing. Latest checked step was `310000`, with checkpoint saved at `2026-05-21 07:10:14 EDT`. |

Accounting for the older ML v4 MD jobs that were originally tracked here shows all ten completed cleanly on 2026-05-19 with `ExitCode=0:0`: `8205299`, `8205302`, `8205305`, `8205314`, `8205319`, `8205321`, `8205322`, `8205326`, `8205331`, `8205332`.

Accounting for `8381309` (`dpgb10x_32g_3125_lin`) now shows `COMPLETED`, elapsed `00:07:11`, `ExitCode=0:0`, on 2026-05-20. It is no longer queued.

Current next checks:

```bash
squeue -j 8375105,8533968
sacct -j 8375105,8533968 --format=JobID,JobName,Partition,State,Elapsed,Start,End,ExitCode
tail -8 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i24/train/lcurve.out
```

Keep this note in `ONGOING/DELLA` because job `8375105` is still pending and `8533968` is running.

## Update - 2026-05-21 07:23 EDT

Submitted two independent-repeat diagnostics for the previously failed/unstable `8gpu_7k_linear_10x` case to test model-quality variation with random initialization/training seed.

| Job ID | Case | State at submission check | Seed fields changed | Working directory |
| --- | --- | --- | --- | --- |
| `8545809` | `8gpu_7k_linear_10x_seed2` | `PENDING` on `gputest`, reason `(None)` | `model.descriptor.seed = 2`, `model.fitting_net.seed = 2`, `training.seed = 2` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/long_steps_10x/8gpu_7k_linear_10x_seed2` |
| `8545808` | `8gpu_7k_linear_10x_seed3` | `PENDING` on `gputest`, reason `(None)` | `model.descriptor.seed = 3`, `model.fitting_net.seed = 3`, `training.seed = 3` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/long_steps_10x/8gpu_7k_linear_10x_seed3` |

Both repeats copy only setup files from `8gpu_7k_linear_10x`; old checkpoints/logs were not copied. Resource shape matches the original: `2` Della GPU nodes, `4` A100 GPUs/node, `8` total ranks, `70000` steps, `linear` scale-by-worker, `01:00:00` walltime.

Check with:

```bash
squeue -j 8545809,8545808
sacct -j 8545809,8545808 --format=JobID,JobName,Partition,State,Elapsed,Start,End,ExitCode
```

After completion, compare final and best-checkpoint `lcurve.out` behavior against the original `8gpu_7k_linear_10x`, then freeze/compress and pseudo-validate if either repeat looks promising.

## Update - 2026-05-21 07:24 EDT

Submitted freeze/compress/pseudo-validation for the completed 32-GPU diagnostic `32gpu_3125_linear_10x`.

| Job ID | Job name | State at submission check | Purpose | Working directory | Output root |
| --- | --- | --- | --- | --- | --- |
| `8545874_0` | `dpgb32g_pseudoval` | `PENDING` on `gputest`, reason `(None)` | Freeze latest checkpoint, compress model, and run `dp test` over the existing pseudo-validation systems. | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_32gpu_20260521` |

Created helper files:

- `EXPERIMENT_MATRIX_32GPU_COMPLETED_FOR_PSEUDOVAL.tsv`
- `run_pseudo_validation_32gpu.sbatch`

The job follows the existing `run_pseudo_validation_10x.sbatch` route, but with one array task for `32gpu_3125_linear_10x`. It freezes the latest checkpoint by default (`model.ckpt-3125`). If this final-checkpoint pseudo-validation is mediocre, a follow-up should freeze/check the better lcurve point near step `2280`.

Check with:

```bash
squeue -j 8545874
sacct -j 8545874 --format=JobID,JobName,Partition,State,Elapsed,Start,End,ExitCode
ls -lh /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_32gpu_20260521/results/32gpu_3125_linear_10x
```

## Update - 2026-05-21 07:34 EDT

Current check while reassessing the training-loss trends:

| Job ID | Job name | State | Runtime / limit | Notes |
| --- | --- | --- | --- | --- |
| `8545808` | `dpgb10x_8g_70k_s3` | `RUNNING` on `gputest` | `~10:30 / 01:00:00` | `8gpu_7k_linear_10x_seed3` has started on `della-l09g[5,7]`; early `lcurve.out` already shows the same late-degradation pattern as the original failed case, but the job is still active. |
| `8545809` | `dpgb10x_8g_70k_s2` | `PENDING` on `gputest` | `0:00 / 01:00:00` | Waiting with reason `Priority`. |
| `8545874_0` | `dpgb32g_pseudoval` | `RUNNING` on `gputest` | `~08:59 / 01:00:00` | 32-GPU freeze/compress/pseudo-validation is active. |

Check commands remain:

```bash
squeue -j 8545808,8545809,8545874
sacct -j 8545808,8545809,8545874 --format=JobID,JobName,Partition,State,Elapsed,ExitCode,Start,End
```

## Update - 2026-05-21 07:54 EDT

32-GPU pseudo-validation job `8545874_0` completed cleanly (`COMPLETED`, `ExitCode=0:0`, elapsed `00:19:48`). Summary written to:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_32gpu_20260521/PSEUDO_VALIDATION_SUMMARY.tsv
```

Aggregate final-checkpoint result for `32gpu_3125_linear_10x` over 2547 pseudo-validation frames:

| Metric | Value |
| --- | ---: |
| Energy RMSE/atom | `0.22219831` eV/atom |
| Force RMSE | `0.90859673` eV/A |
| Virial RMSE/Natoms | `0.21269677` eV/atom |

Interpretation: the 32GPU final checkpoint is not catastrophic, but it is worse than the leading 1/2/4/8/16GPU 10x models by energy, force, and virial. Training lcurve best was around step `2280`, while this validation used the latest checkpoint (`model.ckpt-3125`). Consider validating nearby saved checkpoints (`2000` and/or `3000`) if a better 32GPU checkpoint is desired.

## Update - 2026-05-21 07:59 EDT

Submitted best-training-total checkpoint freeze/compress/pseudo-validation for all completed 10x MLPs plus the 32GPU extra case. Existing final-checkpoint compressed models and pseudo-validation stats are intentionally left untouched. This pass writes to a new validation root.

| Item | Value |
| --- | --- |
| Slurm job array | `8546762` |
| Job name | `dpgb10x_bestval` |
| Partition/QOS | `gputest` / `gpu-test` |
| Array shape | `0-15%4` |
| Cases | 16 = 15 completed 10x models + `32gpu_3125_linear_10x` |
| Matrix | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/EXPERIMENT_MATRIX_10x_BEST_TRAIN_TOTAL_FOR_PSEUDOVAL.tsv` |
| Script | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/run_pseudo_validation_10x_best_train_total.sbatch` |
| Output root | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_10x_best_train_total_20260521` |

Selection rule: for each case, choose the saved checkpoint with the lowest total training RMSE in `lcurve.out`. If the true lcurve minimum is between saved checkpoints, use the best available saved checkpoint and record both values in the matrix. Example: 32GPU true lcurve best is step `2280`, but selected saved checkpoint is `2000`.

Initial queue snapshot: `8546762_0` and `8546762_1` running on `della-i12g1`; `8546762_[2-15]` pending with reason `QOSMaxJobsPerUserLimit`.

Check with:

```bash
squeue -j 8546762
sacct -j 8546762 --format=JobID,JobName,Partition,State,Elapsed,ExitCode,Start,End
```

After completion, summarize with:

```bash
python /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/scripts/summarize_pseudo_validation_any.py --root /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517 --matrix /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/EXPERIMENT_MATRIX_10x_BEST_TRAIN_TOTAL_FOR_PSEUDOVAL.tsv --validation-root /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_10x_best_train_total_20260521 --output /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_10x_best_train_total_20260521/PSEUDO_VALIDATION_SUMMARY.tsv
```

## Update - 2026-05-21 best-checkpoint pseudo-validation completed

Job array `8546762` completed all 16 tasks cleanly (`ExitCode=0:0`). Summary and comparison files:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_10x_best_train_total_20260521/PSEUDO_VALIDATION_SUMMARY.tsv
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_10x_best_train_total_20260521/BEST_VS_FINAL_COMPARISON.tsv
```

Headline results:

- Best-saved-train-total checkpoint improved energy in `5/16`, force in `3/16`, virial in `5/16` cases.
- Biggest change: `4gpu_2500_sqrt_10x` improved from final-checkpoint E/F/V `4.303055/3.767129/5.121701` to best-checkpoint `0.177478/0.757913/0.075724`, making it competitive by energy but still behind the top force group.
- `reuse_1gpu_10k_10x` improved slightly: E `0.183390 -> 0.181580`, F `0.692176 -> 0.686267`.
- `32gpu_3125_linear_10x` improved energy modestly: E `0.222198 -> 0.217466`, but force/virial slightly worsened (`0.908597 -> 0.911729`, `0.212697 -> 0.214329`); still worse than the leading 1/2/4/8/16GPU cases.
- Most final-checkpoint rankings remain unchanged; validation-based checkpoint selection helps specific unstable cases but does not generally make large-GPU schedules superior.

## Update - 2026-05-21 seed-repeat analysis

The two repeat jobs for `8gpu_7k_linear_10x` completed cleanly:

| Job ID | Case | State | Elapsed | ExitCode |
| --- | --- | --- | --- | --- |
| `8545809` | `8gpu_7k_linear_10x_seed2` | `COMPLETED` | `00:34:25` | `0:0` |
| `8545808` | `8gpu_7k_linear_10x_seed3` | `COMPLETED` | `00:34:50` | `0:0` |

Training-curve comparison:

| Case | Exact best step | Exact best total | Best saved step | Best saved total | Final total | Final E/F/V |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `8gpu_7k_linear_10x` | `11330` | `2.42` | `50000` | `361` | `396` | `17.7 / 4.76 / 3.8` |
| `8gpu_7k_linear_10x_seed2` | `7250` | `3.59` | `60000` | `11.5` | `37` | `0.581 / 2.21 / 1.58` |
| `8gpu_7k_linear_10x_seed3` | `4930` | `5.04` | `50000` | `519` | `635` | `28.5 / 3.8 / 8.35` |

Interpretation: there is real seed sensitivity. Seed2 is much less catastrophic than the original at the saved/final checkpoints, while seed3 is worse. But all three runs have their true low-loss region early (`~5k-11k` steps), before the first saved checkpoint at `30000`. The schedule is unstable after the early optimum; seed variation changes how bad the late degradation gets, but does not make this 8GPU/70k schedule robust. Future repeats should use a much shorter `save_freq` or intentional validation/early-stop behavior if this schedule is explored further.

## Update - 2026-05-21 100k none-scaling runs submitted

Submitted three fresh DeePMD training jobs to test whether the apparently stable `none` learning-rate worker scaling remains stable to `100000` steps.

Root:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/none_100k
```

Key settings for all three:

- `training.numb_steps = 100000`
- `learning_rate.decay_steps = 100000`
- `learning_rate.scale_by_worker = none`
- `training.save_freq = 100`

Submitted jobs:

| Job ID | Case | GPUs | Nodes | Time limit | Initial status snapshot |
| --- | --- | ---: | ---: | --- | --- |
| `8560447` | `4gpu_100k_none` | 4 | 1 | `01:00:00` | `RUNNING` on `della-l06g12` at submission check |
| `8560446` | `8gpu_100k_none` | 8 | 2 | `02:00:00` | `PENDING` |
| `8560448` | `16gpu_100k_none` | 16 | 4 | `02:00:00` | `PENDING` |

Check status:

```bash
squeue -j 8560446,8560447,8560448 -o '%i %.18j %.8T %.10M %.9l %.6D %R'
sacct -j 8560446,8560447,8560448 --format=JobID,JobName,Partition,State,Elapsed,ExitCode,Start,End
```

Once complete, inspect `lcurve.out` in each case folder, plot with the ALCHEMY `plots_mod.py` helper, then run freeze/compress/pseudo-validation for final and probably best-total checkpoints.

### Adjustment - 2026-05-21 8GPU none diagnostic walltime shortened

Per diagnostic intent, updated live job `8560446` (`8gpu_100k_none`) from `02:00:00` to `01:00:00` using `scontrol update`, and updated its `run_srun_train_mem.sbatch` accordingly. It is acceptable if this run stops before 100k steps; the goal is to diagnose whether `scale_by_worker = none` avoids the early/late training-loss blowup, even if it reaches only roughly `80k-90k` steps.

### Adjustment - 2026-05-21 8GPU none diagnostic relaunched

The in-place-updated 8GPU job was cancelled and relaunched so the queue entry is fresh with the intended 1-hour request.

| Old job | New job | Case | Change |
| --- | --- | --- | --- |
| `8560446` | `8561215` | `8gpu_100k_none` | cancelled pending job and relaunched with `#SBATCH --time=01:00:00` |

Current check command:

```bash
squeue -j 8560447,8561215,8560448 -o '%i %.18j %.8T %.10M %.9l %.6D %R'
sacct -j 8560446,8561215 --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode
```

## Update - 2026-05-21 8GPU linear warmup diagnostic submitted

Submitted a fresh warmup test for the previously unstable `8gpu_7k_linear_10x` schedule.

Case root:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/warmup_tests/8gpu_7k_linear_10x_warmup5k
```

Job:

| Job ID | Case | GPUs | Nodes | Time limit | Initial status |
| --- | --- | ---: | ---: | --- | --- |
| `8562968` | `8gpu_7k_linear_10x_warmup5k` | 8 | 2 | `01:00:00` | `PENDING` |

Key settings:

- `training.numb_steps = 70000`
- `learning_rate.decay_steps = 70000`
- `learning_rate.scale_by_worker = linear`
- `learning_rate.warmup_steps = 5000`
- `learning_rate.warmup_start_factor = 0.0`
- `training.save_freq = 100`

Purpose: test whether a 5k-step LR warmup prevents the loss blowup seen in `8gpu_7k_linear_10x`, while preserving dense checkpoints around the early low-loss region.

Check status:

```bash
squeue -j 8562968 -o '%i %.18j %.8T %.10M %.9l %.6D %R'
sacct -j 8562968 --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode,Start,End
```

## Update - 2026-05-21 4GPU 100k none completed

Job `8560447` (`4gpu_100k_none`) completed cleanly in `00:50:15` (`ExitCode=0:0`). Key training-curve result:

| Case | Best step | Best total | Best E/F/V | Final total | Final E/F/V | Final/best |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| `4gpu_100k_none` | `89330` | `0.316` | `0.00143 / 0.223 / 0.00725` | `1.10` | `0.0188 / 0.612 / 0.0318` | `3.48` |

Comparison: this essentially matches the best 4GPU 100k linear/reuse training behavior (`reuse_4gpu_10k_10x`: best total `0.303`, final `0.997`) while improving substantially over the shorter `4gpu_2500_none_10x` (`best=0.566`, final `1.89`). It does not show the catastrophic blowups seen in bad 8GPU/linear schedules.

Important checkpoint caveat: although `save_freq=100`, DeePMD keeps only `max_ckpt_keep=5` by default. The true best step `89330` is visible in `lcurve.out` but its checkpoint was deleted; only the final few checkpoints remain. Future dense-checkpoint diagnostics should set `training.max_ckpt_keep` high enough, or use a targeted checkpoint-copy/watch script if preserving every 100-step checkpoint is too expensive.

## Update - 2026-05-21 none-100k final-checkpoint validation submitted

Submitted pseudo-validation for completed none-scaling 100k runs. This validates final retained checkpoints because the lcurve-best checkpoints were not kept under DeePMD's default `max_ckpt_keep=5`.

Validation job array: `8569764`

Matrix:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/EXPERIMENT_MATRIX_NONE_100K_FINAL_FOR_PSEUDOVAL.tsv
```

Validation root:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_none_100k_final_20260521
```

Cases:

| Array index | Case | Checkpoint |
| ---: | --- | --- |
| `0` | `4gpu_100k_none` | `model.ckpt-100000` |
| `1` | `8gpu_100k_none` | `model.ckpt-100000` |

Check status:

```bash
squeue -j 8569764 -o '%i %.22j %.8T %.10M %.9l %.6D %R'
sacct -j 8569764 --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode,Start,End
```

After completion, summarize with:

```bash
python /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/scripts/summarize_pseudo_validation_any.py --root /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517 --matrix /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/EXPERIMENT_MATRIX_NONE_100K_FINAL_FOR_PSEUDOVAL.tsv --validation-root /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_none_100k_final_20260521 --output /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_none_100k_final_20260521/PSEUDO_VALIDATION_SUMMARY.tsv
```

## Update - 2026-05-21 none-100k final-checkpoint validation completed

Validation array `8569764` completed cleanly for both retained final checkpoints.

Aggregate pseudo-validation over 2547 frames:

| Case | Checkpoint | Energy RMSE/atom | Force RMSE | Virial RMSE/atom |
| --- | ---: | ---: | ---: | ---: |
| `4gpu_100k_none` | `100000` | `0.17495026` | `0.58141612` | `0.048947926` |
| `8gpu_100k_none` | `100000` | `17.914549` | `3.3463369` | `0.63368278` |

Summary file:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_none_100k_final_20260521/PSEUDO_VALIDATION_SUMMARY.tsv
```

Interpretation: `4gpu_100k_none` is now the strongest validated case in this benchmark set by force and virial, and narrowly best by energy among the clean 10x-like runs. It beats `reuse_4gpu_10k_10x` on all three aggregate metrics. `8gpu_100k_none` confirms that removing LR worker scaling alone does not cure the high-GPU long-run instability; its final checkpoint is catastrophically bad by energy and force, consistent with its final training lcurve values.

Caveat: these are final-checkpoint validations only. The true lcurve-best checkpoints were not retained because DeePMD defaulted to `max_ckpt_keep=5`. Future dense-checkpoint runs should set `training.max_ckpt_keep` high enough, or explicitly preserve selected checkpoints.

Remaining active/pending related jobs: `8560448` (`16gpu_100k_none`) was cancelled by request after the 4GPU and 8GPU none validations completed; `8562968` (`8gpu_7k_linear_10x_warmup5k`) failed quickly, consistent with `warmup_steps` being PyTorch-only in this DeepMD 3.0.0 path while this workflow uses TensorFlow/Horovod.


## Update - 2026-05-21 19:30 EDT

Submitted DeePMD PyTorch backend repeats for the no-LR-scaling diagnostics.
These runs use `ALCHEMY_env__PT`, `dp --pt train`, `scale_by_worker = none`,
`save_freq = 100`, and 1-hour walltime. The original TF/Horovod runs and
compressed-model outputs are left untouched.

### PT fixed-100k `none` runs

| Job ID | Job name | GPUs | Nodes | Steps | Working directory |
| --- | --- | ---: | ---: | ---: | --- |
| `8573109` | `pt1g_100k_none` | 1 | 1 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/1gpu_100k_none_pt` |
| `8573110` | `pt4g_100k_none` | 4 | 1 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/4gpu_100k_none_pt` |
| `8573111` | `pt8g_100k_none` | 8 | 2 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/8gpu_100k_none_pt` |
| `8573112` | `pt16g_100k_none` | 16 | 4 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/16gpu_100k_none_pt` |

### PT step-scaled `none` runs

These keep total sample exposure roughly matched by setting steps and LR
decay steps to `100000 / n_gpus`.

| Job ID | Job name | GPUs | Nodes | Steps / decay_steps | Working directory |
| --- | --- | ---: | ---: | ---: | --- |
| `8573159` | `ptss1g_100000` | 1 | 1 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_step_scaled_none/1gpu_100k_stepscaled_none_pt` |
| `8573160` | `ptss4g_25000` | 4 | 1 | 25000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_step_scaled_none/4gpu_25k_stepscaled_none_pt` |
| `8573161` | `ptss8g_12500` | 8 | 2 | 12500 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_step_scaled_none/8gpu_12500_stepscaled_none_pt` |
| `8573162` | `ptss16g_6250` | 16 | 4 | 6250 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_step_scaled_none/16gpu_6250_stepscaled_none_pt` |

Current submission snapshot: all eight are `PENDING` on `gputest` with
1-hour limits and pending reason `(None)`.

Check with:

```bash
squeue -j 8573109,8573110,8573111,8573112,8573159,8573160,8573161,8573162
sacct -j 8573109,8573110,8573111,8573112,8573159,8573160,8573161,8573162 --format=JobID,JobName,Partition,State,Elapsed,Start,End,ExitCode
```

After completion, compare PT fixed-100k vs PT step-scaled curves against the
TF/Horovod `none_100k` and `long_steps_10x` controls, then run freeze/compress
and pseudo-validation for the non-pathological checkpoints.


## Update - 2026-05-22 10:14 EDT

The first PT submissions from 2026-05-21 failed immediately before training.
Root cause was the Slurm launcher using `set -u` while the
`ALCHEMY_env__PT` conda activation hook references unset path variables
(`PKG_CONFIG_PATH`, etc.). The failure happened in 1--9 seconds, so no
training curves were produced.

Patched all eight PT launchers to temporarily disable nounset only around
`conda activate ALCHEMY_env__PT`, then re-enable it before training.

Old failed jobs:

```bash
8573109,8573110,8573111,8573112,8573159,8573160,8573161,8573162
```

Fresh relaunches:

| Job ID | Job name | Case family | GPUs | Nodes | Steps |
| --- | --- | --- | ---: | ---: | ---: |
| `8611536` | `pt1g_100k_none` | fixed-100k PT none | 1 | 1 | 100000 |
| `8611537` | `pt4g_100k_none` | fixed-100k PT none | 4 | 1 | 100000 |
| `8611538` | `pt8g_100k_none` | fixed-100k PT none | 8 | 2 | 100000 |
| `8611539` | `pt16g_100k_none` | fixed-100k PT none | 16 | 4 | 100000 |
| `8611540` | `ptss1g_100000` | step-scaled PT none | 1 | 1 | 100000 |
| `8611541` | `ptss4g_25000` | step-scaled PT none | 4 | 1 | 25000 |
| `8611542` | `ptss8g_12500` | step-scaled PT none | 8 | 2 | 12500 |
| `8611543` | `ptss16g_6250` | step-scaled PT none | 16 | 4 | 6250 |

Current snapshot after relaunch: all eight are `PENDING` on `gputest` with
1-hour limits and pending reason `(None)`.

Check with:

```bash
squeue -j 8611536,8611537,8611538,8611539,8611540,8611541,8611542,8611543
sacct -j 8611536,8611537,8611538,8611539,8611540,8611541,8611542,8611543 --format=JobID,JobName,Partition,State,Elapsed,Start,End,ExitCode
```

## Update - 2026-05-22 PT launchers benchmark-aligned and relaunched

The second PT relaunches (`8611536`--`8611543`) also failed, but this time they reached DeePMD startup and the first checkpoint save. Root cause was:

```text
RuntimeError: Parent directory model-compression does not exist.
```

This was a launcher/materialization issue, not evidence about PT training quality. The existing successful PT benchmark records in:

```bash
benchmarks/deepmd/DELLA/deepmd_pt_vs_tf_20260406/
```

use the same `save_ckpt = model-compression/model.ckpt` pattern, and their completed scratch run directories already contain `model-compression/`. The new PT directories lacked that parent directory. The earlier nounset problem was also a deviation from the known-good PT sbatch files, which do not use `set -u` around conda activation.

Fixes applied to all eight live PT launchers:

- create `model-compression/` before `dp --pt train`;
- use `set -eo pipefail` rather than `set -euo pipefail`;
- keep `ALCHEMY_env__PT`, the known module stack, and the benchmark PT DDP launch pattern (`python -m torch.distributed.run ... dp --pt train`).

The partial startup `lcurve.out` files from the failed `86115xx` jobs were renamed to `lcurve.failed_861_startup.out` before relaunch, so the new runs start with clean training curves.

Fresh relaunches:

| Job ID | Job name | Case family | GPUs | Nodes | Steps | Working directory |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `8615991` | `pt1g_100k_none` | fixed-100k PT none | 1 | 1 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/1gpu_100k_none_pt` |
| `8615992` | `pt4g_100k_none` | fixed-100k PT none | 4 | 1 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/4gpu_100k_none_pt` |
| `8615993` | `pt8g_100k_none` | fixed-100k PT none | 8 | 2 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/8gpu_100k_none_pt` |
| `8615994` | `pt16g_100k_none` | fixed-100k PT none | 16 | 4 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/16gpu_100k_none_pt` |
| `8615995` | `ptss1g_100000` | step-scaled PT none | 1 | 1 | 100000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_step_scaled_none/1gpu_100k_stepscaled_none_pt` |
| `8615996` | `ptss4g_25000` | step-scaled PT none | 4 | 1 | 25000 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_step_scaled_none/4gpu_25k_stepscaled_none_pt` |
| `8615998` | `ptss8g_12500` | step-scaled PT none | 8 | 2 | 12500 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_step_scaled_none/8gpu_12500_stepscaled_none_pt` |
| `8615997` | `ptss16g_6250` | step-scaled PT none | 16 | 4 | 6250 | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_step_scaled_none/16gpu_6250_stepscaled_none_pt` |

Current snapshot after relaunch: all eight are `PENDING` on `gputest` with 1-hour limits and pending reason `(None)`.

Immediate monitoring target: once any job starts, confirm it gets past the batch-100 checkpoint and writes `model-compression/model.ckpt-100.pt` or a later `.pt` checkpoint.

Check with:

```bash
squeue -j 8615991,8615992,8615993,8615994,8615995,8615996,8615998,8615997
sacct -j 8615991,8615992,8615993,8615994,8615995,8615996,8615998,8615997 --format=JobID,JobName,Partition,State,Elapsed,Start,End,ExitCode
```

### Final status snapshot for PT none reruns: 2026-05-22 evening

All eight relaunched PT jobs left the queue. Six reached the configured final training step and wrote the final checkpoint; two fixed-100k multi-node jobs were killed by the 1-hour walltime after producing usable partial checkpoints.

| Job ID | Job name | Result | Last lcurve step | Target step | Final checkpoint | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `8615991` | `pt1g_100k_none` | completed | 100000 | 100000 | `model-compression/model.ckpt-100000.pt` | clean `JOB_END`; `sacct` `COMPLETED 0:0` |
| `8615992` | `pt4g_100k_none` | completed | 100000 | 100000 | `model-compression/model.ckpt-100000.pt` | clean `JOB_END`; `sacct` `COMPLETED 0:0` |
| `8615993` | `pt8g_100k_none` | timed out | 85630 | 100000 | `model-compression/model.ckpt-85600.pt` | killed by walltime; not a training crash |
| `8615994` | `pt16g_100k_none` | timed out | 80550 | 100000 | `model-compression/model.ckpt-80500.pt` | killed by walltime; not a training crash |
| `8615995` | `ptss1g_100000` | completed | 100000 | 100000 | `model-compression/model.ckpt-100000.pt` | clean `JOB_END`; `sacct` `COMPLETED 0:0` |
| `8615996` | `ptss4g_25000` | completed | 25000 | 25000 | `model-compression/model.ckpt-25000.pt` | clean final checkpoint; `sacct` `COMPLETED 0:0` |
| `8615998` | `ptss8g_12500` | completed | 12500 | 12500 | `model-compression/model.ckpt-12500.pt` | clean final checkpoint; `sacct` `COMPLETED 0:0` |
| `8615997` | `ptss16g_6250` | completed | 6250 | 6250 | `model-compression/model.ckpt-6250.pt` | clean `JOB_END`; `sacct` `COMPLETED 0:0` |

Conclusion: the fixed-100k 8-GPU and 16-GPU PT runs need a longer walltime or a restart-from-checkpoint continuation if we require the 100k endpoint. The step-scaled PT matrix completed correctly.


## Update - 2026-05-22 23:23 EDT

Submitted freeze/compress/pseudo-validation for the PyTorch none-scaling runs and for the missing TensorFlow 100k best-saved checkpoints.

| Job ID | Job name | Array | Purpose | Output root |
| --- | --- | --- | --- | --- |
| `8626495` | `dpgb_pt_none_val` | `0-15%4` | For each PT none run, freeze/compress/test both the final saved potential and the best saved checkpoint by total training RMSE. Includes fixed-100k PT and step-scaled PT cases. | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_pt_none_final_best_20260522` |
| `8626496` | `dpgb_tf100k_best` | `0-1%4` | For the completed TF `none_100k` runs, freeze/compress/test the best saved checkpoint not already covered by the final-checkpoint pass. | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_tf_none_100k_best_20260522` |

Prepared files:

```bash
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/EXPERIMENT_MATRIX_PT_NONE_FINAL_BEST_FOR_PSEUDOVAL.tsv
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/EXPERIMENT_MATRIX_TF_NONE_100K_BEST_FOR_PSEUDOVAL.tsv
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/run_pseudo_validation_pt_none_final_best.sbatch
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/run_pseudo_validation_tf_none_100k_best.sbatch
```

Notes:

- PT validation uses the native `dp --pt freeze`, `dp --pt compress`, and `dp --pt test` route. A local smoke test confirmed freeze, compress, and `dp --pt test` work on `4gpu_100k_none_pt`.
- TF final-checkpoint validation for `4gpu_100k_none` and `8gpu_100k_none` already exists under `pseudo_validation_none_100k_final_20260521`; this new TF job covers only the best saved checkpoints.
- `16gpu_100k_none` is scaffold-only right now: no `lcurve.out` and no checkpoints, so no TF final/best validation can be run for it yet.
- For TF `8gpu_100k_none`, the true lcurve minimum near step `9580` is not available as a checkpoint; the selected "best" checkpoint is therefore the best saved checkpoint among the available late checkpoints.

Check with:

```bash
squeue -j 8626495,8626496
sacct -j 8626495,8626496 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End
```

## Update - 2026-05-22 23:27 EDT

Current Della queue refresh:

- `8626495` (`dpgb_pt_none_val`) is active for PT final/best validation. Array tasks `_0`, `_1`, and `_2` are running on `gputest`; `_3-15` are pending because of `QOSMaxJobsPerUserLimit`.
- `8626496` (`dpgb_tf100k_best`) is still pending for TF 100k best-checkpoint validation; array tasks `_0` and `_1` are held by the same `gputest` user job limit.
- Validation output roots remain:
  - PT: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_pt_none_final_best_20260522`
  - TF: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_tf_none_100k_best_20260522`
- `8375105` (`dpgb16g_100k_90m`) is still pending on the `gpu` partition for `Priority`.
- A new `v8_i25` MD batch is queued on the `gpu` partition: many one-node jobs named `MD_ZONE_*_177949*`, all currently pending for `Priority` with 4 hour limits. Example checked job: `8622849`, workdir `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i25/md/ZONE_1/90H2_10NH3`, planned start shown as `2026-05-23T03:05:23`.

Quick checks:

```bash
squeue -u ag5805 -o "%.18i %.9P %.36j %.8u %.2t %.10M %.10l %.6D %.40R"
sacct -j 8626495,8626496 --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit,Submit,Start,End -P
```

Next steps when arrays finish: parse PT/TF validation TSVs, compare final vs best checkpoints, and add the new PT curves/results to the existing 10x/none benchmark summaries.

## Update - 2026-05-23 01:05 EDT

PT/TF pseudo-validation arrays completed cleanly:

- `8626495` (`dpgb_pt_none_val`): all array tasks `_0-15` completed with Slurm exit `0:0`.
- `8626496` (`dpgb_tf100k_best`): both array tasks `_0-1` completed with Slurm exit `0:0`.
- No relaunch was needed; Slurm logs did not match `Traceback`, `ERROR`, `FAILED`, `TIMEOUT`, or `CANCELLED` during the post-run check.

Result artifacts:

- Aggregate TSV: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/PT_TF_NONE_VALIDATION_SUMMARY_20260523.tsv`
- Analysis markdown: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/PT_TF_NONE_VALIDATION_ANALYSIS_20260523.md`

Key outcome: PT fixed-100k none-scaling is stable across 1/4/8/16 GPUs; TF 4-GPU best-total is comparable to PT 4-GPU, but TF 8-GPU best-total is unusable and matches the training-loss blowup. Step-scaled PT runs worsen as optimizer steps are reduced with GPU count. Best-training-total checkpoints do not reliably improve validation over final checkpoints.
