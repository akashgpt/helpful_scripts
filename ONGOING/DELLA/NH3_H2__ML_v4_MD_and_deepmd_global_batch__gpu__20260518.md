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
| `8381309` | DeepMD global batch | `gputest` | `PD` | `01:00:00` | 8 | `Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/long_steps_10x/8node_32gpu_3125_linear_10x` |

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
- `8381309`: `long_steps_10x/8node_32gpu_3125_linear_10x`, job name `dpgb10x_32g_3125_lin`, `gputest`, 8 nodes, 32 A100 GPUs, 60 minutes.

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
