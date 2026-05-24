# NH3/H2 DeePMD PT 4GPU `none` True-Restart Chain

## Status Snapshot

- Date: 2026-05-24 11:52 EDT
- Cluster: Della
- Slurm partition/QOS: `gputest`
- Current job ID: `8698985`
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_15min_chain`

## Purpose

Test DeePMD PyTorch restart behavior under the same 15-minute interrupted-job continuation pattern used for the TF/Horovod 4GPU `none` tests.

This keeps the original 100k-step schedule and uses `dp --pt train ... --restart <checkpoint>` only after a checkpoint has been created in this directory.

## Key Configuration

- Source run: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/4gpu_100k_none_pt`
- Framework: DeePMD PyTorch
- GPUs: 4 A100 on 1 Della node
- Wall time per slice: 15 minutes
- Target step: `100000`
- `learning_rate.scale_by_worker = none`
- `training.numb_steps = 100000`
- `learning_rate.decay_steps = 100000`
- `training.save_freq = 100`
- Auto-resubmits the same `run_interrupt_chain.sbatch` until target step or `max_chain_jobs = 12`.
- Restart checkpoint selection reads the root `checkpoint` file written by PT DeePMD and passes that path to `dp --pt train --restart`.

## How To Check

```bash
squeue -j 8698985
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_15min_chain
tail -n 5 lcurve.out
cat checkpoint
cat CHAIN_ATTEMPTS.txt
cat CHAIN_HISTORY.tsv
```

If the job resubmits, use the newest `next_job=` value in `CHAIN_HISTORY.tsv`.

## Next Step Once Done

Compare the stitched PT restart-chain curve against:

- uninterrupted `pt4g_100k_none`
- TF true 15-minute restart chain
- TF save-frequency and second-latest-checkpoint restart tests

The core question is whether PT restart continuation stays on a healthier trajectory than TF under the same walltime-sliced pattern.

Do not remove this ONGOING note without explicit user confirmation.

## Superseded by 10-minute chain

Updated: 2026-05-24 11:54 EDT

Job `8698985` was cancelled shortly after launch because the requested slice length changed from 15 minutes to 10 minutes. A `.resubmitted_8698985` marker was created before cancellation to prevent the 15-minute job from self-resubmitting. Use the new 10-minute run instead: job `8699042`, directory `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain`.
