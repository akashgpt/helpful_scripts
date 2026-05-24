# NH3/H2 DeePMD PT 4GPU `none` 10-Minute True-Restart Chain

## Status Snapshot

- Date: 2026-05-24 13:04 EDT
- Cluster: Della
- Slurm partition/QOS: `gputest`
- Last job ID: `8702949`
- Supersedes: cancelled 15-minute PT chain job `8698985`
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain`
- Status: completed cleanly. Slices `8699042`, `8699821`, `8700915`, and `8701781` finished/resubmitted; final slice `8702949` reached step `100000` and exited `0:0`. Final lcurve row: train total `9.98`, train E `0.0289`, train F `0.526`, train V `0.0234`, LR `3.6e-04`. Slurm log ended with `CHAIN_DONE step=100000 target=100000 reason=exit_code_0`. No `pt4g_none10m` job remains in `squeue`. Keep this note until user confirms cleanup.

## Purpose

Test DeePMD PyTorch restart behavior with 10-minute interrupted-job slices for the 4GPU `none` case. This is the PT analogue of the TF true-restart-chain diagnostics, with more frequent forced restarts.

## Key Configuration

- Source run: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/pt_none_100k/4gpu_100k_none_pt`
- Framework: DeePMD PyTorch
- GPUs: 4 A100 on 1 Della node
- Wall time per slice: 10 minutes
- Target step: `100000`
- `learning_rate.scale_by_worker = none`
- `training.numb_steps = 100000`
- `learning_rate.decay_steps = 100000`
- `training.save_freq = 100`
- Auto-resubmits the same `run_interrupt_chain.sbatch` until target step or `max_chain_jobs = 12`.
- Restart checkpoint selection reads the root `checkpoint` file written by PT DeePMD and passes that path to `dp --pt train --restart`.

## How To Check

```bash
squeue -j 8702949
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain
tail -n 5 lcurve.out
cat checkpoint
cat CHAIN_ATTEMPTS.txt
cat CHAIN_HISTORY.tsv
```

If the job resubmits, use the newest `next_job=` value in `CHAIN_HISTORY.tsv`.

## Next Step Once Done

Compare the stitched PT 10-minute restart-chain curve against uninterrupted `pt4g_100k_none`, the cancelled/superseded PT 15-minute attempt, and the TF restart-chain diagnostics.

Do not remove this ONGOING note without explicit user confirmation.
