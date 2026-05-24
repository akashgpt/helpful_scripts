# NH3/H2 DeePMD TF 4GPU `none` 10-Minute Healthgate + Second-Latest Restart Chain

## Status Snapshot

- Date: 2026-05-24 13:11 EDT
- Cluster: Della
- Slurm partition/QOS: `gputest`
- Last job ID: `8702890`
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_10min_chain_healthgate_second_latest_ckpt`
- Status: completed target at step `100000` in final slice `8702890`. Slurm allocation state is `TIMEOUT`, but batch/extern completed and the run log shows `CHAIN_DONE step=100000 target=100000 reason=exit_code_0` after `CHAIN_SIGNAL_USR1`; no `tf4g_hgate_2nd_10m` job remains in `squeue`. Final lcurve row: train total `0.898`, train E `0.0148`, train F `0.489`, train V `0.0281`, LR `1.0e-08`. Keep this note until user confirms cleanup.

## Purpose

Test a combined robust restart strategy for the TF/Horovod 4GPU `none` case:

- use 10-minute interrupted-job slices,
- restart from the second-latest saved checkpoint when available,
- run the recent-loss health gate before resubmitting,
- continue the same original 100k-step schedule using `dp train --restart`.

## Key Configuration

- Source settings copied from the successful TF `4gpu_100k_none` / healthgate restart tests.
- Framework: DeePMD TensorFlow/Horovod via Apptainer.
- GPUs: 4 A100 on 1 Della node.
- Wall time per slice: 10 minutes.
- Target step: `100000`.
- `learning_rate.scale_by_worker = none`.
- `training.numb_steps = 100000`.
- `learning_rate.decay_steps = 100000`.
- `training.save_freq = 100`.
- Restart checkpoint selection: second-latest `all_model_checkpoint_paths` entry when available; latest if only one exists.
- Health gate: after step 50000, stop resubmitting if recent mean total/energy/virial losses exceed the configured thresholds.

## How To Check

```bash
sacct -j 8702890 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End -P
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_10min_chain_healthgate_second_latest_ckpt
tail -n 5 lcurve.out
cat CHAIN_ATTEMPTS.txt
cat CHAIN_HISTORY.tsv
cat HEALTH_GATE.tsv
```

If the job resubmits, use the newest `next_job=` value in `CHAIN_HISTORY.tsv`.

## Next Step Once Done

Compare the stitched curve against:

- uninterrupted `4gpu_100k_none`,
- TF latest-checkpoint 15-minute restart chain,
- TF second-latest-checkpoint 15-minute restart chain,
- TF healthgate 15-minute restart chain,
- PT 10-minute restart chain.

Do not remove this ONGOING note without explicit user confirmation.
