# NH3/H2 DeePMD TF 4GPU `none` Health-Gated Restart Chain

## Status Snapshot

- Date: 2026-05-24 10:50 EDT
- Cluster: Della
- Slurm partition/QOS: `gputest`
- Current job ID: `8695475`
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_healthgate`

## Purpose

Test a restart-chain guardrail: keep the same TF 4GPU `none` true-restart setup, but stop automatic resubmission if the late-stage training curve is clearly off-track.

This is not meant to recover a bad model by itself. It is meant to prevent wasting more walltime on a branch that has already drifted away from the known-good trajectory.

## Key Configuration

- Framework: TF/Horovod
- GPUs: 4 A100 on 1 Della node
- Wall time per slice: 15 minutes
- Target step: `100000`
- `learning_rate.scale_by_worker = none`
- `training.numb_steps = 100000`
- `learning_rate.decay_steps = 100000`
- `training.save_freq = 100`
- Health gate begins after step `50000`.
- Gate checks the mean of the last 200 numeric `lcurve.out` rows before resubmitting.
- Gate stops the chain if mean total loss `> 8.0`, mean energy RMSE `> 0.25`, or mean virial RMSE `> 0.15`.
- Gate outputs: `HEALTH_GATE.tsv` and, on failure, `HEALTH_GATE_FAILED`.

## How To Check

```bash
squeue -j 8695475
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_healthgate
tail -n 5 lcurve.out
cat HEALTH_GATE.tsv
cat CHAIN_HISTORY.tsv
```

If the job resubmits, use the newest `next_job=` value in `CHAIN_HISTORY.tsv`. If `HEALTH_GATE_FAILED` exists, the chain intentionally stopped and should be analyzed rather than blindly resumed.

## Next Step Once Done

If the gate stops the run around the region where the previous chain drifted, tighten this into a standard restart-chain safety check. If the run remains healthy, compare its final validation behavior to the uninterrupted 4GPU baseline.

Do not remove this ONGOING note without explicit user confirmation.

Script update: reusable scratch script patched after initial submission so future self-resubmitted slices record a failed gate cleanly under set -e; the first slice does not reach the gate because the gate starts after step 50000.

## Completion / Analysis Snapshot

Updated: 2026-05-24 12:00 EDT

Reached step 100000 after self-resubmits at steps 36750 and 85020. The health gate checked the last-200-row means at step 85020 and passed: mean_total=1.01255, mean_energy=0.017057819, mean_virial=0.0274526. Final lcurve row: total=0.878, E=0.0258, F=0.404, V=0.0209, LR=1e-08. This run stayed healthy; the gate did not need to intervene.
