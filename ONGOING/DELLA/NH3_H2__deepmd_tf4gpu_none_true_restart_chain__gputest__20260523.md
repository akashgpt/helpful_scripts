# NH3/H2 DeePMD TF 4GPU `none` True-Restart Chain

## Status Snapshot

- Date: 2026-05-23
- Cluster: Della
- Slurm partition/QOS: `gputest`
- Current job ID: `8647781`
- Previous job ID: `8643969` was cancelled while still pending so the first slice would use the updated self-resubmitting script. Job `8644004` then failed immediately because Slurm used a spool copy for `BASH_SOURCE[0]`; relaunched fixed canonical-path script as `8647781`.
- Current scheduler state after relaunch: submitted job `8647781` after fixing canonical script path
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260523/tf4g_100k_none_15min_chain`

## Purpose

Test DeePMD `--restart` only in the intended "interrupted job continuation"
mode: each slice keeps the same original 100k-step training schedule, and the
next slice resumes from the latest checkpoint if the prior 15-minute allocation
ended before reaching 100k steps.

This is different from the earlier `+100k` continuation tests, which changed
the total target step count and therefore changed the effective learning-rate
and loss schedule.

## Key Configuration

- Source run copied from: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/none_100k/4gpu_100k_none`
- Framework: TF/Horovod
- GPUs: 4 A100 on 1 node
- Wall time per slice: 15 minutes
- Target step: `100000`
- `learning_rate.scale_by_worker = none`
- `training.numb_steps = 100000`
- `learning_rate.decay_steps = 100000`
- `training.save_freq = 100`
- Restart behavior:
  - First slice starts fresh if no checkpoint exists.
  - Later slices run `dp train ... --restart <latest_checkpoint>` when `model-compression/checkpoint` exists.
  - The JSON schedule is not lengthened between slices.
- Auto-resubmission:
  - Slurm sends `USR1` 90 seconds before walltime.
  - The script uses the canonical scratch path to `run_interrupt_chain.sbatch` and resubmits that same file if `lcurve.out` has not reached step `100000`.
  - An `EXIT` trap is a fallback resubmitter.
  - Resubmission is capped by `max_chain_jobs = 12`.

## Files

- Submit script: `run_interrupt_chain.sbatch`
- Input JSON: `myinput.json`
- Run note: `RUN_INFO.md`
- Chain attempt counter: `CHAIN_ATTEMPTS.txt` once started
- Chain resubmission history: `CHAIN_HISTORY.tsv` once a resubmit happens
- Per-slice GPU monitor logs: `gpu_mem_util_<jobid>.csv`

## How To Check

```bash
squeue -j 8647781
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260523/tf4g_100k_none_15min_chain
tail -n 5 lcurve.out
cat CHAIN_ATTEMPTS.txt
cat CHAIN_HISTORY.tsv
ls -lh slurm-*.out
```

If the current job has already resubmitted itself, get the latest child job from
`CHAIN_HISTORY.tsv` and check that job with `squeue -j <jobid>`.

## How To Resume Manually

From the run directory:

```bash
sbatch run_interrupt_chain.sbatch
```

The script will detect `model-compression/checkpoint` and use DeePMD
`--restart` automatically if a checkpoint exists.

## Next Step Once Done

When `lcurve.out` reaches step `100000`, compare this true interrupted-chain
curve against the uninterrupted `TF 4gpu_100k_none` curve. The question is
whether walltime slicing plus `--restart` preserves the same loss trajectory
when the schedule is not altered.

Do not remove this ONGOING note without explicit user confirmation.

## Completion / Analysis Snapshot

Updated: 2026-05-24

The true restart chain reached the 100k target after two automatic resubmits: 8647781 timed out/resubmitted at step 37790, 8648681 timed out/resubmitted at step 86360, and 8650934 completed at step 100000. Final lcurve row: 100000  2.58e+01  1.10e+00  5.78e-01  4.21e-01  1.0e-08. Mechanically, the same script did resume and finish the interrupted schedule, but scientifically the final total/energy/virial losses are much worse than uninterrupted 4gpu_100k_none while force is similar.
