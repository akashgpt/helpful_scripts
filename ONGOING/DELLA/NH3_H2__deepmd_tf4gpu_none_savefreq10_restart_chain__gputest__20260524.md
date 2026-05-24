# NH3/H2 DeePMD TF 4GPU `none` Save-Frequency Restart Chain

## Status Snapshot

- Date: 2026-05-24 10:50 EDT
- Cluster: Della
- Slurm partition/QOS: `gputest`
- Current job ID: `8695474`
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_savefreq10`

## Purpose

Test whether more frequent checkpointing makes the 15-minute true-restart chain behave more like the uninterrupted `TF 4gpu_100k_none` baseline. This isolates `training.save_freq = 10`; the original chain used `save_freq = 100`.

## Key Configuration

- Framework: TF/Horovod
- GPUs: 4 A100 on 1 Della node
- Wall time per slice: 15 minutes
- Target step: `100000`
- `learning_rate.scale_by_worker = none`
- `training.numb_steps = 100000`
- `learning_rate.decay_steps = 100000`
- `training.save_freq = 10`
- Auto-resubmits the same `run_interrupt_chain.sbatch` until target step or `max_chain_jobs = 12`.
- Restart mode: `dp train --restart <latest_checkpoint>` only after this directory has a checkpoint.

## How To Check

```bash
squeue -j 8695474
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_savefreq10
tail -n 5 lcurve.out
cat CHAIN_ATTEMPTS.txt
cat CHAIN_HISTORY.tsv
```

If the job resubmits, use the newest `next_job=` value in `CHAIN_HISTORY.tsv`.

## Next Step Once Done

Compare the stitched curve against the prior 15-minute chain and the uninterrupted `4gpu_100k_none` baseline. The specific question is whether reducing possible checkpoint rollback from up to 99 steps to up to 9 steps reduces restart drift.

Do not remove this ONGOING note without explicit user confirmation.

## Completion / Analysis Snapshot

Updated: 2026-05-24 12:00 EDT

Reached step 100000 after self-resubmits at steps 30200 and 65200. Final lcurve row: total=212, E=4.81, F=2.88, V=8.87, LR=1e-08. This run is much worse than the uninterrupted baseline and also worse than the original failed latest-checkpoint restart chain. The divergence began before the first resubmit: first clear mismatch from baseline at step ~840, so this is not evidence that more frequent checkpoints repair restart behavior. Treat `save_freq=10` in this TF/Horovod setup as actively suspect unless reproduced/explained.
