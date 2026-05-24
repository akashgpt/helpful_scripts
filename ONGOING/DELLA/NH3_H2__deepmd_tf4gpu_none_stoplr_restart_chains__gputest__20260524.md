# NH3/H2 DeePMD TF 4GPU `none` Higher Final-LR Restart Chains

## Status Snapshot

- Date: 2026-05-24 10:56 EDT
- Cluster: Della
- Slurm partition/QOS: `gputest`
- Current job IDs:
  - `8695711`: `stop_lr = 1e-7`
  - `8695710`: `stop_lr = 1e-6`
- Run directories:
  - `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_stoplr1e-7`
  - `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_stoplr1e-6`

## Purpose

Test whether a less-stranded late learning-rate floor can recover better after restart-chain drift. The prior true restart chain reached LR around `4.8e-08` near 86k steps and then ended with high energy/virial losses. These variants keep the same 100k schedule but add a higher DeePMD `stop_lr`.

## Key Configuration

- Framework: TF/Horovod
- GPUs: 4 A100 on 1 Della node
- Wall time per slice: 15 minutes
- Target step: `100000`
- `learning_rate.scale_by_worker = none`
- `training.numb_steps = 100000`
- `learning_rate.decay_steps = 100000`
- `training.save_freq = 100`
- Restart checkpoint strategy: latest checkpoint, as in the original true restart chain.
- Variant 1: `learning_rate.stop_lr = 1e-7`
- Variant 2: `learning_rate.stop_lr = 1e-6`

## How To Check

```bash
squeue -j 8695711,8695710
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_stoplr1e-7
tail -n 5 lcurve.out
cat CHAIN_HISTORY.tsv
cd ../tf4g_100k_none_15min_chain_stoplr1e-6
tail -n 5 lcurve.out
cat CHAIN_HISTORY.tsv
```

If either job resubmits, use the newest `next_job=` value in that run's `CHAIN_HISTORY.tsv`.

## Next Step Once Done

Compare stitched training curves and final validation against:
- uninterrupted `4gpu_100k_none`
- original true restart chain
- save-frequency and health-gate follow-up chains

If higher `stop_lr` reduces late E/V blowup, follow with validation and possibly a production-style schedule sweep.

Do not remove this ONGOING note without explicit user confirmation.

## Cancellation Update

Updated: 2026-05-24 11:00 EDT

Cancelled pending jobs `8695711` (`stop_lr = 1e-7`) and `8695710` (`stop_lr = 1e-6`) before they started. Reason: user correctly noted the main loss worsening begins at the first restart when LR is still ~1e-5, so higher late-stage LR is at best a recovery diagnostic and not a primary restart-correctness test. Keep focus on direct restart-mechanics tests: save_freq=10, second-latest checkpoint, and health gate.
