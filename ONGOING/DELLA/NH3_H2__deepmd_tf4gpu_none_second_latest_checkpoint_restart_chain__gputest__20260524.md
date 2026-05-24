# NH3/H2 DeePMD TF 4GPU `none` Second-Latest-Checkpoint Restart Chain

## Status Snapshot

- Date: 2026-05-24 10:56 EDT
- Cluster: Della
- Slurm partition/QOS: `gputest`
- Current job ID: `8695709`
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_second_latest_ckpt`

## Purpose

Test whether restarting from the second-to-last saved checkpoint is more robust than restarting from the latest checkpoint after a walltime interruption. This directly tests the idea of avoiding a potentially awkward latest checkpoint boundary.

## Key Configuration

- Framework: TF/Horovod
- GPUs: 4 A100 on 1 Della node
- Wall time per slice: 15 minutes
- Target step: `100000`
- `learning_rate.scale_by_worker = none`
- `training.numb_steps = 100000`
- `learning_rate.decay_steps = 100000`
- `training.save_freq = 100`
- Restart checkpoint strategy: use second-to-last `all_model_checkpoint_paths` entry if available; fall back to latest if only one checkpoint exists.
- Auto-resubmits the same `run_interrupt_chain.sbatch` until target step or `max_chain_jobs = 12`.

## How To Check

```bash
squeue -j 8695709
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_second_latest_ckpt
tail -n 5 lcurve.out
cat CHAIN_ATTEMPTS.txt
cat CHAIN_HISTORY.tsv
```

If the job resubmits, use the newest `next_job=` value in `CHAIN_HISTORY.tsv`.

## Next Step Once Done

Compare against the latest-checkpoint true restart chain and the uninterrupted 4GPU baseline. If this works better, the issue may be checkpoint-boundary/rollback sensitivity rather than only LR schedule.

Do not remove this ONGOING note without explicit user confirmation.

## Completion / Analysis Snapshot

Updated: 2026-05-24 12:00 EDT

Reached step 100000 after self-resubmits at steps 37210 and 85250. Restarted from model.ckpt-37100 and model.ckpt-85100, i.e. the second-to-last saved checkpoints. Final lcurve row: total=0.909, E=0.0147, F=0.586, V=0.0207, LR=1e-08. This is at least as good as the uninterrupted TF 4GPU baseline by training loss and avoids the original latest-checkpoint restart blowup. This is the strongest direct restart-mechanics result so far.
