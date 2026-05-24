# NH3/H2 DeePMD PT 4GPU none 10min healthgate + second-latest restart chain

- Cluster: DELLA
- Partition: `gputest`
- Submitted: 2026-05-24
- Current job ID: `8706983`
- Status: restarted cleanly as `v2` after the first attempt exposed a selector bug. Old jobs `8705530` and `8706120` were cancelled because the script selected `model.ckpt-9800.pt` after reaching step `19980`, contaminating the learning curve with a large rollback. The live script now selects checkpoints by parsed numeric step and refuses restarts that roll back more than `ALCHEMY_RESTART_MAX_ROLLBACK_STEPS` (default `2000`). New clean job `8706354` timed out at step 20320 and self-resubmitted to `8706983`; this scratch diagnostic still uses the standalone self-resubmitting script, while the ALCHEMY production templates now use Level-2-owned resubmission.
- Latest monitor snapshot: 2026-05-24 14:06 EDT: first v2 slice `8706354` hit the time limit at step `20320` and resubmitted cleanly to current job `8706983`. Latest chain row: `CHAIN_RESUBMITTED reason=pre_walltime_signal current_job=8706354 next_job=8706983 step=20320 target=100000 attempts=1`. New slice `8706983` is running with `TIME_LIMIT=10:00`.
- Job name: `pt4g_hgate2v2_10m`
- Working directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2`
- Submit script: `run_interrupt_chain.sbatch`

## Purpose

Repeat the PT 4GPU `none` 10-minute chained restart test, but with the same two guardrails used for the better-behaved TF test:

- select the second-latest checkpoint when restarting, implemented by scanning `model-compression/model.ckpt-*.pt` and sorting by parsed numeric checkpoint step;
- fail loudly if the chosen restart checkpoint is unexpectedly far behind the last recorded `lcurve.out` step;
- run the rolling health gate before resubmitting when below the target step.

This specifically tests whether the previous PT LR jump/restart pathology is avoided by using a slightly older checkpoint instead of the latest checkpoint.

## Key Configuration

- Framework: DeePMD PT via `ALCHEMY_env__PT`
- GPUs: 4 A100 on one Della node
- Target: `100000` training steps
- Walltime per slice: `00:10:00`
- Maximum chain jobs: `12`
- LR scaling: `scale_by_worker = none`
- Source input copied from the completed PT 4GPU `none` restart-chain test.

## How To Check

```bash
squeue -j 8706983
sacct -j 8706983 --format=JobID,JobName%30,State,ExitCode,Elapsed
tail -n 5 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2/lcurve.out
cat /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2/CHAIN_HISTORY.tsv
cat /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2/HEALTH_GATE.tsv
```

## Next Step

When the chain finishes or stops, compare LR around restart boundaries against the previous PT latest-checkpoint chain. Update the checkpoint-restart plot and manifest in:

`/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/training_loss_plots_10x_20260521/CHECKPOINT_RESTART_TESTS_20260524`

Do not remove this note without explicit user confirmation.
