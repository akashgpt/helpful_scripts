# NH3/H2 DeePMD PT 4GPU none 10min healthgate + second-latest restart chain

- Cluster: DELLA
- Partition: `gputest`
- Submitted: 2026-05-24
- Last job ID: `8708172`
- Status: completed target step `100000` in final slice `8708172`; no `pt4g_hgate2v2_10m` job remains in `squeue`. The clean v2 chain ran through jobs `8706354 -> 8706983 -> 8707357 -> 8707662 -> 8708172`. It reached target and then froze/compressed the PT model to `model-compression/pv.pth` and `model-compression/pv_comp.pth`. Final lcurve row: train total `7.51`, train E `0.0274`, train F `0.524`, train V `0.0374`, LR `2.0e-04`; last-10%-of-rows median values were total `7.15`, E `0.03525`, F `0.493`, V `0.0347`.
- Latest monitor snapshot: 2026-05-24 15:04 EDT: final chain row in `slurm-8708172.out` shows `CHAIN_DONE step=100000 target=100000 reason=exit_code_0` after freeze/compress completed. Earlier old jobs `8705530` and `8706120` were cancelled because the first script selected `model.ckpt-9800.pt` after reaching step `19980`, contaminating the learning curve with a large rollback; use the `v2` directory for analysis.
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
sacct -j 8708172 --format=JobID,JobName%30,State,ExitCode,Elapsed
tail -n 5 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2/lcurve.out
cat /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2/CHAIN_HISTORY.tsv
cat /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2/HEALTH_GATE.tsv
```

## Next Step

Compare LR around restart boundaries against the previous PT latest-checkpoint chain. Update or regenerate the checkpoint-restart plot and manifest in:

`/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/training_loss_plots_10x_20260521/CHECKPOINT_RESTART_TESTS_20260524`

Do not remove this note without explicit user confirmation.
