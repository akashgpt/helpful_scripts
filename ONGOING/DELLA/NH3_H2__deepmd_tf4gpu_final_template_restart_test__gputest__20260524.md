# NH3/H2 DeePMD TF 4GPU Final Template Restart Test

## Status Snapshot

- Date: 2026-05-24 14:29 EDT
- Cluster: Della
- Slurm partition/QOS: `gputest`
- Last job ID: `8707294`
- Run directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/template_restart_tests_20260524/tf4g_100k_none_final_template_15min_hvd`
- Status: completed target step `100000` in final slice `8707294`; no `tf_tpl_hvd15m` job remains in `squeue`. Slices `8702974`, `8704113`, `8705739`, and `8706739` hit their time limits and resubmitted cleanly before `8707294` finished the remaining work. The final slice printed `finished training`, ran freeze/compress, and produced `model-compression/pv.pb` plus `model-compression/pv_comp.pb`. Final lcurve row: train total `0.877`, train E `0.00155`, train F `0.484`, train V `0.0307`, LR `1.0e-08`; last-10%-of-rows median values were total `0.903`, E `0.0148`, F `0.507`, V `0.0233`.

## Purpose

Test the final ALCHEMY Della/Tiger TF restart template created at:

`/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev/reference_input_files/submission_scripts/DELLA__TIGER/train_1h.apptr.restart.TF.sh`

This test uses the known 4GPU `none` input, with the copied script walltime changed to 15 minutes. The first attempt `8702888` was cancelled because the old template defaulted to `deepmd-kit_latest.sif`, which lacked Horovod and fell back to serial execution; this relaunch uses the corrected `deepmd-kit_3.0.0_cuda126.sif` default. Startup log confirms DeePMD ranks `0-3`, so the TF/Horovod path is active.

## Key Configuration

- Framework: DeePMD TensorFlow/Horovod via Apptainer.
- GPUs: 4 A100 on 1 Della node.
- Wall time per slice: 15 minutes.
- Target step: `100000`.
- `learning_rate.scale_by_worker = none` from the copied `myinput.json`.
- Restart checkpoint strategy: second-latest TF checkpoint when available.
- Health gate: enabled after step `50000`.
- Script under test: `train_1h.apptr.restart.TF.sh`.

## How To Check

```bash
sacct -j 8707294 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End -P
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/template_restart_tests_20260524/tf4g_100k_none_final_template_15min_hvd
tail -n 5 lcurve.out
cat CHAIN_ATTEMPTS.txt
cat CHAIN_HISTORY.tsv
cat HEALTH_GATE.tsv
```

If the job resubmits, use the newest `next_job=` value in `CHAIN_HISTORY.tsv`.

## Next Step Once Done

The copied final template ran, resubmitted, reached 100k, and finalized correctly. Compare against the other restart-chain diagnostics and keep the template behavior as the canonical reference for Della/Tiger TF restart/finalization.

Do not remove this ONGOING note without explicit user confirmation.
