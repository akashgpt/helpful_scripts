# NH3/H2 DeePMD Recent TF Seed/Decay Pseudo-Validation

## Status Snapshot

- Date: 2026-05-23
- Cluster: Della
- Slurm partition/QOS: `gputest` / `gpu-test`
- Job ID: `8647865`
- Run type: Slurm array `0-9%4`; only final-checkpoint tasks `0,2,4,6,8` remain active. Best-checkpoint tasks `1,3,5,7,9` were cancelled.
- Validation root: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_recent_tf_decay_seed_20260523`
- Matrix: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/EXPERIMENT_MATRIX_RECENT_TF_DECAY_SEED_FOR_PSEUDOVAL.tsv`
- Final-only matrix for summaries: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/EXPERIMENT_MATRIX_RECENT_TF_DECAY_SEED_FINAL_ONLY_FOR_PSEUDOVAL.tsv`
- Submit script: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/run_pseudo_validation_recent_tf_decay_seed.sbatch`

## Purpose

Freeze, compress, and pseudo-validate the recent completed TF runs:

- `4gpu_100k_none_seed02`
- `4gpu_100k_none_seed03`
- `4gpu_100k_none_seed04`
- `4gpu_100k_none_seed05`
- `8gpu_100k_none_decay10k`

Only final checkpoints are being evaluated. The initially queued best-training-loss checkpoint tasks were cancelled because prior results suggested best-training-loss selection is not useful for non-exploding cases.

## How To Check

```bash
squeue -j 8647865
sacct -j 8647865 --format=JobID,JobName,State,Elapsed,ExitCode
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517
ls pseudo_validation_recent_tf_decay_seed_20260523/results
```

## Next Step Once Done

Summarize pseudo-validation metrics and compare against the earlier
`PT_TF_NONE_VALIDATION_SUMMARY_EXPLICIT_COLUMNS_20260523.tsv` table. In
particular, decide whether:

- 4GPU `none` is robust across seeds by validation metrics.
- `8gpu_100k_none_decay10k` fixes the old 8GPU validation failure, not just the
  training loss blowup.

Do not remove this ONGOING note without explicit user confirmation.

## Completion Snapshot

Updated: 2026-05-24

The final-only pseudo-validation tasks completed: 8647865_0 seed02 final, 8647865_2 seed03 final, 8647865_4 seed04 final, 8647865_6 seed05 final, and 8647865_8 8gpu_100k_none_decay10k final. Best-training-loss tasks 1,3,5,7,9 were cancelled intentionally and should be ignored unless explicitly revisited. Summary: /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/pseudo_validation_recent_tf_decay_seed_20260523/PSEUDO_VALIDATION_SUMMARY_FINAL_ONLY.tsv
