# He_MgSiO3 fit_deep2x H200 Training

Date: 2026-05-28
Cluster: NCSA Delta
Partition: `gpuH200x8`
Account: `bguf-delta-gpu`
Working directory: `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench`
Variant directory: `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_se_e2_a_TF_fit_deep2x_H200_1M_val`
Job ID: `18588800`

## Purpose

Run one TensorFlow DeePMD `se_e2_a` fitting-depth 2x (`fit_deep2x`) training on one Delta H200 GPU for 1,000,000 steps, with the same MgSiO3 validation-data pattern used by the H200 DPA2 train+validation runs.

## Config Summary

- Shared JSON: `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/shared/train_se_e2_a_fit_deep2x_H200_1M_val.json`
- Submit script: `sub.sh` in the variant directory.
- Base config: copied/adapted from `shared/train_se_e2_a_fit_deep2x.json`.
- Validation block: copied from `shared/train_dpa2_current_auto_1M_H200.json`.
- DeePMD backend: TensorFlow, `dp --tf train`.
- Environment: `ALCHEMY_env`.
- Training systems: `255` MgSiOH__R2SCAN TRAIN systems.
- Validation systems: `61` MgSiO3 `sim_data` systems.
- Steps: `training.numb_steps = 1000000`.
- Validation settings: `batch_size = 1`, `numb_btch = 4`, `auto_prob = prob_sys_size`.
- Slurm resources: 1 node, 1 H200 GPU, 2 CPUs, 64 GB total RAM, 48 h walltime.

The validation set is useful for in-training generalization and overfitting monitoring. It is not a blind scientific held-out test because it matches the MgSiO3 validation-data pattern already used in the H200 DPA2 train+validation runs.

## Submission

Submitted from inside the variant directory:

```bash
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_se_e2_a_TF_fit_deep2x_H200_1M_val
sbatch sub.sh
```

Slurm returned:

```text
Submitted batch job 18588800
```

## Resume Behavior

`sub.sh` copies the shared JSON to `input.json` at launch. If `model.ckpt-*.index` exists in the variant directory, it resumes from the latest TensorFlow checkpoint prefix with:

```bash
dp --tf train --restart "$LAST_CKPT" input.json
```

Otherwise it starts fresh with:

```bash
dp --tf train input.json
```

## How To Check

```bash
squeue -j 18588800 -o "%i %T %P %M %l %j %R"
sacct -j 18588800 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Partition -P
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_se_e2_a_TF_fit_deep2x_H200_1M_val
tail -25 log.train
tail -5 lcurve.out
```

## Status Snapshot

2026-05-28 18:59 CDT:

```text
JOBID STATE PARTITION TIME TIME_LIMIT NAME NODELIST(REASON)
18588800 PENDING gpuH200x8 0:00 2-00:00:00 se_fit2x_H200_1M (None)
```

```text
JobID|JobName|State|ExitCode|Elapsed|Start|End|NodeList|Partition
18588800|se_fit2x_H200_1M|PENDING|0:0|00:00:00|Unknown|Unknown|None assigned|gpuH200x8
```

2026-05-28 19:00 CDT: job started on `gpue05`; `input.json`, `log.train`, and `slurm-18588800.out` exist in the variant directory.

```text
JOBID STATE PARTITION TIME TIME_LIMIT NAME NODELIST(REASON)
18588800 RUNNING gpuH200x8 0:34 2-00:00:00 se_fit2x_H200_1M gpue05
```

```text
JobID|JobName|State|ExitCode|Elapsed|Start|End|NodeList|Partition
18588800|se_fit2x_H200_1M|RUNNING|0:0|00:00:40|2026-05-28T18:59:56|Unknown|gpue05|gpuH200x8
18588800.batch|batch|RUNNING|0:0|00:00:40|2026-05-28T18:59:56|Unknown|gpue05|
18588800.extern|extern|RUNNING|0:0|00:00:40|2026-05-28T18:59:56|Unknown|gpue05|
```

## Next Step

When the job starts, confirm that `input.json`, `log.train`, and `lcurve.out` appear in the variant directory. When complete, use the final TensorFlow checkpoint for the same compressed/frozen validation workflow used for the `se_e2_a` model family.
