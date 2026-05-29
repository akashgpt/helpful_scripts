# He_MgSiO3 DPA2 H200 Stability Suite

Date: 2026-05-27
Cluster: NCSA Delta
Partition: `gpuH200x8`
Working directory: `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench`
Manifest: `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/DPA2_H200_STABILITY_SUITE_20260527.tsv`

## Purpose

Test whether the DPA-2 learning instability seen near 20k steps improves when the current auto-neighbor model is run with explicit three-body neighbors, train+validation tracking, and one stabilizer at a time.

All runs are 200k-step PyTorch DeePMD trainings on one H200 GPU with the same training and validation datasets used by the current H200 train+val config.

## Submitted Jobs

| Case | Job ID | Variant directory | Main change |
| --- | ---: | --- | --- |
| `dpa2_cur3b_clip_H200_200k` | 18541984 | `variant_train_dpa2_PT_current_auto_3body_sel120_clip1_H200_200k` | `current_auto`, `use_three_body=true`, `three_body_sel=120`, stronger clipping with `gradient_max_norm=1.0` |
| `dpa2_docmed_H200_200k` | 18541985 | `variant_train_dpa2_PT_doc_medium_auto_currentloss_H200_200k` | doc-medium comparator with validation data |
| `dpa2_cur3b_lowlr_H200_200k` | 18541986 | `variant_train_dpa2_PT_current_auto_3body_sel120_lowlr_H200_200k` | `current_auto`, `use_three_body=true`, `three_body_sel=120`, lower starting LR `3e-4` |
| `dpa2_cur3b_warm_H200_200k` | 18541987 | `variant_train_dpa2_PT_current_auto_3body_sel120_warmup_H200_200k` | `current_auto`, `use_three_body=true`, `three_body_sel=120`, 20k-step LR warmup |

## Replacement Jobs

The first submission was launched from `/projects/bguf/akashgpt/run_scripts/helpful_scripts`, so `SLURM_SUBMIT_DIR` pointed to the wrong directory and the scripts failed before DeePMD started. The jobs were resubmitted on 2026-05-28 from inside each variant directory.

| Case | Old Job ID | Replacement Job ID | Variant directory |
| --- | ---: | ---: | --- |
| `dpa2_cur3b_clip_H200_200k` | 18541984 | 18560463 | `variant_train_dpa2_PT_current_auto_3body_sel120_clip1_H200_200k` |
| `dpa2_docmed_H200_200k` | 18541985 | 18560462 | `variant_train_dpa2_PT_doc_medium_auto_currentloss_H200_200k` |
| `dpa2_cur3b_lowlr_H200_200k` | 18541986 | 18560460 | `variant_train_dpa2_PT_current_auto_3body_sel120_lowlr_H200_200k` |
| `dpa2_cur3b_warm_H200_200k` | 18541987 | 18560461 | `variant_train_dpa2_PT_current_auto_3body_sel120_warmup_H200_200k` |

## Generated Inputs

Shared JSON files:

- `shared/train_dpa2_PT_current_auto_3body_sel120_clip1_H200_200k.json`
- `shared/train_dpa2_PT_doc_medium_auto_currentloss_H200_200k.json`
- `shared/train_dpa2_PT_current_auto_3body_sel120_lowlr_H200_200k.json`
- `shared/train_dpa2_PT_current_auto_3body_sel120_warmup_H200_200k.json`

Validation notes:

- Current-auto three-body runs use `use_three_body=true` and `three_body_sel=120`.
- The doc-medium comparator keeps its `three_body_sel=auto:1.1` setting.
- All four inputs include `validation_data`.
- The validation split is useful for stability tracking, but it is not a separate blind scientific test set.

## How To Check

```bash
squeue -j 18541984,18541985,18541986,18541987 -o "%i %T %P %M %l %j %R"
sacct -j 18541984,18541985,18541986,18541987 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Partition -P
squeue -j 18560460,18560461,18560462,18560463 -o "%i %T %P %M %l %j %R"
sacct -j 18560460,18560461,18560462,18560463 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Partition -P
```

Per-run logs live in each variant directory as `slurm-<jobid>.out` once Slurm starts the job.

## Resume / Next Steps

1. If the warmup job fails quickly, check whether DeePMD accepts `warmup_steps` and `warmup_start_factor` in the `learning_rate` block for this version.
2. When jobs finish, plot `lcurve.out` with the updated train+validation plotting path and compare train/val total, energy, force, virial RMSE.
3. Compare behavior around 20k steps first, then last-1000-step RMSE summaries.
4. If a case looks stable and scientifically competitive, run the full compressed/frozen validation sweep before promoting it.

## Status Snapshot

2026-05-27 19:22 CDT: Four jobs submitted to `gpuH200x8`.

2026-05-27 19:22 CDT Slurm snapshot:

| Job ID | State | Reason |
| ---: | --- | --- |
| 18541984 | PENDING | Resources |
| 18541985 | PENDING | Priority |
| 18541986 | PENDING | Priority |
| 18541987 | PENDING | Priority |

2026-05-28 12:27 CDT: First job set found failed:

| Job ID | State | Exit code | Root cause |
| ---: | --- | --- | --- |
| 18541984 | FAILED | 1:0 | Submitted from wrong directory; `../shared` not found |
| 18541985 | FAILED | 1:0 | Submitted from wrong directory; `../shared` not found |
| 18541986 | FAILED | 1:0 | Submitted from wrong directory; `../shared` not found |
| 18541987 | FAILED | 1:0 | Submitted from wrong directory; `../shared` not found |

2026-05-28 12:28 CDT: Resubmitted from variant directories as jobs 18560460-18560463.

2026-05-28 12:28 CDT Slurm snapshot:

| Job ID | Case | State | Reason |
| ---: | --- | --- | --- |
| 18560460 | `dpa2_cur3b_lowlr_H200_200k` | PENDING | Priority |
| 18560461 | `dpa2_cur3b_warm_H200_200k` | PENDING | Priority |
| 18560462 | `dpa2_docmed_H200_200k` | PENDING | Priority |
| 18560463 | `dpa2_cur3b_clip_H200_200k` | PENDING | Priority |

2026-05-28 18:44 CDT: Replacement jobs completed successfully.

| Job ID | Case | State | Exit code | Elapsed | Node |
| ---: | --- | --- | --- | --- | --- |
| 18560460 | `dpa2_cur3b_lowlr_H200_200k` | COMPLETED | 0:0 | 04:43:17 | gpue05 |
| 18560461 | `dpa2_cur3b_warm_H200_200k` | COMPLETED | 0:0 | 04:42:32 | gpue05 |
| 18560462 | `dpa2_docmed_H200_200k` | COMPLETED | 0:0 | 04:26:19 | gpue05 |
| 18560463 | `dpa2_cur3b_clip_H200_200k` | COMPLETED | 0:0 | 04:41:03 | gpue05 |

Initial tail check of `lcurve.out` confirms all runs reached step 200000 and wrote `model.ckpt-200000.pt`. The lower-LR and doc-medium cases show low training RMSE in the final rows but still have large intermittent validation spikes. The warmup and stronger-clipping cases have much larger final train/validation RMSE in the tail rows and look less promising from this quick check.
