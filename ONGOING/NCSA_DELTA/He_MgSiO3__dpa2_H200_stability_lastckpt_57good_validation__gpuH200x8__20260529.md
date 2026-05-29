# He_MgSiO3 DPA2 H200 Stability Last-Checkpoint 57-System Validation

Date: 2026-05-29
Cluster: NCSA Delta
Partition: `gpuH200x8`
Validation root: `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_dpa2_H200_stability_lastckpt_57good__20260528`
Slurm job: `18624058`

## Purpose

Run full `dp test` validation for only the recently completed H200 DPA-2 stability jobs, using their last 200k checkpoints and only the 57 in-domain MgSiO3 systems.

Excluded OOD systems: `n0211`, `n0217`, `n0221`, `n0226`.

## Cases

The validation array has four tasks:

- `dpa2_current_auto_3body_sel120_lowlr_H200_200k`
- `dpa2_current_auto_3body_sel120_warmup_H200_200k`
- `dpa2_current_auto_3body_sel120_clip1_H200_200k`
- `dpa2_doc_medium_auto_currentloss_H200_200k`

Older `current_auto_200k`, older non-H200 `doc_medium`, and `current_auto_1M_H200` are intentionally not included.

## Checks

```bash
squeue -j 18624058 -o "%i %T %P %M %l %j %R"
sacct -j 18624058 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Partition -P
```

After completion:

```bash
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_dpa2_H200_stability_lastckpt_57good__20260528
/work/nvme/bguf/agupta46/softwares/conda_envs_dir_secondary/envs/ALCHEMY_env/bin/python summarize_validation.py
```

Expected summary: `VALIDATION_SUMMARY.tsv` with per-system rows and aggregate E/F/V values for the 57-system set.

## Status Snapshot

2026-05-29 10:14 CDT: Submitted validation array job `18624058`.

2026-05-29 10:15 CDT Slurm snapshot:

| Job ID | State | Reason |
| --- | --- | --- |
| `18624058_[0-3]` | PENDING | Priority |
