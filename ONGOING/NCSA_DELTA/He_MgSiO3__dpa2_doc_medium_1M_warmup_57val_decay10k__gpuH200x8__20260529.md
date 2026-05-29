# He_MgSiO3 DPA2 Doc-Medium 1M Warmup 57-Val Decay10k

Date: 2026-05-29
Cluster: NCSA Delta
Partition: `gpuH200x8`
Working directory: `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_doc_medium_auto_currentloss_1M_H200_warmup_57val_decay10k`
Shared JSON: `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/shared/train_dpa2_doc_medium_auto_currentloss_1M_H200_warmup_57val_decay10k.json`
Slurm job: `18624079`

## Purpose

Test the old `doc_medium_auto_currentloss_200k` setup extended to 1M steps, with a 20k-step LR warmup and only the 57 in-domain MgSiO3 systems in the in-training validation set.

Intentional changes from old `doc_medium_auto_currentloss_200k`:

- `training.numb_steps = 1000000`
- Added `training.validation_data` with 57 MgSiO3 systems.
- Excluded OOD systems: `n0211`, `n0217`, `n0221`, `n0226`.
- Added `learning_rate.warmup_steps = 20000`.
- Added `learning_rate.warmup_start_factor = 0.05`.
- Preserved old `learning_rate.decay_steps = 10000`.

## Checks

```bash
squeue -j 18624079 -o "%i %T %P %M %l %j %R"
sacct -j 18624079 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Partition -P
tail -n 40 /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_doc_medium_auto_currentloss_1M_H200_warmup_57val_decay10k/log.train
```

## Status Snapshot

2026-05-29 10:17 CDT: Submitted job `18624079`.

## Follow-Up After Completion

Once this doc-medium 1M run and the matching current-auto 1M control are both complete, set up a network-size scaling test:

- increase network size for the `doc_medium` variant;
- increase network size for the `current_auto` variant;
- keep the comparison controlled against these completed 1M baselines;
- validate the final checkpoints on the 57 in-domain MgSiO3 systems before drawing conclusions.
