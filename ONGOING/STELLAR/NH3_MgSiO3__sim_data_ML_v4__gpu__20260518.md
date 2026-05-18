# NH3_MgSiO3 sim_data_ML_v4 background simulations

Created: 2026-05-18 17:28 EDT  
Cluster: STELLAR (`stellar-intel.princeton.edu`)  
Partition/QOS: `gpu` / `gpu-stellar`  
Account: `astro`  
User: `ag5805`

## Purpose

Track the in-flight NH3_MgSiO3 DeePMD/LAMMPS/PLUMED MD simulations in:

- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/test__sim_data_ML_v4`

This note is for crash/session-loss resilience. Do not remove it from `ONGOING/` until the user confirms these simulations are finished and the note can be cleared.

## Status Snapshot

Snapshot time: 2026-05-18 17:28 EDT

| Job ID | State | Job name | Working directory | Output |
| --- | --- | --- | --- | --- |
| `2774980` | RUNNING on `stellar-m01g6` | `MD_BATCH_v10_i4_ZONE_1_1779130054` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/v10_i4/md/ZONE_1` | `multi_md_2774980.out`, `multi_md_2774980.err` |
| `2774981` | PENDING, reason `Priority` | `MD_BATCH_v10_i4_ZONE_2_1779130056` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/v10_i4/md/ZONE_2` | `multi_md_2774981.out`, `multi_md_2774981.err` |
| `2774982` | PENDING, reason `Priority` | `MD_BATCH_v10_i4_ZONE_3_1779130058` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/v10_i4/md/ZONE_3` | `multi_md_2774982.out`, `multi_md_2774982.err` |
| `2774983` | PENDING, reason `Priority` | `MD_BATCH_v10_i4_ZONE_4_1779130059` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/v10_i4/md/ZONE_4` | `multi_md_2774983.out`, `multi_md_2774983.err` |
| `2773691` | PENDING, reason `Dependency`; depends on `after:2774983` | `MD_BATCH_v10_i4_ALL_ZONES_1779075571` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/test__sim_data_ML_v4/v10_i4/md` | `multi_md_2773691.out`, `multi_md_2773691.err` |

## Submission Details

Production split-zone jobs:

- `2774980`: submitted from `sim_data_ML_v4/v10_i4/md/ZONE_1` with `sbatch --job-name=MD_BATCH_v10_i4_ZONE_1_1779130054 MULTI_sub.lmp_plumed.gpu.24h.sh`
- `2774981`: submitted from `sim_data_ML_v4/v10_i4/md/ZONE_2` with `sbatch --job-name=MD_BATCH_v10_i4_ZONE_2_1779130056 MULTI_sub.lmp_plumed.gpu.24h.sh`
- `2774982`: submitted from `sim_data_ML_v4/v10_i4/md/ZONE_3` with `sbatch --job-name=MD_BATCH_v10_i4_ZONE_3_1779130058 MULTI_sub.lmp_plumed.gpu.24h.sh`
- `2774983`: submitted from `sim_data_ML_v4/v10_i4/md/ZONE_4` with `sbatch --job-name=MD_BATCH_v10_i4_ZONE_4_1779130059 MULTI_sub.lmp_plumed.gpu.24h.sh`

Test all-zones job:

- `2773691`: submitted from `test__sim_data_ML_v4/v10_i4/md` with `sbatch --job-name=MD_BATCH_v10_i4_ALL_ZONES_1779075571 MULTI_sub.lmp_plumed.gpu.2h.sh`
- Dependency: `after:2774983`, so it should become eligible after Zone 4 job `2774983` completes.

## Resource Requests

Zone jobs `2774980`-`2774983`:

- Time limit: 24 h
- Nodes: 1
- Tasks: 2
- GPUs: 2 per node
- CPUs: 2 total, 1 per task
- Memory: 32 GB

Test all-zones job `2773691`:

- Time limit: 2 h
- Nodes: 2
- Tasks: 4
- GPUs: 2 per node, 4 total
- CPUs: 4 total, 1 per task
- Memory: 64 GB

## How To Check

```bash
squeue -u ag5805
squeue -j 2774980,2774981,2774982,2774983,2773691
sacct -j 2774980,2774981,2774982,2774983,2773691 --format=JobID,JobName%42,State,Elapsed,Start,End,NodeList%20
scontrol show job 2774980
scontrol show job 2774981
scontrol show job 2774982
scontrol show job 2774983
scontrol show job 2773691
```

Useful live-log checks:

```bash
tail -n 80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/v10_i4/md/ZONE_1/multi_md_2774980.out
tail -n 80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/v10_i4/md/ZONE_1/multi_md_2774980.err
```

Repeat the same pattern for `ZONE_2`, `ZONE_3`, `ZONE_4`, and the test all-zones directory as those jobs start.

## Resume / Next Step

1. Wait for `2774980`-`2774983` to finish.
2. Confirm the dependent test job `2773691` starts after `2774983` completes.
3. Inspect each `multi_md_<jobid>.out` and `multi_md_<jobid>.err` for normal completion or LAMMPS/PLUMED errors.
4. If a job fails, use its working directory and submission script above for diagnosis/resubmission.
5. Once the production zones and the dependent test run are confirmed complete, ask the user whether this `ONGOING/` note can be removed or archived.

## Notes

- The user requested the note under `/projects/bguf/akashgpt/run_scripts/helpful_scripts/ONGOING`, but that path does not exist in this session.
- The active local HELPFUL_SCRIPTS repo is `/projects/BURROWS/akashgpt/run_scripts/helpful_scripts`, so this note was created under `/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/ONGOING/STELLAR/`.
