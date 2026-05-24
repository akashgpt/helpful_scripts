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



Snapshot time: 2026-05-23 16:40 EDT

Current scheduler state: `squeue -u ag5805` is empty on Stellar. The four `v10_i5` batched LAMMPS/PLUMED MD jobs all completed successfully according to `sacct`:

- `2783128` `MD_BATCH_v10_i5_ZONE_1_1779474142`: COMPLETED, elapsed `10:32:41`, ended `2026-05-23T02:27:30`.
- `2783129` `MD_BATCH_v10_i5_ZONE_2_1779474145`: COMPLETED, elapsed `07:59:32`, ended `2026-05-23T00:36:07`.
- `2783130` `MD_BATCH_v10_i5_ZONE_3_1779474148`: COMPLETED, elapsed `06:54:25`, ended `2026-05-23T07:30:51`.
- `2783131` `MD_BATCH_v10_i5_ZONE_4_1779474151`: COMPLETED, elapsed `08:07:40`, ended `2026-05-23T10:35:28`.

The ALCHEMY workflow is still active as local/orphaned bash processes on `stellar-intel.princeton.edu`, not as Slurm jobs:

- Level 1: PID `1852268`, cwd `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/v10_i5`.
- Level 2: PID `1852548`, cwd `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/v10_i5/md`.
- Level 2 log says all Level 3 post-MD/recal scripts have been submitted and it is waiting for them to finish; latest wait heartbeat seen at `2026-05-23 16:37:20 EDT`.
- `done_lmp` markers are present across the `v10_i5/md/ZONE_*` configuration directories; many directories now have `running_recal` markers, so the active phase is post-MD/recal orchestration.

Next check: inspect the Level 3 config logs and recal markers to see whether the `running_recal` scripts are genuinely progressing, waiting on hidden/finished recal jobs, or stuck behind stale marker files.


Snapshot time: 2026-05-22 15:56 EDT

Follow-up after fix/restart: Slurm job `2783128` is running as `MD_BATCH_v10_i5_ZONE_1_1779474142` on `stellar-m01g2` from `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/v10_i5/md/ZONE_1` using `MULTI_sub.lmp_plumed.gpu.24h.sh`.

Early health check:

- Job state: RUNNING on `gpu` partition, 1 node, 2 GPUs, started `2026-05-22 15:54:49 EDT`.
- Active first wave: `5MgSiO3_55NH3` and `10MgSiO3_50NH3`.
- Generated `plumed.dat` files contain the fixed `cn_Ratio` line with `VAR=x,y,z,w,v`.
- Both active runs passed the old PLUMED failure point and were printing normal MTMB thermo through about step `3300` at the check time.
- No current `PLUMED error`, `MATHEVAL`, `Aborted`, `srun: error`, or `error_lmp` signal was found in the current logs.

Next check: wait for job `2783128` to complete all 7 run directories, then inspect `multi_md_2783128.out`, `multi_md_2783128.err`, and per-config `done_lmp`/`error_lmp` markers.

Snapshot time: 2026-05-22 12:58 EDT

Latest production iteration `v10_i5` in `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4` is stopped with an error. No `NH3_MgSiO3` Slurm jobs are currently running in `squeue -u ag5805`; only unrelated NH3_H2 work was active at this snapshot.

Root cause found in the batched LAMMPS/PLUMED MD stage:

- Level 1 log reports `Error in training iteration v10_i5 (#2)`.
- Level 2 submitted four batched MD jobs for `v10_i5/md/ZONE_1` through `ZONE_4`; all four finished queue execution, then post-MD/recal processing found no valid MD data.
- Each zone recorded seven failed run markers in `md_run_counter_error` / `multi_md_*.out`.
- Representative logs such as `v10_i5/md/ZONE_1/10MgSiO3_50NH3/log.run_lmp_batch` show LAMMPS completed the 1000-step NPT relaxation, then aborted when adding PLUMED:

```text
ERROR in input to action MATHEVAL with label cn_Ratio : Using more than 3 arguments you should explicitly write their names with VAR
```

Bad generated/template line:

```text
cn_Ratio: MATHEVAL ARG=cnNN_raw,nN,cn_NO,beta,eps FUNC=(x/(2*y))/((x/(2*y))+(w*z)+v) PERIODIC=NO
```

This line exists in the active setup templates:

- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/setup_MLMD/ZONES_input_files/ZONE_*/plumed.dat.template`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_MgSiO3/sim_data_ML_v4/setup_MLMD/ref_files/ref_ZONES_input_files/ZONE_i__MgSiOHN_MOL_SYSTEM__MgSiO3_NH3/plumed.dat.template`

Likely fix before restart: add explicit variable names to that `MATHEVAL`, e.g. `VAR=x,y,z,w,v`, and regenerate/restart the failed MD stage.

Update 2026-05-22: fixed the source `setup_MLMD` templates only for restart purposes. Verified `setup_MLMD` has zero remaining old `cn_Ratio` MATHEVAL lines and 10 corrected template copies under `ZONES_input_files/ZONE_*` and `ref_files/ref_ZONES_input_files`.

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
