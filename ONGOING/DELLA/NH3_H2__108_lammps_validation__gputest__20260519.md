# NH3_H2 108-system LAMMPS validation runs

## Status Snapshot

- Date: 2026-05-19 13:46:35 EDT
- Cluster: Della
- Partition observed in `squeue`: `gputest`
- Purpose: Generate LAMMPS `npt.dump` trajectories and prepare the follow-on all-frame VASP validation through the ASAP-style ALCHEMY route.
- Scope: Only the small `108_H2_108_NH3` case, at `1_GPa` and `33_GPa`.

## Submitted Jobs

| Pressure | Job ID | State at submission check | Working directory |
| --- | --- | --- | --- |
| `1_GPa` | `8463271` | `PD (Priority)` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/LMP/1_GPa` |
| `33_GPa` | `8463270` | `PD (Priority)` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/LMP/33_GPa` |

## Key Run Settings

- Submission script: `sub.lmp_npt.gpu.apptr.1h.sh`
- LAMMPS input: `in.lammps_npt_eq`
- Model: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/setup_MLMD/latest_trained_potential/pv_comp.pb`
- System size checked before submission: `648` atoms, with `540` type-1 H atoms and `108` type-2 N atoms in each pressure folder.
- Pressure targets:
  - `1_GPa`: `PZ = 1E4` bar
  - `33_GPa`: `PZ = 33E4` bar
- Temperature: `2000 K`
- Main trajectory output: `npt.dump`
- Final data output: `conf.lmp.end`
- Thermo/plot output: `log.lammps`, then `pressure_volume_energy_vs_time.png` if the post-run plotting step succeeds.

## How To Check

```bash
squeue -j 8463271,8463270
sacct -j 8463271,8463270 --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

After completion, inspect:

```bash
tail -80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/LMP/1_GPa/slurm-8463271.out
tail -80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/LMP/33_GPa/slurm-8463270.out
ls -lh /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/LMP/1_GPa/npt.dump
ls -lh /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/LMP/33_GPa/npt.dump
```

## Later Status Notes

- 2026-05-19: Jobs 8463271 and 8463270 completed LAMMPS trajectory generation but had Slurm FAILED status because the old post-run plotting path was missing. Manual plotting was rerun with ALCHEMY_env and $ALCHEMY__main__MLDP/ALCHEMY/plot_lammps_thermo.py, producing pressure_volume_energy_vs_time.png in both 1_GPa and 33_GPa. npt.dump exists in both folders.
- 2026-05-19 13:46 EDT: Prepared the VASP multi-run launcher for the generated validation frame directories:
  `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/MULTI_RUN_VASP.sh`.
  No VASP Slurm job was submitted in this update.
- 2026-05-19 15:23 EDT: Two single-directory VASP AIMD jobs are submitted on Stellar `pu` for the converted `33_GPa` structure:
  job `2777680` for `DFT/33_GPa` and job `2777681` for `DFT/33_GPa__longer`.

## VASP Validation Launcher

- Script: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/MULTI_RUN_VASP.sh`
- Intended working directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3`
- Cluster/partition intent: Della/Stellar/Tiger CPU-style VASP launcher; no explicit partition line is active.
- VASP binary selection: hostname-aware `VASP_BIN`, using VASP `6.6.0`.
- Environment: `intel-oneapi/2024.2`, `intel-mpi/oneapi/2021.13`, `intel-mkl/2024.2`, `hdf5/oneapi-2024.2/1.14.4`, `anaconda3/2025.12`, `ALCHEMY_env`.
- Slurm sizing currently in script: `10` nodes, `12` tasks per node, `8` CPUs per task, `24:00:00` walltime.
- Per-frame/run timeout: `86400` seconds.
- Directory discovery: recursive, `SEARCH_MODE="folders_at_all_depths"`.
- Run selection: folders containing `INCAR` and a file matching `*to_RUN*`.
- Skip behavior: skips folders that already contain `running_RUN_VASP` or `done_RUN_VASP`.
- Completion markers: creates `done_RUN_VASP` on success and `failed_RUN_VASP` on failed VASP exit.
- Retry behavior: removes `to_RUN*` only after a successful VASP run, so failed calculations remain marked for inspection/retry.

Submit from the intended working directory:

```bash
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3
sbatch MULTI_RUN_VASP.sh
```

Before submission, preview selected folders:

```bash
find . -mindepth 1 -type d | sort | while read -r d; do
    test -f "$d/INCAR" || continue
    find "$d" -maxdepth 1 -type f -name '*to_RUN*' | grep -q . || continue
    find "$d" -maxdepth 1 -type f \( -name 'running_RUN_VASP' -o -name 'done_RUN_VASP' \) | grep -q . && continue
    printf '%s\n' "$d"
done
```

## Single DFT/33_GPa VASP Runs

These are direct VASP AIMD submissions for the converted `33_GPa` LAMMPS structure, separate from the recursive multi-run launcher above.

| Directory | Job ID | Snapshot state | QOS | Walltime | Slurm layout | INCAR parallel setting |
| --- | --- | --- | --- | --- | --- | --- |
| `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/DFT/33_GPa` | `2777680` | `PENDING`, reason `Priority` | `pu-medium-stellar` | `48:00:00` | `4` nodes, `12` tasks/node, `8` CPUs/task, `4G` mem/CPU | `NPAR = 12` |
| `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/DFT/33_GPa__longer` | `2777681` | `PENDING`, reason `QOSGrpCpuLimit` | `pu-long-stellar` | `168:00:00` | `4` nodes, `12` tasks/node, `8` CPUs/task, `4G` mem/CPU | `NPAR = 12` |

Shared setup:

- VASP binary: `/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std`
- Modules: `intel-oneapi/2024.2`, `intel-mpi/oneapi/2021.13`, `intel-mkl/2024.2`, `hdf5/oneapi-2024.2/1.14.4`
- Structure: `648` atoms from the `33_GPa` LAMMPS `conf.lmp`, with POSCAR species order `H N` and counts `540 108`.
- INCAR core settings: `ENCUT = 800`, `METAGGA = R2SCAN`, `LASPH = .TRUE.`, `NSW = 20000`, `POTIM = 0.5`, `TEBEG/TEEND = 2000 K`.
- Benchmark note: Stellar benchmark data suggested fastest known layout for the closest VASP AIMD case was `4` nodes, `8` tasks/node, `12` CPUs/task with `NPAR = 24`; the submitted jobs currently use `12` tasks/node, `8` CPUs/task with `NPAR = 12`.

Check status:

```bash
squeue -j 2777680,2777681
scontrol show job 2777680
scontrol show job 2777681
sacct -j 2777680,2777681 --format=JobID,JobName,Partition,QOS,State,Elapsed,ExitCode
```

After either job starts or finishes, inspect:

```bash
tail -80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/DFT/33_GPa/slurm-2777680.out
tail -80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/DFT/33_GPa__longer/slurm-2777681.out
tail -80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/DFT/33_GPa/log.run_sim
tail -80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/DFT/33_GPa__longer/log.run_sim
```

## Next Step Once Done

1. Confirm both LAMMPS jobs completed successfully.
2. Confirm `npt.dump` exists in both pressure folders and contains the expected frames.
3. Convert all dumped frames into VASP frame directories for the ASAP-style validation route.
4. Confirm the VASP frame directories contain `INCAR` plus `to_RUN*` markers.
5. Monitor jobs `2777680` and `2777681`; when one starts, check that `log.run_sim`, `OUTCAR`, and `XDATCAR` grow normally.
6. After completion, compare the regular and longer `33_GPa` runs and decide whether the recursive `MULTI_RUN_VASP.sh` validation sweep is still needed.
7. For any future recursive validation submission, submit `MULTI_RUN_VASP.sh` only after the dump contents and frame count look sensible, then update this note with the new Slurm job ID.

Do not remove this note until the user confirms these validation LAMMPS runs are fully handled.

## Reassessment - 2026-05-21 07:10 EDT

Current Della queue has no active jobs from the 108-case validation LAMMPS/VASP frame workflow. The original LAMMPS jobs `8463270` and `8463271` remain historical/completed; no new recursive `MULTI_RUN_VASP.sh` job is currently queued on Della.

Current Stellar status for the single direct `33_GPa` VASP checks:

| Job ID | Directory | State | Runtime | Assessment |
| --- | --- | --- | --- | --- |
| `2777680` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/DFT/33_GPa` | `RUNNING` on Stellar `pu` | `1-08:11:59` at check time | Healthy-looking VASP AIMD. `OUTCAR`, `OSZICAR`, `XDATCAR`, `vasprun.xml`, and `vaspout.h5` are growing. Latest `OSZICAR` check showed ionic step `1824`, `T=2007 K`. |
| `2777681` | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/DFT/33_GPa__longer` | `CANCELLED by 359379` | `00:00:00` | No longer active. |

Current next checks from Della:

```bash
ssh stellar 'squeue -j 2777680'
ssh stellar 'sacct -j 2777680,2777681 --format=JobID,JobName,Partition,QOS,State,Elapsed,Start,End,ExitCode'
ssh stellar 'tail -5 /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/VALIDATION_RUNS/H2_NH3/108_H2_108_NH3/DFT/33_GPa/OSZICAR'
```

Keep this note in `ONGOING/DELLA` for the local Della-side 108-case validation history. Remote follow-up should be refreshed only when explicitly requested.

## Update - 2026-05-22 23:27 EDT

Della has no active scheduler jobs tied to the original 108-case LAMMPS validation note in the current local queue snapshot. Remote follow-up status was not refreshed here because generic `ONGOING/` updates should only touch the local cluster unless explicitly requested.
