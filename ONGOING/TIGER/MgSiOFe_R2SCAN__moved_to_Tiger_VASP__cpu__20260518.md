# MgSiOFe R2SCAN VASP Background Runs Moved To Tiger

Last updated: 2026-05-18 18:02:10 EDT on `tiger3.princeton.edu`

## Purpose

Track ongoing MgSiOFe R2SCAN VASP frame recalculation / data-collection runs that were moved to Tiger, so the state can be resumed after a session crash or context loss.

## Root Directory

`/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/main_runs/moved_to_Tiger`

Top-level campaign directories:

- `completed`
- `deepmd_collection_TRAIN_OG`
- `deepmd_collection_TRAIN_OG__500GPa_9000K`

## Active Slurm Snapshot

Command used:

```bash
squeue -u "$USER" -o '%.18i %.9P %.60j %.8T %.19S %.10M %.9l %.6D %R %Z'
```

Relevant MgSiOFe jobs at snapshot time:

| Job ID | Partition | Name | State | Start time | Elapsed | Limit | Nodes | Reason / Node list | Work dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2685107` | `cpu` | `deepmd_collection_TRAIN_OG_3311362482` | `PENDING` | `2026-05-18T19:56:30` | `0:00` | `6-00:00:00` | `26` | `(QOSGrpCpuLimit)` | `.../moved_to_Tiger/deepmd_collection_TRAIN_OG` |
| `2807501` | `cpu` | `qmd` | `PENDING` | `2026-05-18T19:56:30` | `0:00` | `5:00:00` | `4` | `(Priority)` | `.../deepmd_collection_TRAIN_OG__500GPa_9000K/500GPa_9000K/960` |
| `2807427` | `cpu` | `multi_run` | `PENDING` | `2026-05-18T19:57:00` | `0:00` | `1-00:00:00` | `35` | `(Priority)` | `.../moved_to_Tiger/deepmd_collection_TRAIN_OG/it5_mtmb` |
| `2807432` | `cpu` | `multi_run` | `PENDING` | `N/A` | `0:00` | `1-00:00:00` | `26` | `(Priority)` | `.../moved_to_Tiger/deepmd_collection_TRAIN_OG/it4_mtmb` |
| `2778964` | `cpu` | `multi_run` | `PENDING` | `N/A` | `0:00` | `3-00:00:00` | `40` | `(Priority)` | `.../moved_to_Tiger/deepmd_collection_TRAIN_OG__500GPa_9000K` |
| `2807443` | `cpu` | `multi_run` | `PENDING` | `N/A` | `0:00` | `3-00:00:00` | `40` | `(Priority)` | `.../moved_to_Tiger/deepmd_collection_TRAIN_OG/it3_mtmb` |
| `2778780` | `cpu` | `deepmd_collection_TRAIN_OG__500GPa_9000K_2190848856` | `RUNNING` | `2026-05-16T16:09:46` | `2-01:50:57` | `3-00:00:00` | `40` | Tiger GPU-node CPU allocation | `.../moved_to_Tiger/deepmd_collection_TRAIN_OG__500GPa_9000K` |

Note: there are unrelated NH3/H2 recalculation jobs in the same `squeue` output; they are not part of this MgSiOFe note.

## Progress Snapshot

Counts were collected with `find` over each campaign directory. `OUTCAR_elapsed_tail` means the final 20 lines of `OUTCAR` contain `Elapsed`, which is the completion check used by the local auto-resubmit helper.

| Campaign | Numeric run dirs | `to_RUN` | `running_RUN_VASP` | `done_RUN_VASP` | `not_done` | `OUTCAR` files | `OUTCAR_elapsed_tail` | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `completed` | 3000 | 0 | 0 | 3000 | 2839 | 3000 | 3000 | Complete by `done_RUN_VASP` and `OUTCAR` tail checks; legacy `not_done` markers remain. |
| `deepmd_collection_TRAIN_OG` | 1284 | 848 | 0 | 436 | 1134 | 1101 | 436 | Ongoing; tagged auto-resubmit job `2685107` plus new `it*_mtmb` batches are pending. |
| `deepmd_collection_TRAIN_OG__500GPa_9000K` | 1000 | 2 | 9 | 991 | 908 | 1000 | 991 | Nearly complete, but active/pending jobs remain, including targeted frame `960` rerun. |

## Newly Queued / Targeted Runs

These are the new or targeted active items under `moved_to_Tiger` at the 2026-05-18 18:02 EDT snapshot.

| Path | Job ID | State | Nodes | Limit | Current markers / status |
| --- | --- | --- | ---: | --- | --- |
| `deepmd_collection_TRAIN_OG__500GPa_9000K/500GPa_9000K/960` | `2807501` | `PENDING` | 4 | `5:00:00` | Targeted rerun using `RUN_VASP__reference.sh`; `done_RUN_VASP` and `not_done` both existed at inspection time, with no `to_RUN` or `running_RUN_VASP`. |
| `deepmd_collection_TRAIN_OG/it3_mtmb` | `2807443` | `PENDING` | 40 | `3-00:00:00` | 218 numeric dirs, 218 `to_RUN`, 0 `done_RUN_VASP`, 218 `OUTCAR`. |
| `deepmd_collection_TRAIN_OG/it4_mtmb` | `2807432` | `PENDING` | 26 | `1-00:00:00` | 52 numeric dirs, 52 `to_RUN`, 0 `done_RUN_VASP`, 52 `OUTCAR`. |
| `deepmd_collection_TRAIN_OG/it5_mtmb` | `2807427` | `PENDING` | 35 | `1-00:00:00` | 288 numeric dirs, 288 `to_RUN`, 0 `done_RUN_VASP`, 105 `OUTCAR`. |

Frame `960` note: the previous `log.run_sim` shows it was run by parent job `2778780` and completed on 2026-05-18 around 16:23 EDT, but the run log also contains the VASP warning that no readable KPOINTS file was found and automatic k-point generation was used as fallback. The targeted rerun job `2807501` is therefore tracked separately from the broader `deepmd_collection_TRAIN_OG__500GPa_9000K` campaign.

## Submission / Resubmission Setup

Primary scripts in active directories:

- `deepmd_collection_TRAIN_OG/MULTI_RUN_VASP__TRAIN.sh`
  - `#SBATCH --nodes=26`
  - `#SBATCH --ntasks-per-node=14`
  - `#SBATCH --cpus-per-task=8`
  - `#SBATCH --time=144:00:00`
  - Uses `vasp_std` from `/scratch/gpfs/BURROWS/akashgpt/softwares/vasp.6.4.3/bin`
  - Loads `intel-oneapi/2024.2`, `intel-mpi/oneapi/2021.13`, `intel-mkl/2024.2`, `hdf5/oneapi-2024.2/1.14.4`, `anaconda3/2025.12`, then activates `hpc-tools`
  - Scans all subfolders for `INCAR` plus `to_RUN`, skipping `running_RUN_VASP` and `done_RUN_VASP`

- `deepmd_collection_TRAIN_OG__500GPa_9000K/MULTI_RUN_VASP__TRAIN.sh`
  - `#SBATCH --nodes=70`
  - `#SBATCH --ntasks-per-node=14`
  - `#SBATCH --cpus-per-task=8`
  - `#SBATCH --time=24:00:00`
  - Same VASP binary and module setup as above

- `deepmd_collection_TRAIN_OG/it3_mtmb/MULTI_RUN_VASP__TRAIN.sh`
  - `#SBATCH --nodes=40`
  - `#SBATCH --ntasks-per-node=14`
  - `#SBATCH --cpus-per-task=8`
  - `#SBATCH --time=72:00:00`
  - Uses `vasp_std`; runs one VASP frame per node with GNU Parallel timeout `86400` seconds per individual run

- `deepmd_collection_TRAIN_OG/it4_mtmb/MULTI_RUN_VASP__TRAIN.sh`
  - `#SBATCH --nodes=26`
  - `#SBATCH --ntasks-per-node=14`
  - `#SBATCH --cpus-per-task=8`
  - `#SBATCH --time=24:00:00`
  - Uses the same one-frame-per-node GNU Parallel driver as `it3_mtmb`

- `deepmd_collection_TRAIN_OG/it5_mtmb/MULTI_RUN_VASP__TRAIN.sh`
  - `#SBATCH --nodes=35`
  - `#SBATCH --ntasks-per-node=14`
  - `#SBATCH --cpus-per-task=8`
  - `#SBATCH --time=24:00:00`
  - Uses the same one-frame-per-node GNU Parallel driver as `it3_mtmb`

- `deepmd_collection_TRAIN_OG__500GPa_9000K/500GPa_9000K/960/RUN_VASP__reference.sh`
  - `#SBATCH --nodes=4`
  - `#SBATCH --ntasks-per-node=14`
  - `#SBATCH --cpus-per-task=8`
  - `#SBATCH --time=5:00:00`
  - Uses `/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std`
  - Loads `intel-oneapi/2024.2`, `intel-mpi/oneapi/2021.13`, `intel-mkl/2024.2`, and `hdf5/oneapi-2024.2/1.14.4`

Auto-resubmit helper:

- `auto_resubmit__RUN_VASP.sh`
- Stop file: `.stop_resubmit`
- Lock file: `.auto_resubmit.lock`
- Log file: `log.auto_resubmit`
- Sleep interval: `10800` seconds
- Stable job tags:
  - `deepmd_collection_TRAIN_OG_3311362482`
  - `deepmd_collection_TRAIN_OG__500GPa_9000K_2190848856`

Recent auto-resubmit log entries show:

- `deepmd_collection_TRAIN_OG`: submitted job `2685107` on 2026-05-07 00:37:59 EDT and continued seeing the tagged job as running/pending afterward.
- `deepmd_collection_TRAIN_OG__500GPa_9000K`: submitted tagged job `2778780` on 2026-05-16 11:39:47 EDT.

## How To Check

Queue:

```bash
squeue -u "$USER" -o '%.18i %.9P %.60j %.8T %.19S %.10M %.9l %.6D %R %Z' | rg 'MgSiOFe|deepmd_collection_TRAIN_OG|multi_run|moved_to_Tiger'
```

Specific new jobs:

```bash
squeue -j 2807501,2807443,2807432,2807427,2778780,2778964,2685107 -o '%.18i %.9P %.80j %.8T %.19S %.10M %.9l %.6D %R %Z'
```

Progress counts:

```bash
for d in /scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/main_runs/moved_to_Tiger/deepmd_collection_TRAIN_OG*; do
	printf '%s\n' "$d"
	printf 'to_RUN '; find "$d" -name 'to_RUN' -type f | wc -l
	printf 'running_RUN_VASP '; find "$d" -name 'running_RUN_VASP' -type f | wc -l
	printf 'done_RUN_VASP '; find "$d" -name 'done_RUN_VASP' -type f | wc -l
done
```

Auto-resubmit logs:

```bash
tail -n 80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/main_runs/moved_to_Tiger/deepmd_collection_TRAIN_OG/log.auto_resubmit
tail -n 80 /scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/main_runs/moved_to_Tiger/deepmd_collection_TRAIN_OG__500GPa_9000K/log.auto_resubmit
```

## Resume / Intervention Notes

- Do not remove this `ONGOING` note automatically. Clear it only after explicit user confirmation that the campaign is finished.
- To stop future resubmits after the current Slurm job finishes, create `.stop_resubmit` in the relevant campaign directory.
- If a Slurm job disappears while `to_RUN` markers remain, restart the controller from that campaign directory:

```bash
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/main_runs/moved_to_Tiger/deepmd_collection_TRAIN_OG
nohup bash auto_resubmit__RUN_VASP.sh &
```

or:

```bash
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/main_runs/moved_to_Tiger/deepmd_collection_TRAIN_OG__500GPa_9000K
nohup bash auto_resubmit__RUN_VASP.sh &
```

- Before restarting a controller, check `squeue` for the stable tag and inspect `.auto_resubmit.lock` / `log.auto_resubmit` to avoid duplicate resubmit loops.

## Next Step

Monitor targeted frame `960` job `2807501`, new `it3_mtmb`/`it4_mtmb`/`it5_mtmb` jobs `2807443`, `2807432`, and `2807427`, and the broader campaign jobs `2778780`, `2778964`, and `2685107`. After each job leaves the queue, refresh marker counts and check whether any `to_RUN`, `running_RUN_VASP`, or inconsistent `done_RUN_VASP` plus `not_done` markers remain.
