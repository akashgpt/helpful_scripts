# H2S/NH3/H2O VASP MLFF Reference on TIGER

**Date prepared:** 2026-05-21  
**Cluster:** Princeton TIGER CPU Slurm environment  
**Source campaign:** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data`  
**Excluded source subtree:** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/small_cells`

## Purpose

This note records the practical setup pattern used for the H2S/NH3/H2O VASP MLFF campaign. It is meant as a compact reference for future liquid and two-phase VASP MLFF calculations, not as a raw-output archive. The inventory file in this directory summarizes the non-`small_cells` run folders that contained VASP inputs.

## Source Layout

The campaign has two main run types:

- Pure-fluid training boxes: `pure_H2O`, `pure_H2S`, and `pure_NH3`.
- Two-phase mixed boxes: `H2O_NH3`, `H2O_H2S`, and `H2S_NH3`.

The `setup_DIR` folder contains setup material rather than completed VASP run directories, so it was used as context but not listed as a production run in the inventory.

## Shared VASP Setup

The common electronic-structure setup across the sampled run directories is:

- Functional: `METAGGA = R2SCAN`
- Plane-wave cutoff: `ENCUT = 500.00 EV`
- Angular correction: `LASPH = .TRUE.`
- Electronic algorithm: `ALGO = N`
- Real-space projection: `LREAL = A`
- Smearing: `ISMEAR = -1`, `SIGMA = 0.258520`
- Symmetry off: `ISYM = 0`
- Long trajectories: `NSW = 1000000`, `NBLOCK = 1`, `KBLOCK = 1`
- TIGER parallel layout in the run scripts: usually 8 nodes, 14 MPI tasks per node, and 8 CPUs per task for two-phase runs.
- VASP executable in the representative scripts: `/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std`

The sampled `RUN_VASP.sh` scripts load:

```bash
module purge
module load intel-oneapi/2024.2
module load intel-mpi/oneapi/2021.13
module load intel-mkl/2024.2
module load hdf5/oneapi-2024.2/1.14.4
```

They run VASP with `srun "$VASP_BIN"`, write `log.run_sim`, and use marker files such as `running_RUN_VASP` and `done_RUN_VASP`.

## MLFF Setup Pattern

All sampled production run directories use VASP MLFF training mode:

```text
ML_LMLFF  = .TRUE.
ML_MODE   = train
ML_NMDINT = 50
```

The pure-fluid training runs generally use a smaller MLFF capacity:

- `ML_MB = 4000`
- `ML_MCONF = 600`
- `ML_CTIFOR = 1.4E-01` in the later `_b` pure-fluid folders; the earlier pure folders leave the active `ML_CTIFOR` unset or commented.

The two-phase runs use larger capacities:

- Initial H-isotope two-phase attempts commonly used `ML_MB = 16000`, `ML_MCONF = 2000`, `ML_CTIFOR = 3E-01`, and `POTIM = 0.25`.
- Later or D-isotope continuations commonly used `ML_MB = 8000` or `16000`, `ML_MCONF = 1000`, `ML_CTIFOR = 3E-01`, and `POTIM = 0.5` or `1.0`.
- One H2O/NH3 trial used `ML_MB = 24000`, `ML_MCONF = 1000`; that case is useful as a cautionary comparison, not a default.

The scientific pattern is to first train MLFFs on pure molecular liquids under pressure, then use the learned force-field state files from mixed/two-phase training restarts to extend trajectories. The important files to keep together when restarting or copying a run are `INCAR`, `POSCAR`, `KPOINTS`, `POTCAR`, `RUN_VASP.sh`, `ML_AB`, `ML_ABN`, `ML_FFN`, and `ML_LOGFILE`.

## Pure-Fluid Runs

The pure-fluid folders are NPT-style MLFF training runs:

- `ISIF = 3`
- `MDALGO = 3`
- `PSTRESS = 200` kbar, corresponding to 20 GPa
- `TEBEG = TEEND = 3000`
- `POTIM = 0.25`
- `ML_MB = 4000`, `ML_MCONF = 600`

The later `_b` pure-fluid folders have more realistic elongated cells for the subsequent two-phase construction:

- `pure_H2O/250_molecules_b`: `H O`, counts `500 250`, cell about `17 x 17 x 14.749` A
- `pure_H2S/256_molecules_b`: `H S`, counts `512 256`, cell about `17 x 17 x 23.6536` A
- `pure_NH3/256_molecules_b`: `H N`, counts `768 256`, cell about `17 x 17 x 18.903` A

## Two-Phase Runs

The mixed folders are fixed-cell, two-phase MLFF training runs:

- `ISIF = 2`
- No active `MDALGO = 3` or `PSTRESS` in the sampled mixed `INCAR` files
- `TEBEG = 3000`
- `SMASS = 0.00`
- Species order in `POSCAR`/`POTCAR` must be preserved
- `KPOINTS` is the Gamma-only molecular-dynamics style setup

Representative two-phase cells:

| System | Species order | Counts | Cell (A) | Typical MLFF capacity |
| --- | --- | ---: | --- | --- |
| H2O/NH3 | `H O N` | `1268 250 256` | `17 x 17 x 34.28` | `ML_MB = 16000`, `ML_MCONF = 1000-2000` |
| H2O/H2S | `H O S` | `1012 250 256` | `17 x 17 x 38.76` | `ML_MB = 8000-16000`, `ML_MCONF = 1000-2000` |
| H2S/NH3 | `H N S` | `1280 256 256` | `17 x 17 x 43.38` | `ML_MB = 8000-16000`, `ML_MCONF = 1000-2000` |

The D-isotope folders keep the same elemental `POSCAR` species labels (`H`, `O`, `N`, `S`) because VASP still uses element potentials. The isotope difference is represented by the structure/run naming and the chosen MD timestep; the D runs sampled here used `POTIM = 1.0`.

## Restart and Continuation Practice

The campaign uses suffixes such as `__a`, `__b`, `__c`, and later letters as restart or trial variants. Many folders contain both current and historical Slurm output files, so the scheduler files alone do not define the current run state. The useful restart state is the combination of the latest `POSCAR`/trajectory state plus the MLFF files:

- `ML_AB`: learned ab initio database
- `ML_ABN`: normalized/next MLFF database written by VASP
- `ML_FFN`: trained force-field state
- `ML_LOGFILE`: MLFF decisions, accurate/learning status, and BEEF/SFF diagnostics

For future references, record which ML files were copied into the next suffix and which source folder they came from. That is more useful than only recording the Slurm job ID.

## Representative OUTCAR Timing Check

Per the source-campaign request, I parsed only one representative `OUTCAR` in this pass:

```bash
python - <<'PY'
from pathlib import Path
from qmd.vasp.plot_vasp_step_speed import parse_loop_seconds, sliding_window_sum, trim_initial_steps

outcar_path = Path("/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/250_H2O__256_NH3__b/OUTCAR")
series = parse_loop_seconds(outcar_path, "H2O_NH3__b")
trimmed = trim_initial_steps(series, 1)
print(len(series.step_seconds), len(trimmed.step_seconds))
print(3600 * len(trimmed.step_seconds) / trimmed.step_seconds.sum())
for window in (10, 50, 100):
	wall_seconds = sliding_window_sum(trimmed.step_seconds, window)[-1]
	print(window, wall_seconds, 3600 * window / wall_seconds)
PY
```

Result for `H2O_NH3/250_H2O__256_NH3__b`:

| Metric | Value |
| --- | ---: |
| Parsed `LOOP+` timing entries | 2246 |
| Parsed entries after skipping the first step | 2245 |
| Whole-run `LOOP+` speed after skip | 282 steps/hour |
| Last 10-step wall time | 40.96 s |
| Last 10-step speed | 879 steps/hour |
| Last 50-step wall time | 204.89 s |
| Last 50-step speed | 879 steps/hour |
| Last 100-step wall time | 411.74 s |
| Last 100-step speed | 874 steps/hour |

This matches the earlier timing lesson in `benchmarks/vasp/TIGER/vasp_mlff_two_phase_timing__H2O_NH3__20260419.md`: timing comparisons should state the window. Whole-run speed is dominated by early training/refit behavior, while the recent-window speed describes the mature MLFF prediction regime.

## Cautions

- Do not treat `ML_MB = 24000` as the default. It provided extra headroom in H2O/NH3 but was not the best throughput point in the prior timing note.
- Be careful with sulfur systems at very large retained basis sizes. Prior runs showed MPI `internal_Bcast` negative-count failures in large-basis MLFF covariance broadcasts.
- Existing local VASP guidance records that hand-merged `ML_AB` files are not yet a reliable `ML_MODE = select` workflow here. Native single-system `ML_AB` use is safer than manually stitched `ML_AB` files unless a parser-compatible merge recipe is validated.
- Marker files such as `running_RUN_VASP` are filesystem snapshots, not proof that a job is live on 2026-05-21. Use `squeue` or Slurm accounting if the current live status matters.

## Files In This Entry

- `README.md`: this reference note.
- `run_inventory.tsv`: compact inventory of the non-`small_cells` run directories with VASP inputs.
- `incar_examples/`: source-copied example `INCAR` files for the key reusable setups:
  - `INCAR__pure_H2O_250_molecules_b__NPT_train_R2SCAN_MLFF` for pure-fluid NPT MLFF training.
  - `INCAR__H2O_NH3_250_H2O_256_NH3_b__two_phase_NVT_train_R2SCAN_MLFF` for two-phase fixed-cell MLFF training.
