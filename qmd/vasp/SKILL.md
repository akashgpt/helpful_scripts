---
name: qmd-vasp
description: Use when working with the local `qmd/vasp` helpers for VASP job submission, continuation, merged multi-part runs, OUTCAR-based analysis (standard or MLFF), EOS-guided cell-size estimation, snapshot/MSD/speciation analysis, or visualization in this helpful_scripts repo.
---

# QMD VASP

This skill is for the repo-local VASP helper scripts under `qmd/vasp/`. The folder mixes active workflows, historical variants, and a few specialized subfolders, so start from the live entry points below instead of reading everything.

## Start Here

Read these first, in roughly this order:

- `RUN_VASP/LATEST_REFERENCE_FOR_ALL_PU_CLUSTERS/RUN_VASP__reference.sh`
- `RUN_VASP/LATEST_REFERENCE_FOR_ALL_PU_CLUSTERS/MULTI_RUN_VASP__reference.sh`
- `data_4_analysis.sh` (general dispatcher) — and its specialized siblings `data_4_analysis__standard.sh` and `data_4_analysis__MLFF.sh`
- `source__data_4_analysis__for_all.sh`
- `eos_fit__V_at_P.py`
- `merge_vasp_runs.py`
- `continue_run_ase.py`

Use cluster-specific launchers under `RUN_VASP/TIGER3`, `RUN_VASP/TIGER`, `RUN_VASP/STELLAR`, or `RUN_VASP/FRONTERA` only after checking which cluster the user is actually targeting.

## What This Folder Actually Does

- Submit or template VASP jobs with SLURM wrappers in `RUN_VASP/` (single and multi-run, CPU and GPU references).
- Continue long AIMD runs by creating chained directories (`...a`, `...b`, ...).
- Merge chained runs back into one logical dataset.
- Extract time-series data from `OUTCAR` into `analysis/` — separate code paths for standard AIMD and for MLFF training runs.
- Estimate the volume or cell size that best matches a target pressure.
- Summarize MLFF training (storage, basis usage) from `ML_LOGFILE`.
- Profile per-step elapsed time and visualize snapshot/trajectory frames.
- Support a few chemistry-specific analyses and a retraining setup for R2SCAN.

## Workflow Map

### Single run or reference run

- The current single-run reference launcher is `RUN_VASP/LATEST_REFERENCE_FOR_ALL_PU_CLUSTERS/RUN_VASP__reference.sh` — a hostname-aware SLURM wrapper that chooses module loads and `vasp_std` paths for Princeton and Delta-like systems.
- `MULTI_RUN_VASP__reference.sh` is the multi-run (batch) reference launcher.
- `MULTI_RUN_VASP__GPU__reference.sh` and `MULTI_RUN_VASP_gpu__reference_v2.sh` are the GPU equivalents.
- All write `log.run_sim` plus marker files `running_RUN_VASP` and `done_RUN_VASP` (the legacy `sub_vasp.sh` name was renamed repo-wide to `RUN_VASP.sh` — if you see the old name in archived benchmarks, treat as historical).

### Analyze one finished standard AIMD run

- `data_4_analysis.sh` is the active dispatcher; it auto-selects between standard and MLFF code paths.
- `data_4_analysis__standard.sh` is the explicit standard-AIMD parser (kept callable for cases where you want to bypass the dispatcher).
- They parse `OUTCAR` into pressure, volume, energy, and temperature time-series under `analysis/`, detect TI-style doubled-energy patterns, call `peavg.sh`, write `analysis/peavg_summary.out`, and produce a quick-look multi-panel plot.

### Analyze a finished MLFF training run

- `data_4_analysis__MLFF.sh` is the block-based parser for VASP MLFF `OUTCAR`s. Writes `evo_*.dat`, peavg summaries, and a quick-look plot.
- `extract_mlff_training_summary.py` complements it by summarizing storage and basis usage directly from `ML_LOGFILE`.
- Use the MLFF path when the run produced an `ML_AB`/`ML_HEAB` (training mode) rather than only a position trajectory.

### OUTCAR utilities

- `outcar_elapsed_stats.py` summarizes `Elapsed time (sec):` lines from many `OUTCAR`s (inspired by `ALCHEMY_timing.sh`, but VASP-OUTCAR-tailored).
- `extract_band_occupations.py` parses the second-last `band No.  band energies  occupation` table (safer than the final occurrence in a truncated `OUTCAR`).

### Analyze many sibling runs

- `source__data_4_analysis__for_all.sh` loops over child directories, optionally refreshes `data_4_analysis.sh`, and runs sequentially or in parallel.
- First place to look when refreshing a whole campaign directory.

### Continue a stalled or walltime-limited run

- `continue_run_ase.py` restarts from a structure pulled from `XDATCAR`, usually a fixed number of steps before the end.
- `RUN_VASP/continue_run.sh` and the `RUN_VASP_MASTER_extended*.sh` family are the older shell-level continuation helpers for multi-part restarts on cluster.
- Prefer `continue_run_ase.py` when the user wants to regenerate `POSCAR` from trajectory data.

### Merge chained runs back together

- `merge_vasp_runs.py` merges `BASEa`, `BASEb`, `BASEc`, ... into `BASE/`.
- Concatenates `XDATCAR`, appends `OUTCAR` after a restart marker, merges `<calculation>` blocks in `vasprun.xml`, and can rerun `data_4_analysis.sh`.

### Estimate pressure-matched volume

- `eos_fit__V_at_P.py` fits a Birch-Murnaghan EOS to analyzed pressure-volume data.
- Used heavily by the TI workflow (`qmd/TI/`) for pressure-matched cell selection and recalculation frame picking.
- Expects `analysis/` files produced by `data_4_analysis.sh`.

### MD analysis (post-trajectory)

- `msd_calc_v3.py` computes per-species mean-squared displacement using MDAnalysis, plots it, and writes diffusion coefficients to a file under `analysis/`.
- `qmd_analysis.ipynb` is a general post-trajectory analysis notebook.
- `speciation_analysis.ipynb`, `speciation_analysis__v2.ipynb`, `speciation_analysis__v3.ipynb` are the speciation-tracking notebooks; prefer `__v3` unless a comparison with earlier runs is needed.

### Visualization

- `plot_vasp_current_snapshot.py` plots a single frame from `CONTCAR`/`POSCAR` or any frame index of an `XDATCAR` (optional `OUTCAR` cell-axes overlay).
- `plot_vasp_step_speed.py` plots sliding-window time-step speed from `OUTCAR` timing lines.
- `make_vasp_snapshot_gif.py` builds GIF animations from sampled `XDATCAR` frames.

## Important Subfolders

- `RUN_VASP/`: live launch, continuation, and multi-run wrappers, including the canonical `LATEST_REFERENCE_FOR_ALL_PU_CLUSTERS/` set (single-run + multi-run + GPU variants).
- `H2_H2O_specific/`: chemistry-specific RDF, histogram, and MSD helpers.
- `setup_retraining_w_R2SCAN/`: VASP-to-MLMD retraining preparation (uses `$ALCHEMY__main__MLDP/post_recal_rerun.py`; env var is set in `sys/myshortcuts.sh`).
- `VASP_POTPAW__POTCARs/`: local PBE VASP POTCAR library to use when POTCAR files need to be assembled or checked.
- `Box_Lars/`: legacy third-party utilities such as `peavg.sh` and `flyv`; treat as external support code, not the main workflow.
- `installation/`: vendor-side build notes and patches.
- `old/`: archival scripts; use only to reproduce historical behavior.

## POTCAR Guidance

- When a task needs VASP pseudopotentials, use the local PBE library in `qmd/vasp/VASP_POTPAW__POTCARs/` as the first reference.
- The default local convention is to prefer `_pv` variants where they are available and appropriate (e.g., `Mg_pv`, `Fe_pv`).
- If no intended `_pv` variant is available or relevant, fall back to the plain element-name directory.
- The plain element-name directories are not the universal default; they are the fallback when a preferred local variant such as `_pv` is not being used.
- Do not fall back from a non-GW choice to a `_GW` potential just because the preferred non-GW variant is missing. For example, if `Mg_pv` is unavailable, do not automatically substitute `Mg_pv_GW`; reassess the intended non-GW fallback such as `Mg`, or confirm with the user.
- Treat `_GW` potentials as a separate family intended for GW or related unoccupied-state calculations, not as generic backups for standard DFT, AIMD, MLFF, or relaxation work.
- Even with that default, confirm the intended potential choice every time before assembling or changing a `POTCAR`.

## Practical Guidance

- Prefer the `LATEST_REFERENCE_FOR_ALL_PU_CLUSTERS` scripts over older cluster wrappers unless the user explicitly wants historical behavior.
- Assume absolute paths and module loads may be stale until checked on the target cluster.
- Preserve marker files (`running_RUN_VASP`, `done_RUN_VASP`, `failed_RUN_VASP`) and logging conventions when patching these scripts; downstream wrappers and `summarize_results.sh` rely on them.
- If assembling a `POTCAR`, follow the species order in the POSCAR and use `VASP_POTPAW__POTCARs/` as the local source of truth unless the user says otherwise.
- If the task involves INCAR, POSCAR, KPOINTS, POTCAR, or parsing VASP outputs, read `references/external_docs.md`.
- Known current caveat for VASP MLFF two-phase setup: hand-merged `ML_AB` files are not yet reliable in this workflow. In the `H2O_NH3/250_H2O__256_NH3` tests, native single-system `ML_AB` worked with `ML_MODE = select`, but merged `ML_AB` files failed during `Scanning ML_AB file` with `forrtl: severe (59): list-directed I/O syntax error`, including a control case merging `H2O + H2O`. Treat stitched-structure setup and `POSCAR`/`POTCAR` consistency as validated, but treat merged-`ML_AB`/`select` as unresolved and requiring a future parser-compatible recipe. See `references/mlff_select_merge_notes.md`.
