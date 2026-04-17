---
name: qmd-vasp
description: Use when working with the local `qmd/vasp` helpers for VASP job submission, continuation, merged multi-part runs, OUTCAR-based analysis, or EOS-guided cell-size estimation in this helpful_scripts repo.
---

# QMD VASP

This skill is for the repo-local VASP helper scripts under `qmd/vasp/`. The folder mixes active workflows, historical variants, and a few specialized subfolders, so start from the live entry points below instead of reading everything.

## Start Here

Read these first, in roughly this order:

- `RUN_VASP/LATEST_REFERENCE_FOR_ALL_PU_CLUSTERS/RUN_VASP__reference.sh`
- `data_4_analysis.sh`
- `source__data_4_analysis__for_all.sh`
- `eos_fit__V_at_P.py`
- `merge_vasp_runs.py`
- `continue_run_ase.py`

Use cluster-specific launchers under `RUN_VASP/TIGER3`, `RUN_VASP/TIGER`, `RUN_VASP/STELLAR`, or `RUN_VASP/FRONTERA` only after checking which cluster the user is actually targeting.

## What This Folder Actually Does

- Submit or template VASP jobs with SLURM wrappers in `RUN_VASP/`.
- Continue long AIMD runs by creating chained directories (`...a`, `...b`, ...).
- Merge chained runs back into one logical dataset.
- Extract time-series data from `OUTCAR` into `analysis/`.
- Estimate the volume or cell size that best matches a target pressure.
- Support a few chemistry-specific analyses and a retraining setup for R2SCAN.

## Workflow Map

### Single run or reference run

- The current reference launcher is `RUN_VASP/LATEST_REFERENCE_FOR_ALL_PU_CLUSTERS/RUN_VASP__reference.sh`.
- It is a hostname-aware SLURM wrapper that chooses module loads and `vasp_std` paths for Princeton and Delta-like systems.
- It writes `log.run_sim` plus marker files such as `running_RUN_VASP` and `done_RUN_VASP`.

### Analyze one finished run

- `data_4_analysis.sh` is the main post-processing script.
- It parses `OUTCAR` into pressure, volume, energy, and temperature time series under `analysis/`.
- It detects a TI-style doubled-energy pattern by comparing `SCALED FREE ENERGIE` and `free energy` counts, then keeps only the non-scaled entries when needed.
- It calls `peavg.sh`, writes `analysis/peavg_summary.out`, and generates a quick-look multi-panel plot.

### Analyze many sibling runs

- `source__data_4_analysis__for_all.sh` loops over child directories, optionally refreshes `data_4_analysis.sh`, and runs the analysis sequentially or in parallel.
- This is the first place to look when the user wants to refresh a whole campaign directory.

### Continue a stalled or walltime-limited run

- `continue_run_ase.py` restarts from a structure pulled from `XDATCAR`, usually some fixed number of steps before the end.
- `RUN_VASP/continue_run.sh` and the `RUN_VASP_MASTER_extended*.sh` family are the older shell-level continuation helpers for multi-part restarts on cluster.
- Prefer `continue_run_ase.py` when the user wants to regenerate `POSCAR` from trajectory data.

### Merge chained runs back together

- `merge_vasp_runs.py` merges `BASEa`, `BASEb`, `BASEc`, ... into `BASE/`.
- It concatenates `XDATCAR`, appends `OUTCAR` after a restart marker, merges `<calculation>` blocks in `vasprun.xml`, and can rerun `data_4_analysis.sh`.
- Use this after long AIMD jobs were split across several submissions.

### Estimate pressure-matched volume

- `eos_fit__V_at_P.py` fits a Birch-Murnaghan EOS to analyzed pressure-volume data.
- It is used heavily by the TI workflow to pick pressure-matched cell sizes and to select frames for high-precision recalculations.
- It expects `analysis/` files produced by `data_4_analysis.sh`.

## Important Subfolders

- `RUN_VASP/`: live launch, continuation, and multi-run wrappers.
- `H2_H2O_specific/`: chemistry-specific RDF, histogram, and MSD helpers.
- `setup_retraining_w_R2SCAN/`: VASP-to-MLMD retraining preparation.
- `VASP_POTPAW__POTCARs/`: local PBE VASP POTCAR library to use when POTCAR files need to be assembled or checked.
- `Box_Lars/`: legacy third-party utilities such as `peavg.sh` and `flyv`; treat as external support code, not the main workflow.
- `old/`: archival scripts; use only to reproduce historical behavior.

## POTCAR Guidance

- When a task needs VASP pseudopotentials, use the local PBE library in `qmd/vasp/VASP_POTPAW__POTCARs/` as the first reference.
- The default local convention is to prefer `_pv` variants where they are available and appropriate, for example `Mg_pv` and `Fe_pv`.
- If no intended `_pv` variant is available or relevant, then fall back to the plain element-name directory.
- This means the plain element-name directories are not the universal default; they are the fallback when a preferred local variant such as `_pv` is not being used.
- Do not fall back from a non-GW choice to a `_GW` potential just because the preferred non-GW variant is missing. For example, if `Mg_pv` is unavailable, do not automatically substitute `Mg_pv_GW`; instead, reassess the intended non-GW fallback such as `Mg`, or confirm the choice with the user.
- Treat `_GW` potentials as a separate family intended for GW or related unoccupied-state calculations, not as generic backups for standard DFT, AIMD, MLFF, or relaxation work.
- Even with that default, confirm the intended potential choice every time before assembling or changing a `POTCAR`, because some systems really do require the suffixed variants.

## Practical Guidance

- Prefer the `LATEST_REFERENCE_FOR_ALL_PU_CLUSTERS` scripts over older cluster wrappers unless the user explicitly wants historical behavior.
- Assume absolute paths and module loads may be stale until checked on the target cluster.
- Preserve marker files and logging conventions when patching these scripts; downstream wrappers rely on them.
- If assembling a `POTCAR`, follow the species order in the POSCAR and use `VASP_POTPAW__POTCARs/` as the local source of truth unless the user says otherwise.
- If the task involves INCAR, POSCAR, KPOINTS, POTCAR, or parsing VASP outputs, read `references/external_docs.md`.
