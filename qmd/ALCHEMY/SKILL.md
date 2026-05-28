---
name: qmd-alchemy
description: Use when working with the local `qmd/ALCHEMY` helpers for ALCHEMY or DPAL campaigns, especially performance tracking, recal-directory cleanup and refinement, PLUMED CV collection, dp_test analysis, all-frame recal generation, job-management, or environment setup around the main `ALCHEMY__dev` pipeline.
---

# QMD ALCHEMY

This skill is for the helper scripts under `qmd/ALCHEMY/`, but the authoritative pipeline definition is not here. For real ALCHEMY workflow questions, check the local priority repos in this order:

- `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev` (active development; `$ALCHEMY__dev`)
- `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__in_use` (stable; `$ALCHEMY__main`)
- this helper folder in `qmd/ALCHEMY`

Treat `qmd/ALCHEMY/` as the utility belt around the main pipeline, not the canonical implementation of the pipeline itself.

## Start Here

Read these first:

- `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev/README.md`
- `collect_plumed_cv_parameters.py`
- `track_performance.sh`
- `refine_deepmd.sh`
- `recal_update.sh`
- `count.sh`
- `compute_filtered_rmse.py`
- `create_all_frame_recal_from_lammps_dump.py`

## What Lives Here

- Campaign summary helpers: `count.sh`, `track_performance.sh`
- Recal-directory maintenance: `recal_update.sh`, `reset_listing.sh`, `reset_listings_in_all_recal_dirs.sh`
- DeePMD / refinement helpers: `refine_deepmd.sh`, `compute_filtered_rmse.py`
- All-frame recal builder: `create_all_frame_recal_from_lammps_dump.py`
- PLUMED CV-range extraction: `collect_plumed_cv_parameters.py`
- Job / process management: `kill_all_scripts.sh`, `kill_PID_ongoing_scripts.sh`
- Conda environment bootstrap scripts under `conda_env/` (separate TF and PT installers — see below)
- `mldp_ALCHEMY` symlink → `ALCHEMY__dev/TRAIN_MLMD_scripts/ANALYSIS/mldp/ALCHEMY` (the ALCHEMY-helper layer of mldp; the rest of mldp lives one level up at `$ALCHEMY__dev__MLDP` / `$ALCHEMY__main__MLDP`)
- Historical duplicates under `misc/` and dated backup folders under `backup/`

## How This Relates To ALCHEMY__dev

The main ALCHEMY workflow is:

1. Train DeePMD (TF/Horovod or PT backend; see `ALCHEMY__dev/TRAIN_MLMD_scripts/`)
2. Run LAMMPS MD
3. Select informative frames
4. Recalculate with VASP and feed back into training

Recent (~2026-05) refactor of `ALCHEMY__dev/TRAIN_MLMD_scripts/TRAIN_MLMD_LEVEL_2.sh` introduced a restart-chain training driver with new sentinel files (`FREEZE_COMPRESS_DONE`, `TRAINING_COMPLETE`, `HEALTH_GATE.tsv`), env-overridable health-gate thresholds, and explicit support for both TF and PT chain-aware training scripts (`train_1h.apptr.Ngpu.{PT,TF}{,.restart}.sh`). When debugging training stability, expect those sentinels and `HEALTH_GATE.tsv` rows in completed-iteration directories.

This helper folder supports the edges of that loop — it does not implement the loop itself.

## Practical Workflow Map

### Understand the campaign structure

- Read `ALCHEMY__dev/README.md` first; it explains the central parameter file, zone layout, and how VASP, DeePMD, LAMMPS, ASAP, and optional PLUMED fit together.

### Track campaign progress

- `track_performance.sh` walks iteration directories (`v8_i1`, `v8_i2`, ...).
- It runs `count.sh` in each finished iteration and collects selected-frame fractions and average RMSE values into `log.track_progress`.
- Use this for a campaign-wide status snapshot.

### Filtered RMSE evaluation from dp_test

- `compute_filtered_rmse.py` reads per-frame `dp_test.e_peratom.out`, `dp_test.f.out`, and `dp_test.v_peratom.out` across all system subdirectories.
- Excludes frames with `|dE/atom| > energy_upper_cutoff` from all E/F/V metrics (matches the filtering used by `analysis_v3.py` during frame selection).
- This replaces the older approach of reading aggregate RMSE from `log.dp_test`; use it when you need consistency with the active-learning frame-selection cutoffs.

### Refresh or repair recal directories

- `recal_update.sh` is the cleanup and re-extraction helper for `recal/` or `old_recal/` trees.
- It can rebuild DeePMD extraction outputs and, for older campaign patterns, rerun analysis tools on recovered recal directories.
- `reset_listing.sh` renumbers digit-prefixed subfolders sequentially from `1`; useful after manually deleting a few.
- `reset_listings_in_all_recal_dirs.sh` walks every `recal/` and runs `reset_listing.sh` in each.

### Refine frame selection after dp test

- `refine_deepmd.sh` loops over every `recal/` directory, rebuilds DeePMD data, runs `dp test`, applies error-threshold analysis, and re-extracts only the chosen frame IDs.
- Contains hardcoded paths and aggressive cleanup — inspect carefully before running on a new campaign.

### Build a full all-frame recal from a LAMMPS dump

- `create_all_frame_recal_from_lammps_dump.py` is the alternative to the standard FPS/ASAP selective frame picker.
- It mirrors the relevant part of ALCHEMY Level 4: runs ASAP on `npt.dump`, generates an index file containing **every** frame, runs `extract_deepmd.py` inside `pre/`, and reconciles `deepmd/type_map.raw` against `conf.lmp` atom-type labels.
- Use when you want to recalculate every frame (e.g., for full-trajectory validation), not the FPS-selected subset.

### Build PLUMED CV parameter blocks

- `collect_plumed_cv_parameters.py` reads `md/ZONE_*/plumed.info` files across one or more iterations and prints paste-ready `PLUMED_CV_PARAMETERS` blocks.
- Use this to seed or update PLUMED bounds in `TRAIN_MLMD_parameters.txt`.

### Stop or clean up running jobs

- `kill_all_scripts.sh` kills processes matching specific training-script names and cancels pending SLURM jobs for the current user. Use on a node where a training campaign needs to be force-stopped.
- `kill_PID_ongoing_scripts.sh` terminates only the PIDs listed in a file named `PID_ongoing_scripts` in the current directory — safer when you want surgical kill of a specific known set.

### Install or reproduce environments

- `conda_env/` has separate installers for the TF and PT DeePMD backends:
  - `install_deepmd-kit_w_plumed_lmp_and_others___for_ALCHEMY.sh` and `install_deepmd-kit_w_plumed_lmp_soap_and_others___for_ALCHEMY.sh` — TF backend variants (with/without SOAP).
  - `install_deepmd-kit_PT_w_plumed_lmp_and_others___for_ALCHEMY.sh` and `install_deepmd-kit_PT_w_plumed_lmp_soap_and_others___for_ALCHEMY.sh` — PT backend variants (with/without SOAP).
  - `install_deepmd_model_conversion_bw_PT_and_TF___for_ALCHEMY.sh` — for cross-backend model conversion.
- The active conda envs landed by these scripts are `ALCHEMY_env` (TF) and `ALCHEMY_env__PT` (PT). PT-only features such as `gradient_max_norm` and `warmup_steps` are reachable only via the PT env.
- Treat these scripts as recipes, not guaranteed-current installers — re-test pinned versions before reusing on a new system.

## Practical Guidance

- Prefer the top-level scripts over duplicates in `misc/`.
- Assume many absolute paths and conda env names are campaign-specific until verified.
- When the user asks about the core ALCHEMY algorithm, directory format, or parameter semantics, answer from `ALCHEMY__dev/README.md` first.
- When the task involves DeePMD, LAMMPS, or PLUMED behavior, read `references/external_docs.md`.
- For backend-specific behavior (TF vs PT support for restart, clipping, warmup), grep `ALCHEMY_env*/lib/python*/site-packages/deepmd/{tf,pt}` rather than assuming feature parity between backends.
