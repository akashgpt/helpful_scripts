---
name: qmd-alchemy
description: Use when working with the local `qmd/ALCHEMY` helpers for ALCHEMY or DPAL campaigns, especially performance tracking, recal-directory cleanup and refinement, PLUMED CV collection, or environment setup around the main `ALCHEMY__dev` pipeline.
---

# QMD ALCHEMY

This skill is for the helper scripts under `qmd/ALCHEMY/`, but the authoritative pipeline definition is not here. For real ALCHEMY workflow questions, check the local priority repos in this order:

- `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev`
- `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__in_use`
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

## What Lives Here

- Campaign summary helpers such as `count.sh` and `track_performance.sh`
- Recal-directory maintenance scripts such as `recal_update.sh`
- DeePMD and recalibration refinement helpers such as `refine_deepmd.sh`
- PLUMED CV-range extraction with `collect_plumed_cv_parameters.py`
- One-off environment installation scripts under `conda_env/`
- Historical duplicates under `misc/` and backups

## How This Relates To ALCHEMY__dev

The main ALCHEMY workflow is:

1. Train DeePMD
2. Run LAMMPS MD
3. Select informative frames
4. Recalculate them with VASP

That train-MD-select-recal loop, the directory layout, and the parameter file meanings are documented best in `ALCHEMY__dev/README.md`.

This helper folder mainly supports the edges of that loop:

- summarizing convergence and selected-frame fractions
- cleaning or re-extracting recal directories
- refining DeePMD frame selections after test results exist
- harvesting PLUMED CV bounds from previous iterations
- installing environments with DeePMD, PLUMED, and LAMMPS support

## Practical Workflow Map

### Understand the campaign structure

- Read `ALCHEMY__dev/README.md` first.
- It explains the central parameter file, zone layout, and how VASP, DeePMD, LAMMPS, ASAP, and optional PLUMED fit together.

### Track campaign progress

- `track_performance.sh` walks iteration directories like `v8_i1`, `v8_i2`, ...
- It runs `count.sh` in each finished iteration and collects selected-frame fractions and average RMSE values into `log.track_progress`.
- Use this when the user wants a campaign-wide status snapshot rather than raw per-iteration logs.

### Refresh or repair recal directories

- `recal_update.sh` is a cleanup and re-extraction helper for `recal` or `old_recal` trees.
- It can rebuild DeePMD extraction outputs and, for older campaign patterns, rerun analysis tools on recovered recal directories.

### Refine frame selection after dp test

- `refine_deepmd.sh` loops over every `recal` directory, rebuilds DeePMD data, runs `dp test`, applies error-threshold analysis, and re-extracts only the chosen frame IDs.
- This script contains hardcoded paths and aggressive cleanup, so inspect it carefully before running it on a new campaign.

### Build PLUMED CV parameter blocks

- `collect_plumed_cv_parameters.py` reads `md/ZONE_*/plumed.info` files across one or more iterations and prints paste-ready `PLUMED_CV_PARAMETERS` blocks.
- Use this when the user wants to seed or update PLUMED bounds in `TRAIN_MLMD_parameters.txt`.

### Install or reproduce environments

- `conda_env/` contains environment bootstrap scripts for DeePMD, PLUMED, LAMMPS, and conversion utilities.
- Treat them as historical recipes rather than guaranteed current installers.

## Practical Guidance

- Prefer the top-level scripts over duplicates in `misc/`.
- Assume many absolute paths and environment names are campaign-specific until verified.
- When the user asks about the core ALCHEMY algorithm, directory format, or parameter semantics, answer from `ALCHEMY__dev/README.md` first.
- When the task involves DeePMD, LAMMPS, or PLUMED behavior, read `references/external_docs.md`.

