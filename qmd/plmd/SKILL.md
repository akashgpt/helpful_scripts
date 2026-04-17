---
name: qmd-plmd
description: Use when working with the local `qmd/plmd` helpers for PLUMED post-processing, especially `COLVAR` plotting or generating `plumed.info` summaries from VASP recalculation directories.
---

# QMD PLMD

This skill is for the lightweight PLUMED helper scripts under `qmd/plmd/`. These scripts are mainly post-processing utilities. They do not define the full PLUMED setup workflow used in ALCHEMY.

## Start Here

Read these first:

- `ZONE_recal_plumed_info.sh`
- `recal_plumed_info.sh`
- `plot_plmd_COLVAR.py`
- `plot_plmd__XRD_V_E_vs_t.py`

If the user is asking about generating `plumed.dat` or choosing CV ranges for ALCHEMY, also check:

- `qmd/ALCHEMY/collect_plumed_cv_parameters.py`
- `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev/reference_input_files/PLUMED_inputs`

## What This Folder Actually Does

- Summarize energy, temperature, pressure, and volume ranges from VASP recalculation outputs into `plumed.info`
- Plot PLUMED `COLVAR` files
- Provide a narrow bridge between VASP recalculation data and later PLUMED configuration choices

## Workflow Map

### Summarize recalculation ranges for one ZONE

- `ZONE_recal_plumed_info.sh` is the main helper.
- Run it from a ZONE directory.
- It scans `*/pre/recal/*/OUTCAR`, extracts the last `TOTEN`, first cell volume, target temperature, and external pressure, then writes a `plumed.info` summary file.
- This is the script to use when CV or bias ranges need to reflect the recalculation data inside one zone.

### Summarize a flat recal directory

- `recal_plumed_info.sh` is the simpler sibling for flatter layouts where recalculation folders are numeric children of the current directory.
- It writes the same kind of `plumed.info`, but assumes a simpler directory structure.

### Plot `COLVAR`

- `plot_plmd_COLVAR.py` is the current general-purpose plotter.
- It strips the `#! FIELDS` header into a temporary `COLVAR.plot`, reads the result with pandas, and plots the first `N` variables after time or all variables if `-n 0` is given.
- It also applies percentile-based y-limits to avoid single outliers destroying the plot scale.

### Plot older special-case `COLVAR` channels

- `plot_plmd__XRD_V_E_vs_t.py` is an older, narrower plotter that expects specific columns such as `xrd`, `vol`, and `energy`.
- Use it only when the local `COLVAR` file really follows that schema.

## Practical Guidance

- These scripts are tightly tied to VASP-style recalculation folders and `OUTCAR` parsing, not generic PLUMED-only trajectories.
- Numeric child-folder assumptions are common; preserve them unless you are intentionally broadening the scripts.
- For ALCHEMY tasks, use this folder for post-processing and use `qmd/ALCHEMY` plus `ALCHEMY__dev` for setup-generation logic.
- For official PLUMED syntax and `COLVAR` semantics, read `references/external_docs.md`.

