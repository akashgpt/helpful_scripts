---
name: qmd-ti
description: Use when working with the local `qmd/TI` thermodynamic-integration workflow for `SCALEE_*` runs, Gibbs-Helmholtz follow-up analysis, isobar extensions, or KD fitting from Fe and MgSiO3 datasets.
---

# QMD TI

This skill is for the repo-specific thermodynamic integration workflow under `qmd/TI/`. The folder contains many near-duplicate wrappers, but a fairly clear backbone exists once you focus on the workhorse scripts.

## Start Here

Read these first:

- `calculate_GFE_v2.sh`
- `create_KP1.sh`
- `create_KP1x_hp_calc_eos_SCALEE.sh`
- `source_all_SCALEE.sh`
- `Ghp_analysis.py`
- `isobar_Ghp_analysis.py`
- `estimate_KD_generic_v8.py`

Prefer the fuller and newer scripts over older numbered variants unless the user explicitly wants to reproduce an older campaign.

## Expected Directory Layout

Most wrappers assume a `P*_T*` directory containing:

- `master_setup_TI/`
- one or more configuration directories
- inside each configuration, `setup_TI/POSCAR_NPT`

The setup directory is expected to contain files such as:

- `POTCAR`
- `KPOINTS_111`
- `KPOINTS_222`
- `INCAR_NPT`
- `INCAR_SCALEE`
- `INCAR_SPC`
- `RUN_VASP_NPT.sh`
- `RUN_VASP_NVT.sh`
- `RUN_VASP_SCALEE.sh`
- `RUN_VASP_SPC.sh`
- `input.calculate_GFE`

`calculate_GFE_v2.sh` validates this layout before doing real work.

## Workflow Map

### Stage 1: low-cost NPT run and pressure-matched cell estimate

- `calculate_GFE_v2.sh` is the main engine.
- It reads `master_setup_TI/input.calculate_GFE`, checks that the enclosing `P*_T*` folder name matches the chosen pressure and temperature, and launches a low-k-point `V_est/KP1` simulation.
- After KP1 finishes, it copies in `qmd/vasp/data_4_analysis.sh`, runs the analysis, and calls `qmd/vasp/eos_fit__V_at_P.py` to estimate a target cell size.

### Stage 2: create SCALEE and high-precision branches

- The same script then creates the `SCALEE_*` family and, depending on mode, higher-precision recalculation folders.
- Naming is meaningful here:
  - `KP1` means the first low-cost pressure-finding run.
  - `KP1x` indicates follow-up volume-bracketed runs.
  - `hp_calc` means higher-precision single-point or sparse-frame calculations.
  - `eos` means EOS-guided cell-size refinement is included.

### Stage 3: batch creation wrappers

- `create_KP1*.sh` scripts are campaign-level wrappers that fan out the setup across many configuration folders.
- `isobar__create*.sh` wrappers do the same for `isobar_calc` trees.
- `run_isobar__create__.sh` is the top-level driver that walks many `isobar_calc` directories and launches the chosen isobar helper in each.

### Stage 4: re-analysis and triage of finished SCALEE runs

- `source_all_SCALEE.sh` reruns `data_4_analysis.sh` across matching `SCALEE_*` directories.
- It then checks `analysis/peavg.out`, counts completed timesteps, and drops marker files like `to_RUN__failed` or `to_RUN__1000_to_5000`.
- It can also be used as a restart-extension triage tool for under-run trajectories.

### Stage 5: thermodynamic integration over SCALEE

- `Ghp_analysis.py` is the main TI post-processing script.
- It reads the fixed `SCALEE` quadrature grid, parses `analysis/peavg.out`, estimates ideal-gas and mixing terms, and writes TI-derived outputs such as `TI_analysis.csv` and `log.Ghp_analysis`.
- This is the place to study if the user wants the physical meaning of `G_hp`, `TS`, or how the quadrature weights are applied.

### Stage 6: Gibbs-Helmholtz extension along an isobar

- `isobar_Ghp_analysis.py` extends the primary TI result using additional constant-pressure temperature points in `isobar_calc/`.
- It combines the primary `SCALEE_1` enthalpy and `log.Ghp_analysis` result with secondary `T*/SCALEE_0/analysis/peavg.out` data.
- Use this when the user wants temperature interpolation or extrapolation at fixed pressure after the main TI run is done.

### Stage 7: KD estimation and fitting

- `estimate_KD_generic_v8.py` is the most readable current KD estimator and should be the default unless a legacy version is required.
- It expects sibling trees like `Fe_<species>/` and `MgSiO3_<species>/`, parses `log.Ghp_analysis`, combines TI and GH results, and outputs KD and D_wt tables plus plots.
- `fit_KD_PTX*.py` scripts are the next layer for smooth P-T-X fitting once the per-configuration free-energy data exist.

## Practical Guidance

- The folder is wrapper-heavy. If you are deciding what to patch, patch the workhorse first, then only the wrappers that call it.
- Do not casually rename folders like `SCALEE_0`, `SCALEE_1`, `V_est`, `KP1`, or `isobar_calc`; the scripts infer workflow stage from those names.
- Many scripts assume Slurm commands such as `sbatch`, `squeue`, and `scontrol`, plus env vars like `$HELP_SCRIPTS_TI`, `$LOCAL_HELP_SCRIPTS_TI`, and `$HELP_SCRIPTS_vasp`.
- For the physics, prefer the detailed comments in `estimate_KD_generic_v8.py` and `Ghp_analysis.py` over generic internet tutorials, because this workflow is highly customized.
- For the VASP and ASE file conventions these scripts depend on, read `references/external_docs.md`.

