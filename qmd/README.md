# qmd

## Overview

This folder contains helper scripts for quantum molecular dynamics workflows: ALCHEMY active learning, VASP, LAMMPS, PLUMED, thermodynamic integration, and input-structure preparation.

For ALCHEMY work, treat the main ALCHEMY repository as the canonical pipeline. The scripts here are utility and reference helpers.

## Top-Level Scripts

### `RECAL_PHASE_LOCAL.sh`

Runs a local recalculation phase helper for QMD active-learning workflows.

### `RECAL_PHASE_MASTER.sh`

Coordinates recalculation phase work from a higher-level driver.

### `add_systems_2.sh`

Adds or prepares additional systems for a QMD workflow.

### `cancel_jobs.sh`

Cancels queued or running jobs associated with a workflow.

### `check_type_maps.py`

Checks consistency of type maps across DeePMD or related QMD inputs.

### `create_ASAP_index_file.py`

Creates an ASAP index file for analysis or visualization workflows.

### `create_deepmd_collection.py`

Collects or organizes DeepMD training data into a reusable collection.

### `plot_ttr_convergence.py`

Plots convergence behavior for TTR-style calculations.

### `setup_frame_convergence.sh`

Sets up frame-convergence tests for trajectory or training-data studies.

### `update_recal_dir.sh`

Updates recalculation directories after new data or outputs are available.

## Subfolders

### `ALCHEMY/`

Utilities around the ALCHEMY active-learning workflow. Scripts include RMSE filtering, PLUMED CV collection, frame extraction from LAMMPS dumps, performance tracking, listing resets, process cleanup, and old refinement helpers.

Important scripts include:

- `collect_plumed_cv_parameters.py`
- `compute_filtered_rmse.py`
- `count.sh`
- `create_all_frame_recal_from_lammps_dump.py`
- `kill_PID_ongoing_scripts.sh`
- `kill_all_scripts.sh`
- `recal_update.sh`
- `refine_deepmd.sh`
- `reset_listing.sh`
- `reset_listings_in_all_recal_dirs.sh`
- `test.sh`
- `track_performance.sh`

### `TI/`

Thermodynamic-integration utilities. Scripts cover Gibbs free energy calculations, KD estimation, isobar setup, SCALEE run creation, backup/archive helpers, and postprocessing.

Important scripts include:

- `Ghp_analysis.py`
- `calc_KD_from_2phase_data.py`
- `calculate_GFE.sh`
- `calculate_GFE_v2.sh`
- `correct_SCALEE_value_inconsistencies.sh`
- `create_SCALEE_runs.sh`
- `estimate_KD.py`
- `estimate_KD_2.py`
- `estimate_KD_intersections.py`
- `fit_KD_PTX.py`
- `fit_KD_PTX_v2.py`
- `isobar_*`
- `master_continue_SCALEE_7.sh`
- `run_isobar__create__.sh`
- `source_all_SCALEE.sh`
- `tar_*`
- `un_tar_all.sh`

### `lammps/`

LAMMPS visualization helpers.

- `make_lammps_snapshot_gif.py`
- `plot_lammps_current_snapshot.py`

### `plmd/`

PLUMED and COLVAR helpers for recalculation and trajectory analysis.

- `ZONE_recal_plumed_info.sh`
- `plot_plmd_COLVAR.py`
- `plot_plmd__XRD_V_E_vs_t.py`
- `recal_plumed_info.sh`

### `setup_INPUT/`

Input-structure generation and conversion helpers. Scripts create composition series, stitch structures, convert between VASP/LAMMPS/XYZ formats, and organize selected compositions.

Important scripts include:

- `generate_composition_series.py`
- `generate_selected_composition_structures.py`
- `generate_selected_composition_structures__He_MgSiO3.py`
- `initialize_structure_ASE.py`
- `initialize_structure_ASE_v2.py`
- `initialize_structure_ASE_v3.py`
- `join_POSCARS.py`
- `join_conf_lmps.py`
- `lammps_to_vasp_input.py`
- `organize_selected_compositions.py`
- `remove_negative_values_in_POSCAR.py`
- `stitch_two_phase_vasp.py`
- `vasp_to_lammps_input.py`
- `vasp_xdatcar_to_lammps_dump.py`
- `vasp_xdatcar_to_xyz.py`

### `vasp/`

VASP setup, continuation, analysis, and visualization helpers. Scripts cover run continuation, timing analysis, MLFF summary extraction, OUTCAR timing, MSD calculation, snapshot plotting, and conversion/postprocessing utilities.

Important scripts include:

- `continue_run_ase.py`
- `data_4_analysis.sh`
- `data_4_analysis_v2.sh`
- `data_4_analysis_v3.sh`
- `eos_fit__V_at_P.py`
- `extract_band_occupations.py`
- `extract_mlff_training_summary.py`
- `make_vasp_snapshot_gif.py`
- `merge_vasp_runs.py`
- `msd_calc_v3.py`
- `outcar_elapsed_stats.py`
- `plot_vasp_current_snapshot.py`
- `plot_vasp_step_speed.py`
- `source__data_4_analysis__for_all.sh`

## Guides

### `SKILL.md`

Local onboarding guide for QMD tasks in this helper repository.

Each major QMD subfolder may also contain its own `SKILL.md` with more specific workflow notes.

