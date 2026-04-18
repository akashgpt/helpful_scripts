---
name: qmd-setup-input
description: Use when working with the local `qmd/setup_INPUT` utilities for generating initial structures, converting between VASP and LAMMPS inputs, joining slabs or phases along z, or building composition-series folders for QMD and MLMD campaigns.
---

# QMD Setup Input

This skill is for the structure-preparation toolbox under `qmd/setup_INPUT/`. The scripts here are a mix of generic ASE-based utilities and chemistry-specific generators that encode local project assumptions.

## Start Here

Read these first:

- `initialize_structure_ASE.py`
- `join_conf_lmps.py`
- `join_POSCARS.py`
- `vasp_to_lammps_input.py`
- `lammps_to_vasp_input.py`
- `generate_composition_series.py`
- `generate_selected_composition_structures.py`
- `organize_selected_compositions.py`

Then inspect any chemistry-specific initializer that matches the user request, such as:

- `initialize_structure_ASE_H2.py`
- `initialize_structure_ASE_H2O.py`
- `initialize_structure_ASE_H2S.py`
- `initialize_structure_ASE_NH3.py`
- `initialize_structure_ASE_Ice_X.py`
- `initialize_structure_ASE_MgSiO3_bridgmanite.py`

## What This Folder Actually Does

- Create random-packed starting structures with ASE
- Convert between VASP `POSCAR` and LAMMPS `conf.lmp`
- Join two cells along the z axis
- Build composition-series CSVs and selected-composition folders
- Support a few system-specific setup workflows used in planetary-material campaigns

## Workflow Map

### Generate a fresh random-packed cell

- `initialize_structure_ASE.py` is the generic starting point.
- It creates a periodic ASE structure by random packing with a minimum-distance criterion under PBC, then writes both `POSCAR` and `conf.lmp`.
- Use a chemistry-specific initializer when one exists, because those usually encode composition and cell choices already tuned for a project.

### Convert between VASP and LAMMPS

- `vasp_to_lammps_input.py` reads a `POSCAR`, wraps atoms into the unit cell, and writes a LAMMPS data file plus a wrapped `POSCAR_ase`.
- `lammps_to_vasp_input.py` does the reverse for a `conf.lmp`.
- These are simple and useful, but check the hardcoded species order before using them on a new chemistry.

### Join two structures along z

- `join_conf_lmps.py` stitches two `conf.lmp` files along the z axis with an optional gap and explicit output species order.
- `join_POSCARS.py` provides the same logic for VASP POSCAR-like files.
- `stitch_two_phase_vasp.py` is the more flexible VASP helper when live `CONTCAR` files include MD restart trailers or when the two source cells need a deliberate in-plane reconciliation policy such as `average` or `max`.
- Prefer these over ad hoc manual editing when building two-phase or slab-style starting structures.

### Build composition series

- `generate_composition_series.py` creates a CSV describing a discrete composition path, currently geared toward constant-total-atom substitutions such as He for MgSiO3 formula units.
- `generate_selected_composition_structures.py` uses a base POSCAR plus a selection log to create actual structures for chosen compositions.
- `organize_selected_compositions.py` turns those flat generated files into named subfolders with `POSCAR` and `conf.lmp`.

## Practical Guidance

- Many scripts here are reusable, but several are still chemistry-specific despite having generic-looking names.
- Preserve species-order handling carefully when editing conversion or joining scripts.
- Use the ASE-based joiners and converters instead of hand-editing coordinates when reproducibility matters.
- If the task depends on ASE or file-format semantics, read `references/external_docs.md`.
