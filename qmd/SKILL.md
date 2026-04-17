---
name: qmd-router
description: Use when a task touches the local `qmd/` toolbox in helpful_scripts and you need to route quickly to the right sub-skill, such as `vasp`, `TI`, `ALCHEMY`, `plmd`, or `setup_INPUT`.
---

# QMD Router

This skill is the entry point for work under `qmd/` in this repository. Start here when the task is somewhere inside `qmd/`, then immediately read the more specific sub-skill for the folder that matches the job.

## Folder Map

- `vasp/`
  - Read `vasp/SKILL.md`
  - Use for VASP submission, continuation, multi-part run merging, `OUTCAR` analysis, and EOS-guided cell-size estimation.

- `TI/`
  - Read `TI/SKILL.md`
  - Use for the thermodynamic-integration workflow built around `SCALEE_*`, `Ghp_analysis.py`, isobar extensions, and KD estimation.

- `ALCHEMY/`
  - Read `ALCHEMY/SKILL.md`
  - Use for helper utilities around active-learning, DeePMD, LAMMPS, PLUMED, recal maintenance, and performance tracking.
  - For authoritative ALCHEMY pipeline behavior, also check `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev`.

- `plmd/`
  - Read `plmd/SKILL.md`
  - Use for PLUMED `COLVAR` plotting and generating `plumed.info` summaries from recalculation directories.

- `setup_INPUT/`
  - Read `setup_INPUT/SKILL.md`
  - Use for structure generation, joining slabs or phases, and VASP-LAMMPS conversions.

## Multi-Folder Tasks

Some tasks naturally span more than one subfolder. Common combinations are:

- `TI` plus `vasp`
  - TI scripts depend heavily on VASP run outputs and `data_4_analysis.sh`.

- `ALCHEMY` plus `plmd`
  - PLUMED post-processing here often supports ALCHEMY workflows.

- `setup_INPUT` plus `vasp`
  - Initial structures often feed directly into VASP or later VASP-to-LAMMPS conversions.

If a task spans folders, read both relevant sub-skills before making changes.

## Reading Order

For any `qmd/` task:

1. Read this file.
2. Read the specific subfolder `SKILL.md`.
3. Read that sub-skill's `references/` file only if the task depends on external software semantics.
4. For ALCHEMY or DPAL tasks, follow the repository lookup policy in `AGENTS.md` and check `ALCHEMY__dev` first.

## Practical Guidance

- Prefer the local folder skill over guessing from filenames alone.
- Treat old or backup scripts as historical unless the task is explicitly about reproducing legacy behavior.
- Keep the main workflow logic in the local repo as the source of truth, and use external docs only to verify software syntax or file-format semantics.

