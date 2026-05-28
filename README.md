# helpful_scripts

## Overview

This repository collects small utilities, workflow helpers, benchmark references, and cluster setup notes used across QMD, ALCHEMY, VASP, DeePMD, LAMMPS, PLUMED, and general HPC work.

Start here when looking for a reusable script or a known-good cluster reference. The main folders are organized by purpose, and each user-facing folder has its own README with more detail.

## Main Folders

### `qmd/`

QMD workflow helpers for ALCHEMY active learning, VASP, LAMMPS, PLUMED, thermodynamic integration, and input-structure preparation.

Use this folder for scientific workflow scripts and format converters.

### `benchmarks/`

Curated benchmark notes, compact result summaries, and known-good submission templates.

Use this before creating new VASP, DeePMD, LAMMPS, or cluster benchmark scripts.

### `ONGOING/`

Crash-resilient tracking notes for submitted jobs and long-running workflows.

Use this to recover the state of work that may still be running or waiting for follow-up.

### `sys/`

Cluster shell setup, Conda config references, Slurm diagnostics, and system-level helper scripts.

Use this for environment and machine-facing utilities.

### `general/`

Small general-purpose scripts that do not belong to a specific scientific workflow.

### `old/`

Older scripts retained for reference. Check active folders first before copying from here.

## Root Files

### `AGENTS.md`

Repository-specific instructions for coding agents working in this helper repo.

### `CLAUDE.md`

Repository-specific notes for Claude-style coding sessions.

### `REVIEW_GUIDE.md`

Review checklist and guidance for inspecting changes in this repo.

### `helpful_commands.txt`

Loose collection of useful commands.

### `LICENSE`

Repository license.

## Generated Or Tooling Folders

The following folders support editors, agents, or git state and are not normal script collections:

- `.agents/`
- `.claude/`
- `.codex/`
- `.git/`
- `.vscode/`

Generated files such as `.DS_Store` and `__pycache__/` should not be treated as source material.

