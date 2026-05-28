# sys

## Overview

This folder contains shell configuration snippets, cluster setup references, and system-level diagnostic helpers.

Use these scripts for environment setup, cluster health checks, disk tests, Slurm/QOS inspection, and shell shortcut maintenance.

## Root Files

### `AGENTS.md`

Local instructions for coding agents working in this folder.

### `CLAUDE.md`

Local instructions or notes for Claude-style coding sessions.

### `NCSA_DELTA.bashrc`

Reference bash configuration for NCSA Delta.

### `NCSA_DELTA.condarc`

Reference Conda configuration for NCSA Delta.

## Scripts

### `benchmark_cpu_flops__slurm.sh`

Slurm script for running a CPU FLOPS benchmark.

### `check_slurm_health.sh`

Checks Slurm state and reports cluster scheduling health indicators.

### `creation_time.sh`

Reports or inspects file creation-time style metadata where available.

### `grab_qos.py`

Extracts or summarizes Slurm QOS information.

### `myshortcuts.sh`

Shell shortcut collection for common interactive commands.

### `run_vasp_cpu_benchmark__NCSA_DELTA.sh`

Runs a VASP CPU benchmark on NCSA Delta.

### `test_disk_speed.sh`

Tests disk read/write speed for a target filesystem.

## Configuration Collections

### `collections__bashrc/`

Cluster-specific bash configuration references:

- `DELLA.bashrc`
- `NCSA_DELTA.bashrc`
- `PU_CLUSTERS.bashrc`
- `STELLAR.bashrc`
- `TIGER.bashrc`

### `collections__condarc/`

Cluster-specific Conda configuration references:

- `DELLA.condarc`
- `DELTA.condarc`
- `NCSA_DELTA.condarc`
- `STELLAR.condarc`
- `TIGER.condarc`
- `TIGER3.condarc`

