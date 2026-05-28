# ONGOING

## Overview

This folder tracks jobs and workflows that were submitted but may still need follow-up. These notes are meant to make long-running work recoverable after a shell crash, IDE restart, or context reset.

Each note should say where the job was launched, what was submitted, how to check it, and what to do next. Do not treat this folder as an archive of completed work; once a workflow is confirmed finished, move or remove the note only after explicit confirmation.

## Cluster Folders

### `ALCF_POLARIS/`

Reserved for ALCF Polaris workflows. (No current notes — directory is kept so notes can be added without re-creating it.)

### `DELLA/`

Notes for active or recently active Princeton Della workflows. Current notes include NH3/H2 DeePMD restart-chain tests, 4-GPU seed stability runs, PT restart diagnostics, TF decay tests, LAMMPS validation, and ML_v4 active-learning jobs.

### `NCSA_DELTA/`

Notes for active or recently active NCSA Delta workflows. Current notes include He/MgSiO3 DeePMD training, checkpoint validation, H200 retraining, and MgSiO3 R2SCAN recalculation jobs.

### `STELLAR/`

Notes for active or recently active Stellar workflows. Current notes include NH3/H2 VASP and NH3/MgSiO3 ML_v4 GPU work.

### `TIGER/`

Notes for active or recently active Tiger workflows. Current notes include MgSiOFe R2SCAN VASP work moved to Tiger.

## File Naming

Use names like:

```text
<system>__<purpose>__<partition_or_resource>__YYYYMMDD.md
```

This keeps status notes sortable and makes it clear which system and job family each note belongs to.

## What Each Note Should Contain

- Working directory
- Submission command and job ID
- Cluster, partition, node/GPU/CPU request, and important runtime settings
- Key input files or scripts
- How to check status
- How to resume or clean up
- Current status snapshot with date/time
- Next step once the job finishes
