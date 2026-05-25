# ONGOING — MgSiO3 R2SCAN recalculation (NCSA Delta, gpuH200x8)

**Started tracking:** 2026-05-18
**Cluster:** NCSA Delta · partition `gpuH200x8` (H200, 141 GB/GPU) · account `bguf-delta-gpu`
**Working dir:** `/work/nvme/bguf/akashgpt/qmd_data/MgSiO3/sim_data`
**Status as of 2026-05-23:** all **61/61 DONE** · 0 RUNNING · 0 FAILED · DeePMD data extracted from each `n*/OUTCAR` into `n*/deepmd/set.000/` (6,100 R2SCAN frames total) using `${ALCHEMY__main__MLDP}/extract_deepmd.py -d deepmd -ttr 10000 -e Mg Si O H He`. Type_map enforced as `[Mg, Si, O, H, He]` to match the trained `se_e2_a` models (5 types per `sel=[50,50,140,700,500]`). This collection is now the canonical held-out test set for He_MgSiO3 / MgSiOH `se_e2_a` validation; see [[heldout-test-set-mgsio3-sim-data]] memory.

**Path update 2026-05-23:** These VASP simulations were originally
tracked under `/work/nvme/bguf/akashgpt/qmd_data/NH3_MgSiO3/sim_data`,
but that campaign folder has since been renamed/moved to
`/work/nvme/bguf/akashgpt/qmd_data/MgSiO3/sim_data`. Use the `MgSiO3`
path for all checks and resume commands.

---

## What is running

61 short R2SCAN AIMD restarts, one per `n*` directory (`n0201 … n0276`).
All systems are pure **Mg–Si–O** (100 / 150 / 200 atoms). These were
initially staged under an `NH3_MgSiO3` parent name, but there is no N/H in
these cells, and the active parent folder is now `MgSiO3`.

Per-directory setup that was applied:

- **POSCAR** ← that dir's **CONTCAR** (restart from the evolved structure).
- **INCAR** = He_MgSiO3 GPU R2SCAN template with, per dir:
  - `METAGGA=R2SCAN`, `ENCUT=800`, `KSPACING=0.5` (no `KPOINTS` file),
    `NPAR=1`, `KPAR=1`, `NSIM=32`
  - `NSW=100`, `POTIM=1.000`
  - `NBANDS=1000`
  - `TEBEG`/`SIGMA` = **preserved from each dir's original INCAR**
    (groups: 2000 K×12, 3000×6, 4000×14, 5000×6, 6000×13, 8000×10;
    σ = k_B·T, verified consistent)
- **POTCAR** = `MgSiOH__R2SCAN/setup_MLMD` Mg→Si→O, concatenated with
  `awk 1` (trailing-newline fix; plain `cat` corrupts it).
- **Submission:** `sub_vasp_gpu_h200.sh` per dir — 1 H200 GPU, 24 h,
  binary `…/vasp.6.6.0.gpu/bin/vasp_std__NCSA_DELTA_GPU`
  (the validated NVHPC build that runs on both A100 and H200 — see
  `benchmarks/vasp/NCSA_DELTA/H200_VASP_GPU__binary_and_submission_reference__20260518.md`).

**Job IDs:** H200 batch `qmd_h200` ≈ 18317647–18317946 (n0201 is an
earlier survivor; the other 59 were resubmitted 2026-05-18 ~02:11 after
the binary was standardized to `vasp_std__NCSA_DELTA_GPU`).
n02011 (100-atom) finished earlier on A100 (`qmd_gpu`, job 18314668).

---

## Why H200 (history)

- First A100 attempts failed: broken `cat`-ed POTCAR (lexical error) →
  fixed with `awk 1`.
- Then `1 GPU + NBANDS=1536` OOM'd on 40 GB A100 for the 150/200-atom
  cells (60 of 61). Dropped to `NBANDS=1000`; still tight on A100.
- Decision: run everything on **gpuH200x8** (141 GB) — ample memory,
  no OOM. n0201 (200-atom canary) confirmed running fine on H200.
- Binary cross-check (see benchmarks ref) proved the standard
  `vasp_std__NCSA_DELTA_GPU` works on both A100 and H200; standardized
  on it (no `_portable` needed).

---

## Tooling in the working dir (`…/MgSiO3/sim_data/`)

| File                                                 | Purpose                                                                                        |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `setup_R2SCAN_GPU.sh`                                | (re)prepares a dir: backup → POSCAR←CONTCAR, INCAR, POTCAR(awk 1), rm KPOINTS, drop sub script |
| `sub_vasp_gpu_h200.sh`                               | canonical H200 single-run script (per-dir copies exist)                                        |
| `rerun_failed_on_h200.sh`                            | one-shot sweep: resubmits **OOM-only** failures to H200; non-OOM → flagged REVIEW; won't loop  |
| `watch_and_h200.sh`                                  | background watcher — **currently NOT running** (user stopped it)                               |
| `ORIG_pre_R2SCAN/` (per dir)                         | backup of original INCAR/POSCAR/POTCAR/KPOINTS                                                 |
| `.h200_jobid` (per dir)                              | last H200 job id submitted for that dir                                                        |
| `move_to_h200_*.log`, `resubmit_h200_fixedbin_*.log` | old→new job-id maps                                                                            |

> ⚠️ Original large **OUTCAR**s were overwritten during the early failed
> attempts (not backed up). XDATCAR/OSZICAR/analysis from the prior PBE/PS
> runs survive only in dirs that never reran; treat the source these were
> copied from as the archive of the originals.

---

## How to check / resume

```bash
cd /work/nvme/bguf/akashgpt/qmd_data/MgSiO3/sim_data
squeue -u "$USER" -n qmd_h200,qmd_gpu -o "%i %T %M %Z"
# marker tally:
for d in n*; do
  [ -f $d/done_RUN_VASP ]   && echo "$d DONE"   ;
  [ -f $d/failed_RUN_VASP ] && echo "$d FAILED" ;
done
# MD progress of a dir: grep -c 'T=' n0201/OSZICAR   (target NSW=100)
```

- **On a failed dir:** `bash rerun_failed_on_h200.sh` resubmits genuine
  GPU-OOM failures to H200 and flags everything else `REVIEW` (do not
  blindly resubmit non-OOM failures).
- **No watcher is running** — nothing auto-monitors or auto-reruns.
  Restart it with `nohup bash watch_and_h200.sh &` if hands-off
  monitoring is wanted again.

## Done definition / next step

A dir is done when `done_RUN_VASP` exists and OSZICAR shows 100 `T=`
steps.

**DONE 2026-05-23:** all 61 dirs completed; DeePMD numpy data extracted
into `n*/deepmd/set.000/` via the canonical
`${ALCHEMY__main__MLDP}/extract_deepmd.py -d deepmd -ttr 10000 -e Mg Si O H He`
invocation. Extraction logs:

- `extract_deepmd_batch_20260523_130501.log` (initial, 3-element type_map; superseded)
- `extract_deepmd_batch_with_full_type_map_20260523_130946.log` (canonical, 5-element type_map)

Each `n*/deepmd/` is a single-set system with 100 frames (no train/test
split inside the system because `-ttr 10000` keeps all frames as
training-eligible). This data is now consumed by:
`training_bench/validation_MgSiO3_sim_data_all_retained__20260523/`
as the held-out test set for `se_e2_a` validation across all 5 retained
checkpoints per variant.

## Open items

- None for the VASP campaign. This note can be removed/archived after
  user confirmation; the downstream DeePMD validation work has its own
  tracking under
  `training_bench/validation_MgSiO3_sim_data_all_retained__20260523/`.
- Historical note: n0201 ran with the old `_portable`-spooled VASP
  binary (started pre-revert) — harmless (works on H200), just not the
  standardized `vasp_std__NCSA_DELTA_GPU`.
