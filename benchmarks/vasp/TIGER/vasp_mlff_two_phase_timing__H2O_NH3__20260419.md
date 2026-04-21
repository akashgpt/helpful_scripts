# VASP MLFF Timing Note: Two-Phase H2O/NH3 on TIGER

**Date analyzed:** 2026-04-19  
**Last updated:** 2026-04-20  
**Cluster:** Princeton TIGER / TIGER3 CPU Slurm environment  
**Workflow:** VASP MLFF two-phase NVT simulations  
**Main data location:** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3`

## Purpose

This note records a practical timing inference from the H2O/NH3 two-phase MLFF
runs. The goal was to determine whether increasing MLFF capacity, especially
`ML_MB`, was making the simulation slower, and to preserve enough setup context
so future runs are not compared blindly.

This is not a formal hardware benchmark. It is a workflow benchmark for these
specific VASP MLFF liquid/two-phase simulations on TIGER.

## Timing Method

Per-step speed was estimated from VASP `OUTCAR` timing lines:

- Preferred timing line: `LOOP+ ... real time ...`
- Fallback timing line: `LOOP: ... real time ...`
- Speed metric: sliding sum of real time over `N` ionic steps.
- Windows used in the saved plots: 10, 50, and 100 steps.
- The windows are sliding, not block-averaged: for a 100-step window the plotted
  windows are 1-100, 2-101, 3-102, and so on.
- The first timing entry is skipped to reduce initialization/refit contamination.
- The top panel in the plot uses a symlog y-scale to show both normal prediction
  steps and large training/refit bursts on the same axis.

Helper script:

```bash
python qmd/vasp/plot_vasp_step_speed.py \
	/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/250_H2O__256_NH3 \
	/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/250_H2O__256_NH3__b \
	/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/250_H2O__256_NH3__c \
	--labels base __b __c \
	--window 100 \
	--output /scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/step_speed_100step_sliding.png \
	--title H2O_NH3
```

Generated plot families:

- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/step_speed_10step_sliding.png`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/step_speed_50step_sliding.png`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/step_speed_100step_sliding.png`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_H2S/step_speed_10step_sliding.png`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_H2S/step_speed_50step_sliding.png`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_H2S/step_speed_100step_sliding.png`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2S_NH3/step_speed_10step_sliding.png`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2S_NH3/step_speed_50step_sliding.png`
- `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2S_NH3/step_speed_100step_sliding.png`

## Compared Runs

| Folder | Job ID | ML_MB | ML_MCONF | POTIM (fs) | NBANDS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `250_H2O__256_NH3` | 2555146 | 8000 | 2000 | 0.25 | 2912 | Base run; H basis reached ML_MB. |
| `250_H2O__256_NH3__b` | 2556804 | 16000 | 2000 | 0.25 | 2912 | Larger basis; mature prediction steps became fast. |
| `250_H2O__256_NH3__c` | 2559104 | 24000 | 1000 | 0.25 | 2912 | Larger basis, smaller structure cap; current/active run during analysis. |

All runs used the same two-phase H2O/NH3 composition:

- 250 H2O molecules
- 256 NH3 molecules
- Species order in POSCAR/POTCAR: `H O N`
- NVT MLFF setup from the H2S/NH3/H2O simulation campaign
- 8 TIGER CPU nodes in the submitted scripts used for the compared runs

## H2O/NH3 Speed Summary

The following values are based on `OUTCAR` timing, not on analysis-folder
derived summaries. These are local sliding-window values near the end of each
available trajectory, not whole-run averages.

| Folder | ML_MB | Parsed LOOP+ steps after skipping first | Last 10-step speed | Last 50-step speed | Last 100-step speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `250_H2O__256_NH3` | 8000 | 4274 | 24.88 s = 1447 steps/hour | 633.43 s = 284 steps/hour | 757.84 s = 475 steps/hour |
| `250_H2O__256_NH3__b` | 16000 | 2245 | 40.96 s = 879 steps/hour | 204.89 s = 879 steps/hour | 411.74 s = 874 steps/hour |
| `250_H2O__256_NH3__c` | 24000 | 705 | 44.32 s = 812 steps/hour | 880.63 s = 204 steps/hour | 1104.57 s = 326 steps/hour |

Whole-run-like averages tell a different story because early training/refit
bursts are diluted differently depending on the number of completed steps:

| Folder | Approx whole-run LOOP+ speed |
| --- | ---: |
| `250_H2O__256_NH3` | 430 steps/hour |
| `250_H2O__256_NH3__b` | 282 steps/hour |
| `250_H2O__256_NH3__c` | 166 steps/hour |

This distinction matters. "Steps/hour" must always specify a window. Whole-run
speed answers "which historical job accumulated steps fastest overall?" The
sliding-window speed answers "how fast is the current/recent regime?"

## MLFF Health Context

From the MLFF health summaries / parsed ML_LOGFILE state near the time of
analysis:

| Folder | Stored structures | H basis | O basis | N basis | Memory estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `250_H2O__256_NH3` | 61 / 2000 | 8000 / 8000 (100.0%) | 2687 / 8000 (33.6%) | 1959 / 8000 (24.5%) | 46.83 GB |
| `250_H2O__256_NH3__b` | 95 / 2000 | 12799 / 16000 (80.0%) | 5023 / 16000 (31.4%) | 3578 / 16000 (22.4%) | 75.31 GB |
| `250_H2O__256_NH3__c` | 105 / 1000 | 13567 / 24000 (56.5%) | 5609 / 24000 (23.4%) | 3920 / 24000 (16.3%) | 60.40 GB |

DFT-call rates were low in all three H2O/NH3 runs, so the observed slowdown is
not simply "more ab-initio calls":

| Folder | DFT calls / ML steps | Approx DFT-call rate |
| --- | ---: | ---: |
| `250_H2O__256_NH3` | 56 / 4276 | 1.3% |
| `250_H2O__256_NH3__b` | 39 / 2248 | 1.7% |
| `250_H2O__256_NH3__c` | 13 / 549 | 2.4% |

## Main Inference

Increasing `ML_MB` gives more headroom for local reference configurations, but
it does not guarantee faster simulation. There are two competing effects:

- A better/larger MLFF can reduce ab-initio calls and improve stability.
- A larger retained basis can make every MLFF prediction/refit step more
  expensive.

The current H2O/NH3 data show:

- By whole-run average, the base `ML_MB=8000` run looks fastest because it has
  many mature steps and its startup/training costs are strongly diluted.
- By recent 50- and 100-step sliding-window speed, `ML_MB=16000` is fastest for
  H2O/NH3.
- `ML_MB=24000` has more H-basis headroom but is slower in the available
  50/100-step recent windows.

The sulfur systems add more caution:

| System | Fastest recent 100-step comparison | Important caveat |
| --- | --- | --- |
| H2O/NH3 | `ML_MB=16000` | Strongest recent-window result. |
| H2O/H2S | `ML_MB=16000` | Runs are shorter and restart histories differ. |
| H2S/NH3 | `ML_MB=16000` is slightly faster over 100 steps, but `ML_MB=8000` is slightly faster over 200 steps | Differences are small and window-dependent. |

Practical interpretation as of 2026-04-20:

`ML_MB=16000` is a good practical compromise to test first for the two-phase
systems because it often improves recent-window speed without the severe
headroom limitation of `ML_MB=8000`. However, it should not be treated as a
universal optimum. For production choices, compare:

- whole-run accumulated steps/hour,
- recent 50/100/200-step sliding-window speed,
- basis saturation by species,
- absolute retained basis count,
- DFT-call rate and BEEF behavior,
- and whether the run is still in a training/refit-heavy regime.

Percent-full basis is a warning signal for capacity risk, but throughput tends
to track the absolute retained basis size and the current training/refit phase
more than percent-full alone.

## Important Caveats

- These comparisons are from active MLFF production/restart runs, not isolated
  controlled benchmarks. Early training/refit bursts can dominate whole-job
  timing.
- Some `__c` folders were active, short, or failed during analysis, so their
  mature long-time speed may not be represented.
- The base and `__b` jobs were not clean "completed production" benchmarks;
  they were part of the iterative setup/restart campaign. The `OUTCAR` timing
  data are still useful for per-step speed.
- The 10-step sliding window highlights very recent behavior and short bursts.
  The 50/100-step windows are better for judging smoothed recent throughput.
- For short failed runs, a 50- or 100-step window may be impossible or heavily
  dominated by initialization/training.

## Related Sulfur-System Lesson

The sulfur-containing two-phase systems (`H2O_H2S` and `H2S_NH3`) showed a
separate failure mode when the retained MLFF basis became too large. Jobs such
as `2559097`, `2559099`, and `2559944` died during MLFF covariance broadcast
with MPI `internal_Bcast` negative-count errors, before useful MD progress.

Observed pattern:

- H2O/NH3 with `ML_MB=24000` could start and run.
- Sulfur systems failed at large `ML_MB` values because the retained basis was
  much larger, especially for S.
- This did not look like a normal memory-limit/OOM failure; memory estimates
  were far below requested node memory.
- The likely issue is a VASP/MPI large-broadcast/count limitation during MLFF
  covariance matrix broadcast.

Practical sulfur-system guidance:

- Avoid `ML_MB=20000+` for sulfur-containing two-phase TIGER runs unless the
  MPI/VASP broadcast issue is resolved.
- Prefer testing `ML_MB=16000` with `ML_LBASIS_DISCARD = .TRUE.` first.
- Treat the retained basis counts in `ML_LOGFILE` / `MLFF_HEALTH_check_up.dat`
  as more important than the nominal `ML_MB` alone.

## Future Use

When comparing future TIGER VASP MLFF runs, use the same timing helper and keep
the following values together:

- Composition and species order
- `ML_MB`, `ML_MCONF`, `ML_CTIFOR`, `ML_LBASIS_DISCARD`
- Stored training structures after sparsification
- Per-species retained basis counts
- DFT-call rate
- Sliding 10/50/100-step `LOOP+` wall time after skipping the first step
- Whether a speed estimate is whole-run or local sliding-window speed

This keeps speed comparisons tied to the scientific/MLFF state of the run,
instead of treating all VASP jobs as interchangeable wall-clock benchmarks.
