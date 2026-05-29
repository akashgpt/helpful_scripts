# ONGOING — DPA-2 `current_auto` 1M-step retrain on H200 (NCSA Delta, gpuH200x8)

**Started tracking:** 2026-05-24
**Cluster:** NCSA Delta · partition `gpuH200x8` · account `bguf-delta-gpu`
**Working dir:** `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_current_auto_1M_H200`
**Job:** `18452299` (PENDING with `Priority` queue reason at submission, `48:00:00` walltime)

## Purpose

`dpa2_current_auto` at 200k steps on A100 was the **best in-distribution** model (Val E RMSE/atom = 0.0525) but **worst on multi-TPa OOD outliers** (F RMSE on `n0226` = 1634) in the
[MgSiO3 sim_data sweep](../../benchmarks/deepmd/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426/validation_MgSiO3_sim_data_20260523/).

This retrain tests whether the catastrophic OOD failure is **fundamental to the architecture** or just a **symptom of under-training** (200k steps vs 1M for the `se_e2_a` family).

## Config

Source-of-truth shared input: `training_bench/shared/train_dpa2_current_auto_1M_H200.json`.
Identical to the original 200k input *except*:

| Field | 200k baseline | This run (1M) |
| --- | --- | --- |
| `training.numb_steps` | 200,000 | **1,000,000** |
| `learning_rate.decay_steps` | 10,000 | **50,000** (scaled 5× so exp LR floor lands at end of training) |
| `training.validation_data` | (absent) | **all 61 MgSiO3 sim_data systems**, `batch_size=1`, `numb_btch=4` |
| `#SBATCH --partition` | `gpuA100x4` | **`gpuH200x8`** |
| `#SBATCH --time` | `24:00:00` | **`48:00:00`** |

Everything else (training_data = 255 MgSiOH systems, descriptor = DPA-2 with `use_three_body=false` + `auto:1.1` neighbor selection, fitting net, loss prefs) is byte-identical to the 200k baseline.

### Why we have validation_data this time

`lcurve.out` will now log `rmse_e_val`, `rmse_f_val`, `rmse_v_val` every `disp_freq=100` steps over 4 randomly-rotated batches from the 61 MgSiO3 systems. This gives a real-time generalization signal — we'll see whether val E/F/V are trending down (good) or whether OOD systems pull them up over time.

**Methodological caveat:** the val set here is the **same** MgSiO3 sim_data used to evaluate the 4-variant comparison from 2026-05-23. That means this retrained model can no longer be treated as fully blind on MgSiO3 sim_data. **Use the val signal for monitoring / overfitting detection only**; do not use it to declare "MgSiO3 sim_data RMSE improved at 1M steps" — that would be circular.

For a future fully-blind held-out comparison of the 1M retrain, generate a separate independent collection (or carve out a strict-disjoint subset of the MgSiOH TRAIN tree).

## Resource shape

- 1× NVIDIA H200 (141 GB), 2 CPUs, 64 GB RAM, 48 h walltime
- `conda activate ALCHEMY_env__PT`, DeePMD-kit v3.1.3, PyTorch 2.6.0
- Estimated wall: 200k took 8:53:34 on A100; H200 is ≈ 1.5–2× faster per step for these workloads (per [`H200_VASP_GPU__binary_and_submission_reference`](../../benchmarks/vasp/NCSA_DELTA/H200_VASP_GPU__binary_and_submission_reference__20260518.md) and consistent with our recent dp_test timing). Scaling: `8:53:34 × 5 / 1.75 ≈ 25 h`. The 48 h walltime is a safety buffer.

## How to check

```bash
squeue -j 18452299 -o "%i %T %P %M %l %R"
sacct -j 18452299 --format=JobID,State,Elapsed,ExitCode -P -n

cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_current_auto_1M_H200
tail -25 log.train
tail -5 lcurve.out
```

Quick learning-curve render once running:

```bash
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_current_auto_1M_H200
module purge && module load miniforge3-python && eval "$(conda shell.bash hook)" && conda activate ALCHEMY_env
python ${ALCHEMY__main__MLDP}/util/plots_mod.py
```

The resulting `efv_plots.png` will include train AND val curves on each panel (since `lcurve.out` now has `_val` columns).

## Resume

`sub.sh` is auto-restart aware:

```bash
if ls model.ckpt-*.pt > /dev/null 2>&1; then
    LAST_CKPT=$(ls -1 model.ckpt-*.pt | sort -V | tail -1)
    dp --pt train --restart "$LAST_CKPT" input.json
else
    dp --pt train input.json
fi
```

So if walltime hits before 1M steps, just `sbatch sub.sh` again and it picks up the latest ckpt.

## Done definition / next step

A run is done when `log.train` shows `Trained model has been saved to:` after step 1,000,000 (or when the `model.ckpt-1000000.pt` file lands). Once finished:

1. Re-run `dp --pt test` against the 61-system MgSiO3 sim_data via the existing harness at
   `training_bench/validation_MgSiO3_sim_data_dpa2__20260523/run_dp_test_all.py`
   (substitute model path; add a 5th row to the same `CASES.tsv` or use a sibling validation dir).
2. Compare against the original 200k results: does OOD F RMSE drop from `47.7` (all-61) and `1634` (n0226)? If yes — under-training was the issue. If no — DPA-2 architecture lacks the OOD capacity, and the only path forward is training-distribution expansion (add high-pressure MgSiO3 frames to TRAIN).
3. **For a fully-blind comparison** generate independent test data (not the same `MgSiO3/sim_data` collection used as validation here).

## Status snapshot — 2026-05-27

Job `18452299` completed normally:

```text
18452299|dpa2_cur_auto_1M_H200|COMPLETED|0:0|20:56:02|2026-05-25T20:50:21|2026-05-26T17:46:23|gpue05|gpuH200x8
```

The run reached the intended checkpoint:

- `model.ckpt-1000000.pt` exists and `model.ckpt.pt -> model.ckpt-1000000.pt`.
- `log.train` ends with `Trained model has been saved to: model.ckpt`.
- DeePMD reported average training time `0.0753 s/batch` after the first 100 batches.

Training-curve analysis:

| Run | final step | final train E/F/V | last-100k median train E/F/V | final val E/F/V | last-100k median val E/F/V |
| --- | ---: | --- | --- | --- | --- |
| `dpa2_current_auto_200k` | 200000 | `0.00589 / 0.224 / 0.00852` | `0.00543 / 0.260 / 0.0162` | n/a | n/a |
| `dpa2_current_auto_1M_H200` | 1000000 | `0.167 / 5.12 / 3.70` | `0.319 / 3.73 / 2.545` | `0.352 / 2.56 / 0.259` | `0.624 / 3.46 / 2.11` |

Interpretation: the 1M H200 extension did **not** improve the `current_auto` architecture in the training curve. Its late training force/virial errors are much worse than the 200k run. The run completed cleanly, and the only DeePMD warning is the same neighbor-selection warning seen in the 200k baseline (`sel of type 0 ... expected >=107, set to 40`), so this is not explained by a new runtime failure.

Existing 61-system MgSiO3 validation for the 200k DPA-2 suite remains:

| Case | all-61 E/F/V |
| --- | --- |
| `dpa2_current_auto` | `1.810 / 47.736 / 3.656` |
| `dpa2_doc_medium_currentloss` | `1.052 / 1.260 / 1.986` |
| `dpa2_doc_medium_no3body` | `1.166 / 1.278 / 2.162` |
| `dpa2_doc_medium_novirial` | `1.237 / 1.409 / 2.230` |

The 1M checkpoint has not yet been run through the full 61-system `run_dp_test_all.py` harness, so the OOD conclusion should stay provisional until that direct validation row is generated. Based on the lcurve, under-training alone is unlikely to explain the `current_auto` failure.

## Validation sweep — submitted 2026-05-27

Full 61-system MgSiO3 validation for `model.ckpt-1000000.pt` is now submitted.

```text
Job: 18537583
Validation dir: /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_dpa2_current_auto_1M_H200__20260527
Model: /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_current_auto_1M_H200/model.ckpt-1000000.pt
Script: run_validation.sbatch
Array: 0-0
```

Check status:

```bash
squeue -j 18537583 -o "%i %T %P %M %l %R"
sacct -j 18537583 --format=JobID,State,Elapsed,ExitCode -P -n
```

Once complete, summarize or inspect:

```bash
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_dpa2_current_auto_1M_H200__20260527
python summarize_validation.py
grep '^aggregate' VALIDATION_SUMMARY.tsv
```

Learning-curve plot generated:

```text
/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_current_auto_1M_H200/dpa2_current_auto_1M_H200__train_val_rmse.png
```

Per policy this `ONGOING/` note stays until user explicitly confirms removal.

## Validation sweep — completed 2026-05-27

Job `18537583_0` completed cleanly on `gpue02`:

```text
18537583_0|val_dpa2_1M_H200|COMPLETED|0:0|00:04:41|2026-05-27T15:47:03|2026-05-27T15:51:44|gpue02|gpuH200x8
```

Full 61-system aggregate from `VALIDATION_SUMMARY.tsv`:

| case | frames | val E RMSE/atom | val F RMSE | val V RMSE/atom |
| --- | ---: | ---: | ---: | ---: |
| `dpa2_current_auto_1M_H200` | 6100 | 2.53832 | 4.01468 | 4.03399 |

Worst systems by energy/force/virial are the same high-error middle block:

| system | E RMSE/atom | F RMSE | V RMSE/atom |
| --- | ---: | ---: | ---: |
| `n0226` | 40.92477 | 17.41309 | 45.08185 |
| `n0217` | 40.51059 | 15.32028 | 44.89741 |
| `n0221` | 17.31966 | 12.32141 | 24.66431 |
| `n0211` | 17.11817 | 10.72095 | 24.51017 |
| `n0222` | 7.18072 | 8.86597 | 13.18194 |
| `n0212` | 6.93093 | 7.59251 | 12.97933 |

Conclusion: 1M continued training did not rescue `current_auto`. It reduced the old 200k force aggregate (`47.736`) but remains scientifically poor (`F=4.015 eV/A`, `V=4.034 eV/atom`) and has catastrophic OOD systems. The lcurve also shows validation staying noisy and above train, so this is architecture/setup/data-coverage failure rather than a simple walltime-shortfall issue.
