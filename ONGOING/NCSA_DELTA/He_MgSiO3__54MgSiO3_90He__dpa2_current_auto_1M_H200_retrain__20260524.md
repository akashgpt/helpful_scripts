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

Per policy this `ONGOING/` note stays until user explicitly confirms removal.
