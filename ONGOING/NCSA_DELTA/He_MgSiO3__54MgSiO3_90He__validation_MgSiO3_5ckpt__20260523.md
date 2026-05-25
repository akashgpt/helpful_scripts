# ONGOING — He_MgSiO3 / MgSiOH held-out validation on MgSiO3 sim_data (NCSA Delta, gpuH200x8)

**Started tracking:** 2026-05-23
**Cluster:** NCSA Delta · partition `gpuH200x8` · account `bguf-delta-gpu`
**Working dirs:**
- `se_e2_a` (10 variants × 5 ckpts = 50 cases): `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_all_retained__20260523`
- DPA-2 PT (4 variants × 1 ckpt = 4 cases): `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_dpa2__20260523`
**Status as of 2026-05-24:**
- `se_e2_a` validation: smoke-test `18440242_0` (`base__ck996000`) COMPLETED in 2:16; main array `18440747_[1-49]` all 49 tasks COMPLETED with 0 failures (1:39–6:14 each).
- DPA-2 validation: first smoke-test `18442772_0` FAILED at 24 s (wrong PT import path); harness patched to use unified `deepmd.infer.deep_pot.DeepPot`; resubmit `18449967_0` COMPLETED in 3:54; remaining `18451353_[1-3]` all COMPLETED in 3:04–4:02. All 4 DPA-2 cases done.
- **Both sweeps complete.** Curated comparison + READMEs landed at `benchmarks/deepmd/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426/validation_MgSiO3_sim_data_20260523/`.

---

## What is running

Compressed held-out validation of **all 5 retained checkpoints** for each of the **10 previously-validated `se_e2_a` variants**. Total: 50 cases (1 already done via smoke-test + 49 in the main array).

For each `(variant, ckpt_step)` case the per-array-task pipeline is:

1. Copy `model.ckpt-${ckpt_step}.{index,meta,data-00000-of-00001}` from the variant's training dir into a private `model-compression/<case>/` scratch.
2. Write a synthetic `checkpoint` file pointing at that step so `dp --tf freeze` picks the right meta/data.
3. `dp --tf freeze -o pv.pb` (single H200 GPU).
4. `dp --tf compress -i pv.pb -o pv_comp.pb` (falls back to `pv.pb` for `dp test` if compress fails, so the `big` variant still gets numbers).
5. **Single-process Python harness** `run_dp_test_all.py` loads `pv_comp.pb` once and loops over all 61 systems in `SYSTEMS.tsv`, writing one `log.dp_test` per system in the same format the existing `summarize_validation.py` parses. Smoke-test: 56 s for all 61 systems including model load.

### Variants × checkpoints

10 variants × 5 retained ckpts each = 50 cases:

| Variant | Retained ckpts |
| --- | --- |
| `base` (2.67 M) | 996k, 997k, 998k, 999k, 1000k |
| `big` (23.91 M) | 996k, 997k, 998k, 999k, 1000k |
| `big2x` (5.29 M) | 996k, 997k, 998k, 999k, 1000k |
| `big5x` (13.41 M) | 996k, 997k, 998k, 999k, 1000k |
| `balanced_2x` (5.22 M) | 996k, 997k, 998k, 999k, 1000k |
| `balanced_5x` (13.69 M) | 996k, 997k, 998k, 999k, 1000k |
| `balanced_10x` (26.80 M) | 996k, 997k, 998k, 999k, 1000k |
| `fit_deep2x` (5.28 M) | 996k, 997k, 998k, 999k, 1000k |
| `fit_deep10x` (26.77 M) | 996k, 997k, 998k, 999k, 1000k |
| `both_deep2x` (5.38 M) | 996k, 997k, 998k, 999k, 1000k |

The 4 partial runs (`fit_deep5x` NaN, `desc_deep2x` / `desc_deep10x` / `both_deep10x` TIMEOUT) are **excluded** per the user instruction "for which we previously did such runs"; only the 10 fully-trained variants are in scope.

### Held-out test set (canonical going forward)

61 DeePMD systems × 100 R2SCAN frames each = **6,100 frames** at:

```
/work/nvme/bguf/akashgpt/qmd_data/MgSiO3/sim_data/n*/deepmd
```

Built 2026-05-23 from completed R2SCAN OUTCARs via
`${ALCHEMY__main__MLDP}/extract_deepmd.py -d deepmd -ttr 10000 -e Mg Si O H He`.
Type_map is `[Mg, Si, O, H, He]` (H and He as zero-atom types) to match the
trained `se_e2_a` models (5 types per `sel=[50,50,140,700,500]`).

This replaces the older `71MgSiO3_5He` validation set (13 systems, 324 frames)
for new comparison work. Old `71MgSiO3_5He` validation dirs and the curated
`compressed_validation_20260518/VALIDATION_RMSE_COMPARISON__71MgSiO3_5He__20260518.tsv`
are kept intact for historical reference — do not modify.

---

## Resources

Per-task `--array=1-49%6` (max 6 concurrent):

- `#SBATCH --partition=gpuH200x8`
- `#SBATCH --gpus-per-task=1` (1 H200)
- `#SBATCH --cpus-per-task=2`
- `#SBATCH --mem=80G`
- `#SBATCH --time=00:30:00` (smoke-test wall: 2:16; larger models expected ~3-5 min)
- `DP_INFER_BATCH_SIZE=100`

---

## How to check

```bash
squeue -j 18440747 -o "%i %T %P %M %l %R"
sacct -j 18440242,18440747 --format=JobID,JobName%20,State,Elapsed,ExitCode -P -n | grep -v "\."
```

Spot-check a single case:

```bash
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_all_retained__20260523
tail -20 slurm-18440747_<task>.out
tail -10 results/<case_id>/log.harness
ls results/<case_id>/ | wc -l       # should be 62 = 61 systems + log.harness
```

Aggregate after completion:

```bash
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_all_retained__20260523
python summarize_validation.py
head -2 VALIDATION_SUMMARY.tsv
awk -F'\t' '$1=="aggregate" && $4=="all" && $5=="all" && $6=="all"' VALIDATION_SUMMARY.tsv
```

---

## Resume

If a task fails, resubmit just that array index:

```bash
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_all_retained__20260523
sbatch --array=<idx> --time=00:30:00 run_validation.sbatch
```

To redo the whole array if needed (e.g., if the harness needs a fix):

```bash
sbatch --array=0-49%6 --time=00:30:00 run_validation.sbatch
```

---

## Done definition / next step

**DONE 2026-05-24:**

1. ✅ `VALIDATION_SUMMARY.tsv` regenerated for both validation dirs (50 se_e2_a cases + 4 DPA-2 cases).
2. ✅ Per-variant mean ± std of held-out E/F/V RMSE/atom computed (5 ckpts for se_e2_a → std ≤ 1.5e-4 across 5 retained ckpts on E; single ckpt-200000 for DPA-2).
3. ✅ Curated comparison TSV at `benchmarks/deepmd/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426/validation_MgSiO3_sim_data_20260523/VALIDATION_RMSE_COMPARISON__MgSiO3_sim_data__20260523.tsv`.
4. ✅ Sibling artifacts in the same folder: `README.md` (combined scientific reading), `validation_RMSE_panel_2x2_20260523.png` (2×2 panel of val E/F for all-61 vs in-distribution).
5. ✅ Legacy `compressed_validation_20260518/*` left intact for historical reference (per policy).

**Outstanding (informational only — no in-flight Slurm work):**

- This `ONGOING/` note can be removed/archived after user confirmation. The work it tracks is complete; only the parent diagnostic note `He_MgSiO3__54MgSiO3_90He__deepmd_training__20260518.md` should remain until the DPA-2 follow-up question (longer DPA-2 run to test whether OOD failure persists at 1M steps) is resolved.

## Key results (summary)

**In-distribution (57 systems, P ≲ 1 TPa):** DPA-2 `current_auto` wins both E and F RMSE at ~30 % less wall time than the best `se_e2_a` (`fit_deep2x`).

**All 61 (with 4 multi-TPa OOD outliers):** DPA-2 `current_auto` is the *worst* — F RMSE explodes to ~48 (driven by ~1634 on n0226, ~1218 on n0217). The `doc_medium_*` DPA-2 variants degrade much less. Whatever the 3-body block contributes to extrapolation is load-bearing in the high-density tail.

**Headline finding:** for ≲ 400 GPa MgSiO3, deploy `dpa2_current_auto`. For deep super-Earth / gas-giant interiors (≳ 2 TPa) **none of these 14 models work** — the training collection lacks coverage there.

## Notes

- The legacy 71MgSiO3_5He validation pipeline (in
  `training_bench/validation_71MgSiO3_5He__*` dirs) used per-system `srun ... dp test`
  invocations and paid ~30-40 s of srun + dp startup overhead per system. The
  refactored `run_dp_test_all.py` harness loads the model once per case and
  loops over all 61 systems in a single Python process — dropped per-case wall
  from a projected ~3 h to ~3 min.
- Two earlier validation experiments under this branch were cancelled before
  any task ran (no resources spent):
  - `validation_71MgSiO3_5He__best_retained_ckpt__20260523/` (14-case
    best-of-retained on the legacy test set; superseded by the new MgSiO3
    held-out work in this note).
  - Job `18440109` (50-case version of this same sweep but with the slow
    per-system srun pipeline; replaced by job `18440747` with the harness).
