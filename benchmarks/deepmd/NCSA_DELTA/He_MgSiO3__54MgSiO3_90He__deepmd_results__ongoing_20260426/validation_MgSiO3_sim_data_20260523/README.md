# Held-out Validation on MgSiO3 sim_data — 2026-05-23

Combined held-out comparison of the **10 fully-trained `se_e2_a` TF variants** and the **4 DPA-2 PT variants** against the new canonical held-out test set:

```
/work/nvme/bguf/akashgpt/qmd_data/MgSiO3/sim_data/n*/deepmd
```

61 deepmd systems × 100 R2SCAN frames each = **6,100 frames**. Cells of 100 / 150 / 200 atoms; conditions span 2000 – 8000 K and 0 – ~7.4 TPa (see per-system T/P/ρ characterization in [`../../../ONGOING/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__validation_MgSiO3_5ckpt__20260523.md`](../../../ONGOING/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__validation_MgSiO3_5ckpt__20260523.md)). Type_map enforced as `[Mg, Si, O, H, He]` to match the trained 5-type models.

## Files

- [`VALIDATION_RMSE_COMPARISON__MgSiO3_sim_data__20260523.tsv`](VALIDATION_RMSE_COMPARISON__MgSiO3_sim_data__20260523.tsv) — curated TSV (both cuts).
  Columns: `variant, kind, params, train_wall, n_ckpts, A_e_mean, A_e_std, A_f_mean, A_f_std, A_v_mean, A_v_std, B_e_mean, B_e_std, B_f_mean, B_f_std, B_v_mean, B_v_std`.
  Where **A** = all 61 systems (frame-weighted), **B** = 57 in-distribution systems (4 OOD high-P outliers `n0211 / n0217 / n0221 / n0226` excluded).
- [`VALIDATION_RMSE_COMPARISON__MgSiO3_sim_data__in_distribution_only__20260523.tsv`](VALIDATION_RMSE_COMPARISON__MgSiO3_sim_data__in_distribution_only__20260523.tsv) — trimmed TSV with just the in-distribution (B) columns, plus `train_seconds`, `gpu_type`, and `effective_train_seconds_2x_h200` (H200 runs counted ×2; rule is encoded but no validated variant used H200 — all 14 ran on A100).
- [`validation_RMSE_panel_2x2_20260523.png`](validation_RMSE_panel_2x2_20260523.png) — 2×2 panel barplot of E and F RMSE for the all-61 vs in-distribution cuts.
- [`validation_RMSE_in_distribution_with_runtime_20260523.png`](validation_RMSE_in_distribution_with_runtime_20260523.png) — 4-row panel barplot focused on **in-distribution** numbers: E RMSE/atom, F RMSE, V RMSE/atom, and effective train time (h). Sorted ascending by E. Each bar tagged with its GPU type. H200 effective-time × 2 rule documented in the caption.
- [`../learning_curves_20260523/learning_curves_overlay_all14_20260523.png`](../learning_curves_20260523/learning_curves_overlay_all14_20260523.png) — **Overlay of all 14 lcurve.out files** (10 TF + 4 PT) of all 14 lcurve.out files on the 2×2 (Total/E/F/V) panels from `${ALCHEMY__main__MLDP}/util/plots_mod.py`. TF blue family / DPA-2 orange family; solid = first-half of each family, dashed = second-half. Smoothing matches `plots_mod.py` default rolling-mean window. `novirial` is absent from the V panel (no `rmse_v_trn` column in its lcurve.out).

## Source validation directories

- `se_e2_a` (50 cases = 10 variants × 5 retained ckpts each):
  `../../../../testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_all_retained__20260523/`
- DPA-2 PT (4 cases = 4 variants × 1 ckpt at step 200,000):
  `../../../../testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_MgSiO3_sim_data_dpa2__20260523/`

Both used a single-process Python harness (`run_dp_test_all.py`) that loads the model once and iterates over the 61 systems in one process — eliminates the per-system srun + dp-startup overhead (~30–40 s × 61 = ~30 min per case) that the old per-system pipeline paid. Total wall: ~25 min for the 50-task se_e2_a array (gpuH200x8, %6 concurrent) and ~10 min for the 4-case DPA-2 array.

## Headline findings

### In-distribution (57 systems, P ≲ 1 TPa)

`dpa2_current_auto` (DPA-2 PT, no 3-body, 200k steps, A100, `08:53:34` train wall) is the **best model on every metric**:

- **E RMSE/atom = 0.0525 eV/atom** (next best `fit_deep2x` 0.0573).
- **F RMSE = 0.326 eV/Å** (next best `balanced_2x` 0.367; the best se_e2_a on F).
- Trained 5× fewer steps and ~30 % less wall time than the best `se_e2_a` competitor.

The other 3 DPA-2 variants (the `doc_medium_*` group, with the 3-body block kept in schema) are mid-pack, similar to the better `se_e2_a` intermediates.

### All-61 (includes 4 multi-TPa OOD outliers)

Picture flips. `dpa2_current_auto` is the **worst** model:

- E RMSE/atom jumps to **1.810** (vs `fit_deep2x` 0.637 = the best).
- F RMSE explodes to **47.7** — driven by `n0226` F RMSE of `1634` and `n0217` F RMSE of `1218`. Both are ~7.4 TPa MgSiO3.

The `doc_medium_*` DPA-2 variants degrade much less catastrophically on OOD (1.05–1.24 vs 1.81 on E). Whatever the 3-body block contributes to OOD generalization is load-bearing in the high-density tail.

### Take-aways

- **DPA-2 sample efficiency claim verifies in-distribution.** 200k steps + ~9 h beats `se_e2_a`'s 1M steps + 11–46 h on both E and F.
- **DPA-2 simple architecture (`current_auto`, no 3-body) over-fits the training distribution.** Sharp transition: best in-distribution, worst on the multi-TPa edge of the test set.
- **3-body block (in the `doc_medium_*` schema) is the extrapolation cushion.** Models with it stay reasonable on OOD; the model without it fails 10–100× worse on the same systems.
- **For Earth-mantle / shallow super-Earth pressures (≲ 400 GPa)**, the best deployable model from this sweep is `dpa2_current_auto`. **For deep super-Earth / gas-giant interiors (≳ 2 TPa)**, *none of these 14 models work* — that regime is missing from the training distribution. The fix is to add high-pressure MgSiO3 frames to the training collection and retrain.

## Caveats

- DPA-2 results are from a **single checkpoint** (step 200,000); `se_e2_a` results are means across **5 retained checkpoints** (steps 996,000–1,000,000). Std is `n/a` for DPA-2 rows in the TSV. The retained-ckpt std for `se_e2_a` is ≤ 1.5e-4 on E across all variants — i.e., the LR-tail of `se_e2_a` has converged completely — so the missing DPA-2 std is unlikely to flip the rankings.
- DPA-2 models were trained for only 200,000 steps. Whether `dpa2_current_auto`'s OOD failure persists at 1M steps is an open question. Re-training to 1M would cost ~`45 h` per variant on A100, similar to `balanced_10x`.
- `big` (`se_e2_a`) compression failed on the TF GraphDef serialization limit; its all-row values come from the **non-compressed** `pv.pb`, kept in the TSV as a reference. The other 9 `se_e2_a` variants are compressed `freeze + compress + dp test`.
- The legacy `71MgSiO3_5He` validation (13 systems, 324 frames; `compressed_validation_20260518/`) is **preserved untouched** for historical reference but should not drive new ranking decisions — see [`heldout-test-set-mgsio3-sim-data`](file:///u/agupta46/.claude/projects/-projects-bguf-akashgpt-run-scripts-helpful-scripts/memory/heldout-test-set-mgsio3-sim-data.md) memory note.
