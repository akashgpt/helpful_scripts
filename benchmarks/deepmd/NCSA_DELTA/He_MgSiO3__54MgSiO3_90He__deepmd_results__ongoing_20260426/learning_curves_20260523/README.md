# DeePMD `se_e2_a` Learning Curves — 2026-05-23

Each `efv_plots__<variant>.png` is the 2×2 EFV (Energy / Force / Virial / Total) loss curve produced by

```
python ${ALCHEMY__main__MLDP}/util/plots_mod.py
```

run inside the corresponding `training_bench/variant_train_se_e2_a_TF_<variant>/` directory (which contains the source `lcurve.out`). The script auto-detects DeePMD-kit v2/v3 schema (5 numeric columns: `step rmse_trn rmse_e_trn rmse_f_trn rmse_v_trn lr`); the LR column is overlaid on the total-loss panel via a secondary y-axis. Plots use the rolling-mean / rolling-median smoothing the script applies internally.

## Variants

All 10 are the previously-held-out-validated TF `se_e2_a` runs (1,000,000 steps each, single A100 on `gpuA100x4`):

| File | Variant | Params | Train wall |
| --- | --- | ---: | ---: |
| `efv_plots__base.png` | baseline | 2,665,655 | `10:59:12` |
| `efv_plots__big.png` | width-only big | 23,910,455 | `22:56:06` |
| `efv_plots__big2x.png` | width-only 2x | 5,289,155 | `13:31:10` |
| `efv_plots__big5x.png` | width-only 5x | 13,406,905 | `17:26:08` |
| `efv_plots__balanced_2x.png` | balanced 2x | 5,216,405 | `15:29:47` |
| `efv_plots__balanced_5x.png` | balanced 5x | 13,690,655 | `27:35:33` |
| `efv_plots__balanced_10x.png` | balanced 10x | 26,804,305 | `46:08:53` |
| `efv_plots__fit_deep2x.png` | fitting-depth 2x | 5,279,255 | `12:48:53` |
| `efv_plots__fit_deep10x.png` | fitting-depth 10x | 26,768,855 | `28:54:00` |
| `efv_plots__both_deep2x.png` | desc+fit depth 2x | 5,380,155 | `28:04:27` |

## Source `lcurve.out` files

Located at:

```
/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_se_e2_a_TF_<variant>/lcurve.out
```

The original `efv_plots.png` written by `plots_mod.py` also exists in each training dir (e.g. `variant_train_se_e2_a_TF_balanced_5x/efv_plots.png`); the files in this folder are renamed copies for one-stop sharing.

## DPA-2 PT variants (added 2026-05-23)

The 4 200k-step PyTorch DPA-2 diagnostic runs (jobs `18320016/17/18/19`) are
included here purely for the learning-curve view; **they have not been
held-out validated** against any test set yet, so no `dp test` performance
numbers exist for them. Same training collection as the `se_e2_a` family
(255 MgSiOH systems).

| File | Variant | Train wall |
| --- | --- | ---: |
| `efv_plots__dpa2_current_auto.png` | original DPA-2 (`use_three_body=false`) | `08:53:34` |
| `efv_plots__dpa2_doc_medium_currentloss.png` | doc-medium DPA-2 with current loss | `09:36:21` |
| `efv_plots__dpa2_doc_medium_novirial.png` | doc-medium DPA-2, no virial loss | `09:37:24` |
| `efv_plots__dpa2_doc_medium_no3body.png` | doc-medium DPA-2 without 3-body | `08:13:06` |

> The DPA-2 `efv_plots__*.png` shows training-only loss columns — these runs
> have no `validation_data` block in `input.json`, same as the `se_e2_a`
> family. For an apples-to-apples DPA-2-vs-`se_e2_a` performance comparison,
> a `dp --pt test` run on the new 6,100-frame `MgSiO3/sim_data` collection
> is needed; see [`../validation_MgSiO3_sim_data_all_retained__20260523/`](../validation_MgSiO3_sim_data_all_retained__20260523/) and
> [`../../../ONGOING/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_training__20260518.md`](../../../ONGOING/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_training__20260518.md).

## Not included

The 4 partial/failed `se_e2_a` runs (`fit_deep5x` NaN at step 283,900; `desc_deep2x` / `desc_deep10x` / `both_deep10x` TIMEOUT before 1 M steps) are excluded because they're not in the validated set. If desired, the same script can be run in their dirs to generate truncated learning curves.
