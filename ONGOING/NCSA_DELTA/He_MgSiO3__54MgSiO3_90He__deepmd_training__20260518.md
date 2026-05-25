# ONGOING — He_MgSiO3 / MgSiOH DeePMD training diagnostics (NCSA Delta, gpuA100x4)

**Started tracking:** 2026-05-18
**Cluster:** NCSA Delta · partition `gpuA100x4` · account `bguf-delta-gpu`
**Working dir:** `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench`
**Benchmark record:** `/projects/bguf/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426`
**Status as of 2026-05-23:** all 4 `se_e2_a` intermediates trained + held-out validated (`balanced_5x` H200 validation finished 2026-05-23, curated TSV updated) · compressed-validation finished except original `big` compression failure · 4 DPA-2 diagnostics all COMPLETED CLEAN at 200k steps (held-out validation plan-only, pending user input on test set) · 1 NaN failure (`fit_deep5x`)
**Current validation table:** `benchmarks/deepmd/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426/compressed_validation_20260518/VALIDATION_RMSE_COMPARISON__71MgSiO3_5He__20260518.tsv`

---

## What is running / pending

### `se_e2_a` intermediate scaling runs

These are the TensorFlow `se_e2_a` intermediate width/depth tests requested after the
initial energy-focused validation on held-out `71MgSiO3_5He`.

All four trainings have completed and have held-out validation rows in the curated
comparison TSV:

| Job ID | Name | Variant | Train wall time | Validation job | Validated 2026 | Validation partition |
| --- | --- | --- | --- | --- | --- | --- |
| `18313921` | `se_big2x` | `variant_train_se_e2_a_TF_big2x` | `13:31:10` | `18324408_0` | 05-18 | gpuA100x4 |
| `18313922` | `se_bal2x` | `variant_train_se_e2_a_TF_balanced_2x` | `15:29:47` | `18324408_1` | 05-18 | gpuA100x4 |
| `18313923` | `se_big5x` | `variant_train_se_e2_a_TF_big5x` | `17:26:08` | `18324408_2` | 05-18 | gpuA100x4 |
| `18313924` | `se_bal5x` | `variant_train_se_e2_a_TF_balanced_5x` | `1-03:35:33` (27:35:33) | `18421078_4` | 05-23 | gpuH200x8 |

Completed at the scheduler level but failed scientifically:

| Job ID | Name | Variant | State | Wall time | Failure |
| --- | --- | --- | --- | --- | --- |
| `18313925` | `se_fit5x` | `variant_train_se_e2_a_TF_fit_deep5x` | COMPLETED | `05:30:40` | DeePMD NaN at batch `284000`; last finite `lcurve.out` row is batch `283900`. |

Purpose:

- Decide whether cheaper `2x`/`5x` width or balanced models recover most of the
  `balanced_10x` held-out energy advantage.
- Main metric remains held-out energy RMSE/atom on `CONFIG=71MgSiO3_5He`.
- Existing benchmark notes live under:
  `.../deepmd_results__ongoing_20260426/intermediate_scaling_20260517`.

Read of the ranking after `balanced_5x` was added (curated TSV
`compressed_validation_20260518/VALIDATION_RMSE_COMPARISON__71MgSiO3_5He__20260518.tsv`):

- `balanced_10x` (26.80 M params) remains best on held-out energy RMSE/atom
  (`0.060336`).
- `both_deep2x` (5.38 M params) is the best intermediate compressed result
  (`0.125400`).
- `balanced_5x` (13.69 M params) lands at `0.242313` — between `fit_deep2x`
  (`0.241183` @ 5.28 M) and `balanced_2x` (`0.256734` @ 5.22 M). Balanced 5x
  scaling is not a sweet spot: at 2.6× the parameters of `fit_deep2x` it
  produces essentially the same held-out energy error, and the much smaller
  `both_deep2x` dominates it.
- `balanced_5x` does post the second-best held-out force RMSE in the table
  (`1.728125`), only `balanced_10x` is comparable (`1.729089`).

### DPA-2 diagnostic suite on `MgSiOH__R2SCAN`

These are short 200k-step PyTorch DPA-2 diagnostics submitted after checking DeePMD
DPA-2 documentation and the DPA-2 literature. They intentionally use the same
`MgSiOH__R2SCAN` dataset as the earlier speed/setup benchmark, not the He/MgSiO3
validation split.

| Job ID | Name | Variant | State | Wall time | Purpose |
| --- | --- | --- | --- | --- | --- |
| `18320018` | `dpa2_cur_auto` | `variant_train_dpa2_PT_current_auto_200k` | COMPLETED | `08:53:34` | old DPA-2 architecture with `auto:1.1` neighbor selection |
| `18320019` | `dpa2_doc_cur` | `variant_train_dpa2_PT_doc_medium_auto_currentloss_200k` | COMPLETED | `09:36:21` | official-medium-style DPA-2 with current energy/force/virial loss |
| `18320016` | `dpa2_doc_nov` | `variant_train_dpa2_PT_doc_medium_auto_novirial_200k` | COMPLETED | `09:37:24` | official-medium-style DPA-2 with no virial loss |
| `18320017` | `dpa2_doc_no3` | `variant_train_dpa2_PT_doc_medium_auto_no3body_200k` | COMPLETED | `08:13:06` | official-medium-style DPA-2 without the three-body block |

All four finished cleanly at 200k steps with `model.ckpt-200000.pt` saved on 2026-05-22. Curated startup/health summary (verdict CLEAN for all four) written 2026-05-19:

- `training_bench/DPA2_STARTUP_HEALTH_20260519.md`
- `training_bench/DPA2_STARTUP_HEALTH_20260519.tsv`
- `training_bench/DPA2_HELDOUT_VALIDATION_PLAN_20260519.md` (plan-only; no Slurm submitted)

Key startup findings: only `current_auto` and `no3body` emit the harmless `sel of type 0 is not enough` 3-body nsel warning (those variants have `use_three_body: false`, so the under-sized 40 is never used; the two enabled variants auto-resolve to `120 >= 107` and emit no warning). No NaN / inf / Bus error / CUDA OOM / Traceback in any log. Final training force RMSE 2.24-2.76e-01 eV/A; throughput 348-407 steps/min depending on variant.

No held-out validation was executed: there is no separate MgSiOH test split (the input.json files only define `training_data`), so the choice of test set (HeMgSiO3 systems vs untrained MgSiOH subset) needs user input before submitting the planned Slurm validation job.

Purpose:

- Test whether the previous poor DPA-2 behavior was due to neighbor under-selection,
  non-official descriptor settings, virial-loss coupling, missing three-body terms, or
  PyTorch data-path fragility.
- First checks after startup:
  - No `sel is not enough` warnings in `log.train`.
  - `lcurve.out` force RMSE descends normally during the first few thousand steps.
  - No PyTorch `Bus error`, shared-memory, or filesystem I/O failure.

Curated benchmark records:

- `training_variants/DPA2_DIAGNOSIS.md`
- `training_variants/DPA2_DIAGNOSTIC_SUITE_20260518.tsv`
- `training_variants/reference_scripts/setup_dpa2_diagnostic_suite_20260518.py`

### Compressed validation on held-out `71MgSiO3_5He`

Submitted compressed freeze/compress/test validation arrays:

| Job ID | Script | Scope | Final state | Resources |
| --- | --- | --- | --- | --- |
| `18323754` | `validation_71MgSiO3_5He__v1_i_train_test__20260517/run_validation.sbatch` | original 5 cases | FAILED during `dp compress` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18323924` | `validation_71MgSiO3_5He__compressed_intermediates__20260518/run_validation.sbatch` | intermediate cases | CANCELED after hold; pre-fix submission | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324077` | `validation_71MgSiO3_5He__v1_i_train_test__20260517/run_validation.sbatch` | original 5 cases | FAILED during `dp compress` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324078` | `validation_71MgSiO3_5He__compressed_intermediates__20260518/run_validation.sbatch` | `big2x`, `balanced_2x`, `big5x`, `both_deep2x` | FAILED during `dp compress` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324348` | `validation_71MgSiO3_5He__v1_i_train_test__20260517/run_validation.sbatch` | original 5 cases | FAILED/CANCELED after no-`-t` Apptainer test still hit `loss_func` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324349` | `validation_71MgSiO3_5He__compressed_intermediates__20260518/run_validation.sbatch` | `big2x`, `balanced_2x`, `big5x`, `both_deep2x` | FAILED after no-`-t` Apptainer test still hit `loss_func` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324377` | `validation_71MgSiO3_5He__v1_i_train_test__20260517/run_validation.sbatch` | original 5 cases | FAILED before DeePMD: conda activation hit `set -u` / unset `INCLUDE` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324378` | `validation_71MgSiO3_5He__compressed_intermediates__20260518/run_validation.sbatch` | `big2x`, `balanced_2x`, `big5x`, `both_deep2x` | FAILED before DeePMD: conda activation hit `set -u` / unset `INCLUDE` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324409` | `validation_71MgSiO3_5He__v1_i_train_test__20260517/run_validation.sbatch` | original 5 cases | `base`, `fit_deep2x`, `fit_deep10x` COMPLETED; `big` FAILED during compression; `balanced_10x` timed out during `dp test` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324408` | `validation_71MgSiO3_5He__compressed_intermediates__20260518/run_validation.sbatch` | `big2x`, `balanced_2x`, `big5x`, `both_deep2x` | COMPLETED | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324747` | `validation_71MgSiO3_5He__v1_i_train_test__20260517/run_validation_balanced10x_1h.sbatch` | `balanced_10x` only | COMPLETED in `15:38` after `18324409_4` exceeded 15 min during `dp test` | 1 A100 GPU, 2 CPUs, 80G, 1h |

The helper scripts first tested the ALCHEMY-style model-prep sequence:
`dp freeze -o pv.pb`, then `dp compress -i pv.pb -o pv_comp.pb`, then
`dp test -m pv_comp.pb`. Jobs `18324348` and `18324349` still failed with the
same `loss_func` strict-parser error when run through the older Apptainer image,
so remaining tasks were canceled.

The current working diagnosis is an environment mismatch:

- Successful qmd_data compression trained and compressed with the Apptainer image
  reporting DeePMD `v3.1.3` commit `b2c8511e`; those generated `out.json` files
  contain no `loss_func`.
- These benchmark TF variants were trained with `ALCHEMY_env`, reporting DeePMD
  `v3.1.3-29-gefc27cf7`; their generated `out.json` files contain
  `loss.loss_func`.
- Compression with the older Apptainer image rejects the newer checkpoint/frozen
  graph metadata. The helper scripts have therefore been patched to load the same
  module stack and `conda activate ALCHEMY_env`, then run `dp --tf freeze`,
  `dp --tf compress`, and `dp --tf test`.

Current failure:

- First failure mode: `dp compress` rejected legacy `loss.loss_func` in the
  training JSON.
- Follow-up patch generates `myinput.compress.json` without `loss_func`, and those
  files are clean on disk.
- The resubmitted arrays still failed with the same `loss_func` strict-parser error.
  This likely means `dp compress` is reading training-script metadata embedded in the
  frozen `pv.pb` files, or freeze is embedding normalized metadata with `loss_func`.
- No compressed validation numbers are available from these arrays.
- Follow-up diagnosis showed that successful ALCHEMY He/MgSiO3 training compression
  does not use `-t`; it runs native `dp compress -i pv.pb -o pv_comp.pb` inside the
  training `model-compression` directory. Jobs `18324348` and `18324349` tested
  that corrected path but still used the wrong DeePMD environment.
- New `big` failure: with the training-consistent `ALCHEMY_env`, `big` gets past
  the old `loss_func` parser issue but fails while TensorFlow exports the
  compressed checkpoint meta graph:
  `google.protobuf.message.DecodeError: Error parsing message with type 'tensorflow.GraphDef'`.
  The compressed data and index files are written, but no `model.ckpt.meta` is
  produced under `model-compression/big/model-compression`. This is consistent
  with the original `big` compressed graph crossing a TensorFlow/Protobuf
  GraphDef serialization size limit. For comparison, `big5x` succeeds with a
  compressed `model.ckpt.meta` of about `1.8G`, while original `big` is wider
  (`descrpt [75,150,300]`, fitting `[720,720,720]`) and fails before emitting
  its `.meta`.
- The `big` rows in `VALIDATION_SUMMARY.tsv` are stale from prior validation
  logs and should not be treated as compressed-validation results from
  `18324409`; compressed `big` never reached `dp test`. It is still included
  as a non-compressed reference row in the curated comparison TSV:
  `compressed_validation_20260518/VALIDATION_RMSE_COMPARISON__71MgSiO3_5He__20260518.tsv`.
- The curated comparison TSV now includes `parameter_count`. Completed compressed
  rows are from `freeze` + `compress` + `dp test`; original `big` is comparison-only
  because compressed export failed with a TensorFlow GraphDef/meta-graph decode error.
- Training RMSE columns in the curated table are last-1000-step `lcurve.out` averages
  over steps `999000-1000000` (`11` logged rows). `train_total_rmse` is a DeePMD
  loss-scale value, not a physical-unit error metric.
- Current energy-RMSE read from the curated table: `balanced_10x` is best among
  completed compressed validations (`0.060336` energy RMSE/atom), while
  `both_deep2x` is the best completed intermediate compressed result (`0.125400`).

Check and summarize:

```bash
sacct -j 18323754,18323924,18324077,18324078,18324348,18324349,18324377,18324378,18324408,18324409,18324747 --format=JobID,JobName%28,State,Elapsed,ExitCode -P
squeue -j 18324747 -o "%i %T %j %P %D %C %b %M %R"
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_71MgSiO3_5He__v1_i_train_test__20260517
tail -80 results/base/log.compress
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/validation_71MgSiO3_5He__compressed_intermediates__20260518
tail -80 results/big2x/log.compress
```

---

## How to check

```bash
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench

squeue -j 18313922,18313923,18313924,18320016,18320017,18320018,18320019 \
  -o "%i %T %j %P %D %C %b %M %R"

sacct -j 18313921,18313922,18313923,18313924,18313925,18320016,18320017,18320018,18320019 \
  --format=JobID,JobName%28,State,Elapsed,AllocTRES%50,ExitCode -P
```

Early DPA-2 diagnostics:

```bash
for d in variant_train_dpa2_PT_current_auto_200k \
         variant_train_dpa2_PT_doc_medium_auto_currentloss_200k \
         variant_train_dpa2_PT_doc_medium_auto_novirial_200k \
         variant_train_dpa2_PT_doc_medium_auto_no3body_200k; do
  echo "### $d"
  grep -E "sel of type|maximum neighbor size|Bus error|Input/output error|Batch" "$d/log.train" 2>/dev/null | tail -20
  tail -5 "$d/lcurve.out" 2>/dev/null
done
```

Intermediate `se_e2_a` progress:

```bash
for d in variant_train_se_e2_a_TF_big2x \
         variant_train_se_e2_a_TF_balanced_2x \
         variant_train_se_e2_a_TF_big5x \
         variant_train_se_e2_a_TF_balanced_5x \
         variant_train_se_e2_a_TF_fit_deep5x; do
  echo "### $d"
  tail -5 "$d/lcurve.out" 2>/dev/null
done
```

## Done definition / next steps

- For `se_e2_a` intermediates:
  - DONE: `balanced_5x` finished training (job `18313924`, `1-03:35:33`) and held-out
    validated on H200 (job `18421078_4`, `00:12:29`); curated TSV + README updated
    2026-05-23 with parameter count `13,690,655`.
  - DONE: held-out validation rows for `big2x`, `balanced_2x`, `big5x`, `both_deep2x`,
    `balanced_5x` plus the earlier `base`, `fit_deep2x`, `fit_deep10x`, `balanced_10x`
    are all in the curated comparison TSV.
  - Remaining: treat `fit_deep5x` as failed unless intentionally resuming from a
    pre-NaN checkpoint with a lower/stabilized learning-rate schedule. The current
    table is sufficient to conclude the `se_e2_a` scaling sweep on this dataset.
  - Open: original `big` (23.91 M) compression failure (TF GraphDef serialization
    limit during compressed meta-graph export) is unresolved; row remains in the
    curated TSV as a non-compressed reference. Fix would require either reducing
    `descrpt`/`fitting` widths so `model.ckpt.meta` stays under the protobuf size
    limit, or a code-level workaround in the compress path.
- For DPA-2 diagnostics:
  - DONE: startup-log inspection for all four 200k-step variants (verdict CLEAN);
    curated `training_bench/DPA2_STARTUP_HEALTH_20260519.{md,tsv}`.
  - PENDING USER INPUT: held-out validation plan
    (`training_bench/DPA2_HELDOUT_VALIDATION_PLAN_20260519.md`) — needs decision on
    test set (HeMgSiO3 OOD vs MgSiOH in-distribution holdout) before submitting the
    one-shot Slurm validation job described in that plan.
- Keep the benchmark folder curated: summarize results into markdown/TSV and keep setup
  scripts/JSONs, but do not copy raw `log.train`, `slurm-*`, checkpoints, or model files.
