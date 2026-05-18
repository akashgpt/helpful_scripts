# ONGOING — He_MgSiO3 / MgSiOH DeePMD training diagnostics (NCSA Delta, gpuA100x4)

**Started tracking:** 2026-05-18
**Cluster:** NCSA Delta · partition `gpuA100x4` · account `bguf-delta-gpu`
**Working dir:** `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench`
**Benchmark record:** `/projects/bguf/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426`
**Status as of 2026-05-18 15:20 CDT:** 1 training RUNNING · compressed-validation arrays RUNNING after conda `nounset` fix · 4 DPA-2 diagnostics PENDING · 3 intermediate trainings cleanly COMPLETED · 1 NaN failure

---

## What is running / pending

### `se_e2_a` intermediate scaling runs

These are the TensorFlow `se_e2_a` intermediate width/depth tests requested after the
initial energy-focused validation on held-out `71MgSiO3_5He`.

| Job ID | Name | Variant | State | Elapsed at check | Resources |
| --- | --- | --- | --- | --- | --- |
| `18313924` | `se_bal5x` | `variant_train_se_e2_a_TF_balanced_5x` | RUNNING | `18:59:08` | 1 A100 GPU, 1 CPU, 64G |

Recently completed and ready for analysis/validation:

| Job ID | Name | Variant | State | Wall time | Resources |
| --- | --- | --- | --- | --- | --- |
| `18313921` | `se_big2x` | `variant_train_se_e2_a_TF_big2x` | COMPLETED | `13:31:10` | 1 A100 GPU, 1 CPU, 48G |
| `18313922` | `se_bal2x` | `variant_train_se_e2_a_TF_balanced_2x` | COMPLETED | `15:29:47` | 1 A100 GPU, 1 CPU, 48G |
| `18313923` | `se_big5x` | `variant_train_se_e2_a_TF_big5x` | COMPLETED | `17:26:08` | 1 A100 GPU, 1 CPU, 64G |

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

### DPA-2 diagnostic suite on `MgSiOH__R2SCAN`

These are short 200k-step PyTorch DPA-2 diagnostics submitted after checking DeePMD
DPA-2 documentation and the DPA-2 literature. They intentionally use the same
`MgSiOH__R2SCAN` dataset as the earlier speed/setup benchmark, not the He/MgSiO3
validation split.

| Job ID | Name | Variant | State | Purpose |
| --- | --- | --- | --- | --- |
| `18320018` | `dpa2_cur_auto` | `variant_train_dpa2_PT_current_auto_200k` | PENDING | old DPA-2 architecture with `auto:1.1` neighbor selection |
| `18320019` | `dpa2_doc_cur` | `variant_train_dpa2_PT_doc_medium_auto_currentloss_200k` | PENDING | official-medium-style DPA-2 with current energy/force/virial loss |
| `18320016` | `dpa2_doc_nov` | `variant_train_dpa2_PT_doc_medium_auto_novirial_200k` | PENDING | official-medium-style DPA-2 with no virial loss |
| `18320017` | `dpa2_doc_no3` | `variant_train_dpa2_PT_doc_medium_auto_no3body_200k` | PENDING | official-medium-style DPA-2 without the three-body block |

All four were still pending with reason `Priority` at the 2026-05-18 14:23 CDT status check. Their run directories contain only `input.json` and `sub.sh`, with no `log.train`, `lcurve.out`, or `slurm-*.out` yet.

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
| `18324409` | `validation_71MgSiO3_5He__v1_i_train_test__20260517/run_validation.sbatch` | original 5 cases | RUNNING; first tasks passed freeze/compress startup and reached `finished compressing` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |
| `18324408` | `validation_71MgSiO3_5He__compressed_intermediates__20260518/run_validation.sbatch` | `big2x`, `balanced_2x`, `big5x`, `both_deep2x` | RUNNING; first tasks passed freeze/compress startup and reached `finished compressing` | 1 A100 GPU, 2 CPUs, 80G, 15 min per array task, max 2 concurrent |

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

Check and summarize:

```bash
sacct -j 18323754,18323924,18324077,18324078,18324348,18324349,18324377,18324378,18324408,18324409 --format=JobID,JobName%28,State,Elapsed,ExitCode -P
squeue -j 18324408,18324409 -o "%i %T %j %P %D %C %b %M %R"
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
  - Wait for `balanced_5x` to complete or fail. At batch `687400`, its own ETA was about 2026-05-18 22:50 CDT.
  - Validate cleanly completed `big2x`, `balanced_2x`, and `big5x` models on the copied `71MgSiO3_5He` held-out train/test sets.
  - Treat `fit_deep5x` as failed unless intentionally resuming from a pre-NaN checkpoint with a lower/stabilized learning-rate schedule.
  - Update the energy RMSE/atom ranking with parameters, wall time, and GPU time.
- For DPA-2 diagnostics:
  - Once jobs start, first inspect startup logs for neighbor-selection warnings.
  - If all official-medium-style diagnostics still show bad force descent, do not spend
    full 1M-step time until the config/data-path issue is understood.
  - If one diagnostic shows normal force descent, extend that setup or run a longer
    version before comparing held-out energy validation.
- Keep the benchmark folder curated: summarize results into markdown/TSV and keep setup
  scripts/JSONs, but do not copy raw `log.train`, `slurm-*`, checkpoints, or model files.
