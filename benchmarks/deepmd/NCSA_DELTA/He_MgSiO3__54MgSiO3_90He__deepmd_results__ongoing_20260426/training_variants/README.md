# Training Variants

This folder records the DeePMD training-variant benchmarks for the
`54MgSiO3_90He` system on NCSA Delta.

The curated record is:

- `TRAINING_VARIANTS_SUMMARY.tsv`: distilled run status, timing, model size, and
  late-training force-error statistics extracted from the raw training logs.
- `*/input.json`: the original DeepMD training input for each variant.
- `*/sub.sh`: the submission script used for each variant.

Raw `log.train`, `lcurve.out`, `slurm-*`, `dp_test.log`, generated
`input_v2_compat.json`, and generated `out.json` files were intentionally
removed from this benchmark folder. Useful information from those files belongs
in compact summaries like `TRAINING_VARIANTS_SUMMARY.tsv`, not as copied raw
runtime output.

