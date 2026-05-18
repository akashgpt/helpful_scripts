# Compressed Validation, 2026-05-18

This folder records the curated `71MgSiO3_5He` validation comparison for the
`He_MgSiO3__54MgSiO3_90He` DeePMD benchmark family.

## Files

- `VALIDATION_RMSE_COMPARISON__71MgSiO3_5He__20260518.tsv`
  Current comparison table. The table now includes `parameter_count`.

Raw Slurm logs, full DeePMD logs, checkpoints, frozen graphs, and compressed model
artifacts are intentionally not archived here.

## Comparison

Validation metrics are from `freeze` + `compress` + `dp test` where compression
completed. The `big` row is kept only as a noncompressed reference because compressed
export failed before `dp test` with a TensorFlow GraphDef/meta-graph decode error.

Training RMSEs are last-1000-step `lcurve.out` averages over steps `999000-1000000`
(`11` logged rows). `train_total_rmse` is DeePMD loss-scale output, not a physical-unit
error metric; use energy/force/virial RMSE columns for physical comparisons.

| Model | Parameters | Mode | Energy RMSE/atom | Force RMSE | Virial RMSE/atom | Note |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `balanced_10x` | 26,804,305 | compressed | 0.060336 | 1.729089 | 0.059074 | best compressed held-out energy RMSE |
| `big` | 23,910,455 | noncompressed reference | 0.102139 | 1.777232 | 0.106930 | compressed export failed; comparison only |
| `both_deep2x` | 5,380,155 | compressed | 0.125400 | 1.767890 | 0.107342 | best completed intermediate compressed result |
| `big2x` | 5,289,155 | compressed | 0.222238 | 1.884397 | 0.117651 | width-only 2x |
| `fit_deep2x` | 5,279,255 | compressed | 0.241183 | 1.929167 | 0.130265 | fitting-depth 2x |
| `balanced_2x` | 5,216,405 | compressed | 0.256734 | 1.810257 | 0.118384 | balanced 2x |
| `fit_deep10x` | 26,768,855 | compressed | 0.266868 | 1.780698 | 0.110708 | fitting-depth 10x |
| `big5x` | 13,406,905 | compressed | 0.319975 | 1.928073 | 0.132518 | width-only 5x |
| `base` | 2,665,655 | compressed | 0.383706 | 1.974220 | 0.204557 | baseline |

Main read: `balanced_10x` remains the best completed compressed model on held-out
energy RMSE/atom. `both_deep2x` is the best completed intermediate compressed result
in this table, but it does not recover the `balanced_10x` energy advantage.
