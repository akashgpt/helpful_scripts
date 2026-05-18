# `variant_train_se_e2_a_TF`

## Curated Result

- Status: complete; finished-training marker was present.
- Backend / descriptor: TensorFlow / `se_e2_a`.
- Target steps: `1000000`.
- Last logged step: `1000000` (`100.0%`).
- Average training time: `0.0386 s/batch`.
- Wall time: `39474.074 s`.
- Parameter count: `2.665665 M`.
- Descriptor neurons: `[25,50,100]`.
- Fitting net: `[240,240,240]`.
- Systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `0.36`, best `0.108`, last-100 median `0.3485`,
  last-100 p10/p90 `0.213` / `0.608`.

## Lesson

This is the base TensorFlow `se_e2_a` reference. It trained much faster than the PyTorch
`se_e2_a` run for the same architecture and is the baseline used for later TensorFlow
width/depth scaling comparisons.

Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, and
generated `out.json` were summarized here and removed from the benchmark folder.

