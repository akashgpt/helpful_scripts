# `variant_train_se_e2_a_PT`

## Curated Result

- Status: partial; the run reached the target step in `lcurve.out`, but the raw log did
  not contain a finished-training marker.
- Backend / descriptor: PyTorch / `se_e2_a`.
- Target steps: `1000000`.
- Last logged step: `1000000` (`100.0%`).
- Average training time: `0.1024 s/batch`.
- Parameter count: `2.665665 M`.
- Descriptor neurons: `[25,50,100]`.
- Fitting net: `[240,240,240]`.
- Systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `0.598`, best `0.088`, last-100 median `0.352`,
  last-100 p10/p90 `0.259` / `0.598`.

## Lesson

The PyTorch `se_e2_a` baseline reached the planned step count, but was much slower per
batch than the TensorFlow `se_e2_a` baseline in this benchmark family. Its force-error
signal is far better than the DPA-2 PyTorch runs here.

Raw `log.train`, `lcurve.out`, `slurm-*`, `dp_test.log`, generated
`input_v2_compat.json`, and generated `out.json` were summarized here and removed from
the benchmark folder.

