# `variant_train_dpa2_PT_big_strip`

## Curated Result

- Status: partial; the run reached the target step in `lcurve.out`, but the raw log did
  not contain a finished-training marker.
- Backend / descriptor: PyTorch / `dpa2`.
- Target steps: `1000000`.
- Last logged step: `1000000` (`100.0%`).
- Average training time: `0.1028 s/batch`.
- Descriptor setup: `repinit=[75,150,300]; nsel=120/30; layers=6; g=192/96`.
- Fitting net: `[720,720,720]`.
- Systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `3.49`, best `0.504`, last-100 median `3.71`,
  last-100 p10/p90 `2.2` / `4.68`.

## Lesson

The stripped larger DPA-2 setup behaved like the other completed DPA-2 PyTorch variants
in the logged force-error summary. It remains a useful setup record, but not evidence of
better predictive quality.

Raw `log.train`, `lcurve.out`, `slurm-*`, `dp_test.log`, generated
`input_v2_compat.json`, and generated `out.json` were summarized here and removed from
the benchmark folder.

