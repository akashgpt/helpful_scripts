# `variant_train_dpa2_PT_big_strip_v2`

## Curated Result

- Status: failed.
- Failure reason: filesystem input/output error while reading data.
- Backend / descriptor: PyTorch / `dpa2`.
- Target steps: `1000000`.
- Last logged step: `499900` (`49.99%`).
- Descriptor setup: `repinit=[75,150,300]; nsel=360/120; layers=6; g=192/96`.
- Fitting net: `[720,720,720]`.
- Systems: `255`.
- L-curve rows: `5000`.
- Force RMSE signal before failure: final `3.76`, best `1.05`, last-100 median `3.68`,
  last-100 p10/p90 `2.52` / `4.86`.

## Lesson

This corrected/expanded DPA-2 selection test did not complete because the data read path
hit a filesystem I/O error. Treat it as a failed infrastructure/setup run, not as a
model-quality result.

Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, and
generated `out.json` were summarized here and removed from the benchmark folder.

