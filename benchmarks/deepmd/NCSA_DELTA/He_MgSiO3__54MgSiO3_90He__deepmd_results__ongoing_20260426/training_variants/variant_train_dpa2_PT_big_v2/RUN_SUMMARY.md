# `variant_train_dpa2_PT_big_v2`

## Curated Result

- Status: failed.
- Failure reason: PyTorch dataloader bus error / shared-memory pressure.
- Backend / descriptor: PyTorch / `dpa2`.
- Target steps: `1000000`.
- Last logged step: `556100` (`55.61%`).
- Descriptor setup: `repinit=[75,150,300]; nsel=360/120; layers=6; g=192/96`.
- Fitting net: `[720,720,720]`.
- Systems: `255`.
- L-curve rows: `5562`.
- Force RMSE signal before failure: final `2.85`, best `1.05`, last-100 median `3.76`,
  last-100 p10/p90 `2.27` / `5.07`.

## Lesson

The larger corrected DPA-2 PyTorch case exposed dataloader/shared-memory fragility before
completion. It is useful as a resource-failure record for future DPA-2 setup, but it
should not be used to rank model quality.

Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, and
generated `out.json` were summarized here and removed from the benchmark folder.

