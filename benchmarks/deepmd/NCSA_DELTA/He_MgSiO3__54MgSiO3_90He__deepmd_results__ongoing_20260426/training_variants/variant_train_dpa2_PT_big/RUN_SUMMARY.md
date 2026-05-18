# `variant_train_dpa2_PT_big`

## Curated Result

- Status: partial; the run reached the target step in `lcurve.out`, but the raw log did
  not contain a finished-training marker.
- Backend / descriptor: PyTorch / `dpa2`.
- Target steps: `1000000`.
- Last logged step: `1000000` (`100.0%`).
- Average training time: `0.0991 s/batch`.
- Descriptor setup: `repinit=[75,150,300]; nsel=120/30; layers=6; g=192/96`.
- Fitting net: `[720,720,720]`.
- Systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `3.49`, best `0.504`, last-100 median `3.71`,
  last-100 p10/p90 `2.2` / `4.68`.

## Lesson

The larger DPA-2 PyTorch settings did not improve the logged force-error signal in this
benchmark record. Runtime was comparable to the smaller DPA-2 baseline in this
small-batch setup, so this is mainly useful as a width-scaling setup/runtime reference.

Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, and
generated `out.json` were summarized here and removed from the benchmark folder.

