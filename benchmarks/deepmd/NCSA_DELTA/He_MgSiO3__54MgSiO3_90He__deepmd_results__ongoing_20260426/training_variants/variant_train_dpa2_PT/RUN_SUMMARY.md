# `variant_train_dpa2_PT`

## Curated Result

- Status: partial; the run reached the target step in `lcurve.out`, but the raw log did
  not contain a finished-training marker.
- Backend / descriptor: PyTorch / `dpa2`.
- Target steps: `1000000`.
- Last logged step: `1000000` (`100.0%`).
- Average training time: `0.1011 s/batch`.
- Descriptor setup: `repinit=[25,50,100]; nsel=120/30; layers=6; g=64/32`.
- Fitting net: `[240,240,240]`.
- Systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `3.49`, best `0.504`, last-100 median `3.71`,
  last-100 p10/p90 `2.2` / `4.68`.

## Lesson

This baseline DPA-2 PyTorch run is a training-throughput reference, not a validated
production model. The force-error trace is much worse than the `se_e2_a` variants in
this benchmark family, so it mainly informs DPA-2 setup/runtime behavior.

Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, and
generated `out.json` were summarized here and removed from the benchmark folder.

