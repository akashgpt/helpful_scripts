# `variant_train_dpa2_PT_v2`

## Curated Result

- Status: failed.
- Failure reason: PyTorch dataloader bus error / shared-memory pressure.
- Backend / descriptor: PyTorch / `dpa2`.
- Target steps: `1000000`.
- Last logged step: `972800` (`97.28%`).
- Descriptor setup: `repinit=[25,50,100]; nsel=360/120; layers=6; g=64/32`.
- Fitting net: `[240,240,240]`.
- Systems: `255`.
- L-curve rows: `9729`.
- Force RMSE signal before failure: final `3.29`, best `0.504`, last-100 median `3.555`,
  last-100 p10/p90 `2.23` / `4.6`.

## Lesson

The corrected baseline DPA-2 selection case almost reached the target length but failed
from PyTorch dataloader/shared-memory pressure. This points to resource handling rather
than a scientific stopping criterion.

Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, and
generated `out.json` were summarized here and removed from the benchmark folder.

