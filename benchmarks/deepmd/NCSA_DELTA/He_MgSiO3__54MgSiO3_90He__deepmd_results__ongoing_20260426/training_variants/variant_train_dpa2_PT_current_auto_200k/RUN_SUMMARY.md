# `variant_train_dpa2_PT_current_auto_200k`

## Curated Result

- Submitted: 2026-05-18.
- Slurm job id: `18320018`.
- Status at submission check: `PENDING` on `gpuA100x4`, reason `Priority`.
- Steps: `200000`.
- Backend / descriptor: PyTorch / DPA-2.
- Purpose: keep the old hand-written DPA-2 architecture but replace fixed neighbor
  selections with `auto:1.1`.
- Main hypothesis: if neighbor under-selection was the dominant problem, this should
  remove `sel is not enough` warnings and improve the early force-RMSE trajectory
  relative to `variant_train_dpa2_PT` and `variant_train_dpa2_PT_v2`.

Only `input.json`, `sub.sh`, and this summary are archived here. Raw `log.train`,
`lcurve.out`, Slurm output, checkpoints, and generated artifacts belong in the live run
directory until distilled.

