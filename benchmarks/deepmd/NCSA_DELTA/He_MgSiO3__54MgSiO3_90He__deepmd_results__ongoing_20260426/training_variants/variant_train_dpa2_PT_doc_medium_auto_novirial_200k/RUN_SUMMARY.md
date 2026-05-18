# `variant_train_dpa2_PT_doc_medium_auto_novirial_200k`

## Curated Result

- Submitted: 2026-05-18.
- Slurm job id: `18320016`.
- Status at submission check: `PENDING` on `gpuA100x4`, reason `Priority`.
- Steps: `200000`.
- Backend / descriptor: PyTorch / DPA-2.
- Purpose: test the same official-medium-style DPA-2 descriptor as the current-loss run,
  but with an official-example-style energy/force loss and no virial loss.
- Main hypothesis: if virial weighting was interfering with the from-scratch DPA-2 force
  learning on this dataset, this run should show cleaner early force-RMSE descent.

Only `input.json`, `sub.sh`, and this summary are archived here. Raw `log.train`,
`lcurve.out`, Slurm output, checkpoints, and generated artifacts belong in the live run
directory until distilled.

