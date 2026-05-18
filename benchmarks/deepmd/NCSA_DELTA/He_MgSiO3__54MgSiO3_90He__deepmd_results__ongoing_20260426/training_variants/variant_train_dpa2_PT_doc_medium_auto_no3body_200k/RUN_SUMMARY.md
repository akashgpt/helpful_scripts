# `variant_train_dpa2_PT_doc_medium_auto_no3body_200k`

## Curated Result

- Submitted: 2026-05-18.
- Slurm job id: `18320017`.
- Status at submission check: `PENDING` on `gpuA100x4`, reason `Priority`.
- Steps: `200000`.
- Backend / descriptor: PyTorch / DPA-2.
- Purpose: test the official-medium-style DPA-2 update/smoothing/auto-selection recipe
  while removing the three-body `repinit` block.
- Main hypothesis: this isolates the cost/benefit of the three-body term. If this run
  behaves like the current-loss official-style run, three-body may not be necessary for
  this MgSiOH diagnostic. If it regresses, three-body is likely important for the
  corrected DPA-2 setup.

Only `input.json`, `sub.sh`, and this summary are archived here. Raw `log.train`,
`lcurve.out`, Slurm output, checkpoints, and generated artifacts belong in the live run
directory until distilled.

