# `variant_train_dpa2_PT_doc_medium_auto_currentloss_200k`

## Curated Result

- Submitted: 2026-05-18.
- Slurm job id: `18320019`.
- Status at submission check: `PENDING` on `gpuA100x4`, reason `Priority`.
- Steps: `200000`.
- Backend / descriptor: PyTorch / DPA-2.
- Purpose: test an official-medium-style DPA-2 descriptor with `auto:1.1` neighbor
  selections, three-body `repinit`, smoother repformer cutoff, residual update style,
  gradient clipping, and the same energy/force/virial loss used by prior benchmarks.
- Main hypothesis: if the earlier bad DPA-2 force behavior came from a poor descriptor
  recipe rather than the loss function, this run should show better early force-RMSE
  descent while keeping the same loss weighting as the previous `se_e2_a` comparisons.

Only `input.json`, `sub.sh`, and this summary are archived here. Raw `log.train`,
`lcurve.out`, Slurm output, checkpoints, and generated artifacts belong in the live run
directory until distilled.

