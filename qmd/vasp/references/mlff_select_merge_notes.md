# VASP MLFF `ML_AB` Merge Note

This note records the current unresolved behavior seen while testing the
two-phase `H2O_NH3/250_H2O__256_NH3` setup under
`/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/250_H2O__256_NH3`.

## What Was Validated

- A plain stitched two-phase `POSCAR` plus matching `POTCAR` ordering is fine.
- `ML_MODE = select` itself works in this environment when using a native
	(single-source, unmerged) VASP `ML_AB`.
- A control run using the pure-H2O `ML_AB` in
	`select_mode_control_h2o/` completed machine-learning initialization,
	selected local references, and entered the MD loop successfully.

## What Failed

Hand-merged `ML_AB` files are not yet reliable in this workflow.

Observed failure mode:

- VASP reaches `- Scanning ML_AB file`
- then exits with:
	`forrtl: severe (59): list-directed I/O syntax error`

This happened for:

- merged `H2O + NH3`
- merged `H2O + H2O` control
- merged files built from both `ML_ABN` and the more stable source `ML_AB`

So the current blocker is not just mixed chemistry. Even the same-chemistry
merged control still failed during `ML_AB` parsing.

## Practical Conclusion

Treat these as separate states:

- stitched-structure setup: validated
- `POSCAR`/`POTCAR` order checking: validated
- hand-merged `ML_AB` plus `ML_MODE = select`: unresolved

If this is revisited later, start from the working single-source
`select_mode_control_h2o/` case and compare the exact merged `ML_AB`
format against that known-good file.
