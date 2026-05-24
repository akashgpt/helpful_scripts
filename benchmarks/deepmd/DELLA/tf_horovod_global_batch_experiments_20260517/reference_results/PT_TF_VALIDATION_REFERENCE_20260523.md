# PT/TF Validation Reference Tables

Date: 2026-05-23

These compact reference tables summarize the Della NH3/H2 DeePMD TF/Horovod
and PT benchmark validation runs. The first columns are intentionally ordered
for model selection:

```text
framework	name	checkpoint_step	final_total_steps	best_TRAIN_total_loss_step	TRAIN_total_loss	TRAIN_E_RMSE_eV_per_atom	VAL_E_RMSE_eV_per_atom	TRAIN_F_RMSE_eV_per_A	VAL_F_RMSE_eV_per_A
```

Extra columns record checkpoint label, group/status/GPU count, runtimes,
virial metrics, validation frame counts, unique training-frame counts when
available, and scratch provenance paths.

Files:

- `PT_TF_VALIDATION_REFERENCE_20260523.tsv`: all TF and PT rows, sorted by
  validation energy RMSE per atom.
- `PT_VALIDATION_REFERENCE_20260523.tsv`: PT-only slice.
- `TF_VALIDATION_REFERENCE_20260523.tsv`: TF-only slice.

Row counts:

- all: 51
- PT: 16
- TF: 35

Main caution: the 10x-labelled TF runs here use the same ~13.6k-frame training
pool as the 100k-none runs; the label reflects training schedule/steps, not
10x more distinct frames.
