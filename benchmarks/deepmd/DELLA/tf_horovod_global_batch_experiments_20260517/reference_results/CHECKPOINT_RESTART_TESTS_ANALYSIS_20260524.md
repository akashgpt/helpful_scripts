# Checkpoint Restart Tests With Learning Rate Panel

Date: 2026-05-24

These plots compare the 4GPU restart-chain diagnostics, including the learning
rate as an explicit panel. They are optimizer/restart-health diagnostics, not
validation-ranking tables.

Artifacts:

```text
CHECKPOINT_RESTART_TESTS_TRAINING_EVOLUTION_ROLLING_MEAN_20260524.png
CHECKPOINT_RESTART_TESTS_TRAINING_EVOLUTION_ROLLING_MEAN_FIRST20K_20260524.png
CHECKPOINT_RESTART_TESTS_PLOTTED_20260524.tsv
```

Generator:

```text
../reference_scripts/plot_checkpoint_restart_tests_with_lr.py
```

## Main Conclusions

- TF restart behavior is usable in the tested 4GPU `none` case. The final
  template test reached 100000 steps over repeated 15-minute slices, printed
  DeePMD `finished training`, and produced `model-compression/pv.pb` plus
  `model-compression/pv_comp.pb`.
- The TF 10-minute healthgate + second-latest-checkpoint diagnostic also
  reached 100000 steps cleanly, but that scratch diagnostic did not run
  freeze/compress. Use the final-template test as the production-style
  finalization reference.
- PT restart behavior failed relative to the clean 4GPU PT baseline. The three
  PT restart-chain variants reached or approached completion, but their final
  total training loss stayed much higher than the baseline. The clean v2
  healthgate + second-latest run reached 100000 steps and finalized, but still
  ended at total `7.51`.
- The PT v1 healthgate chain is explicitly contaminated by an old checkpoint
  selector bug: it reached about step `19980` but restarted from
  `model.ckpt-9800.pt`. Keep it in plots only as a failure-mode reference.
- PT checkpointing in these restart tests produced many checkpoints, not just
  one: the clean v2 input used `save_freq = 100`, `save_ckpt =
  model-compression/model.ckpt`, and produced checkpoints from
  `model.ckpt-100.pt` through `model.ckpt-100000.pt`.
- Follow-up inspection on 2026-05-25 found a specific PT restart learning-rate
  scheduler problem. Raw `lcurve.out` and checkpoint optimizer states show that
  later PT restarts resume with an inflated LR. The active DeePMD PT trainer
  appears to double-count the restart step when rebuilding the PyTorch
  `LambdaLR` scheduler. See:

```text
PT_RESTART_LR_SCHEDULER_DIAGNOSTIC_20260525.md
```

## Practical Guidance

- For TF 4GPU `none`, the checkpoint-restart path with numeric checkpoint
  selection, second-latest restart selection, rollback guard, and health gate is
  a reasonable production reference.
- For PT, do not trust chained restart/continuation yet for production. Prefer
  uninterrupted PT runs until the scheduler restart behavior is patched and
  validated.
- Keep multiple PT checkpoints if experimenting with PT restart logic. The
  second-latest strategy and rollback guard require more than one saved
  checkpoint to provide any protection beyond "restart from the only checkpoint
  available."
