# Training-Curve Plot Templates

Date: 2026-05-24

This benchmark now keeps the plotting recipes as reusable templates, not just
as one-off figures. For new DeePMD TF/PT benchmarking, start from these scripts
instead of rebuilding the plotting logic from scratch.

## Plain Lcurve Evolution Plots

Use these when each curve comes from one normal training directory with one
``lcurve.out``:

| Plot family | Reference script | Typical output |
|---|---|---|
| TF representative long curves | `reference_scripts/plot_tf_long_training_evolution.py` | `reference_results/TF_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` |
| TF `none` representative curves | `reference_scripts/plot_tf_none_representative_training_evolution.py` | `reference_results/TF_NONE_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` |
| TF linear vs none, 1/4 GPU | `reference_scripts/plot_tf_linear_vs_none_1gpu_4gpu_training_evolution.py` | `reference_results/TF_LINEAR_VS_NONE_1GPU_4GPU_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` |
| PT 100k curves | `reference_scripts/plot_pt_100k_training_evolution.py` | `reference_results/PT_100K_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` |
| TF vs PT 1/4 GPU | `reference_scripts/plot_tf_vs_pt_1gpu_4gpu_training_evolution.py` | `reference_results/TF_VS_PT_1GPU_4GPU_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` |

Common recipe:

- Load each run's numeric ``lcurve.out``.
- Plot the same four training columns everywhere:
  ``TRAIN total loss/RMSE``, ``TRAIN E RMSE (eV/atom)``,
  ``TRAIN F RMSE (eV/A)``, and ``TRAIN virial/stress RMSE (eV/atom)``.
- Use rolling median as the solid line and rolling mean as the dashed line.
- Use log-scale y axes so early and late training are visible together.
- Copy the PNG/TSV/MD outputs to both ``reference_results/`` and the scratch
  plot folder when the plot is still being inspected interactively.

## Zoomed 10k Variants

The 10k plots are not a different algorithm. They are the same lcurve plotting
recipe with the x-axis restricted to the first 10k steps. Use them when the
early optimizer transient matters more than the final checkpoint behavior.

Existing examples:

```text
reference_results/TF_10K_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
reference_results/TF_NONE_10K_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
reference_results/PT_10K_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

## Checkpoint-Restart Stitched Plots

Use this when the question is whether interrupted/restarted training follows a
healthy continuation path:

```text
reference_scripts/plot_checkpoint_restart_stitched_training_evolution.py
```

Current reference figure:

```text
reference_results/TF_PT_4GPU_RESTART_STITCHED_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

Original scratch figure:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/training_loss_plots_10x_20260521/CONTINUATIONS/TF_PT_4GPU_RESTART_STITCHED_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png
```

This plot family differs from the plain lcurve plots:

- The target is restart continuity, not final model ranking.
- Use the cumulative ``lcurve.out`` produced by the restart-chain run
  directory. DeePMD restart tests here recorded absolute post-restart step
  numbers, so do not renumber steps by hand.
- Overlay restart-boundary markers parsed from ``CHAIN_HISTORY.tsv`` when it is
  present. Those vertical markers show where scheduler slices ended and the
  next restart began.
- Keep the same rolling median/mean convention and the same four metric panels
  as the other training-evolution plots.
- Interpret this as an optimizer-health diagnostic. A smooth stitched training
  curve is necessary but not sufficient; validation RMSE still decides whether
  a continued checkpoint is worth using.

The 2026-05-23 stitched restart example compared:

- TF 4GPU linear 10k -> 110k.
- TF 4GPU ``none`` 100k -> 200k.
- PT 4GPU ``none`` 100k -> 200k.

The useful inference was that TF 4GPU ``none`` continued cleanly, while PT
4GPU ``none`` also continued but had noisier late training loss. That is a
restart/optimizer-health observation, not a replacement for validation.

## Avoidable Mistakes

- Do not compare raw total loss across TF/PT restarts without checking the
  component losses too; total-loss definitions or prefactors can make the
  aggregate look more dramatic than energy/force/virial components.
- Do not use "best training total loss" as the production checkpoint selector
  for stable runs. The benchmark showed that best training-total checkpoints
  often did not improve validation.
- Do not assume all plots are schedule-matched. Some TF representative curves
  are linear-scaling or reused baselines; label the schedule in the legend and
  analysis note.
