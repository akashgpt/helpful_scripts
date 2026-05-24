# DeePMD TF/Horovod Global-Batch Benchmark

Date: 2026-05-17

Status: reference snapshot. The first short-run/reused-run matrix and the
completed 10x-step follow-up matrix have both been pseudo-validated. The later
TF/PT validation and training-curve notes include the 2026-05-23 reference
tables, and the 2026-05-24 curated diagnostic summary records the newer 4GPU
seed-repeat and 8GPU/16GPU TF `none` decay10k learning-rate-schedule checks.

Current result snapshot:

```text
RESULTS_SO_FAR.md
```

Archived reference outputs and scripts:

```text
reference_results/
reference_scripts/
```

This experiment bundle tests whether larger Horovod global batch sizes improve
NH3/H2 DeePMD training quality per wall time or GPU-hour. It deliberately reuses
the completed 10k scaling runs under:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422
```

The runnable helper scripts and generated run directories live under the
existing scratch benchmark root:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/
```

The generator creates only the missing runs in a separate scratch tree:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517
```

## Scientific Questions

1. Do larger global batches improve validation/test RMSE at fixed optimizer-step
   count?
2. If `numb_steps` and `decay_steps` are reduced by worker count, does
   multi-GPU training reach similar quality faster?
3. At fixed wall time, do 8/16 GPU runs produce better models than the 1-GPU
   standard?
4. Which configuration is best by GPU-hour, not just wall time?

## Reused Runs

These completed runs should not be repeated:

| Role | GPUs | Steps | Path |
|---|---:|---:|---|
| update-matched baseline | 1 | 10000 | `10k_gpu_scaling_loss_rerun/1gpu` |
| update-matched | 2 | 10000 | `10k_gpu_scaling_loss_rerun/2gpu` |
| update-matched | 4 | 10000 | `10k_gpu_scaling_loss_rerun/4gpu` |
| update-matched | 8 | 10000 | `multinode_scaling_10k_20260516/2node_8gpu` |
| update-matched | 16 | 10000 | `multinode_scaling_10k_20260516/4node_16gpu` |

## Experiment Groups

The new runs were split into these groups:

| Group | Purpose |
|---|---|
| `sample_matched` | Keep approximate rank-batches comparable to 1 GPU x 10k. |
| `lr_sensitivity` | Test DeePMD `scale_by_worker` choices for promising 4/8 GPU cases. |
| `walltime_matched` | Spend about the same wall time as the 1-GPU 10k baseline. |
| `long_steps_10x` | Repeat the matrix with 10x more optimizer steps and 1-hour caps. |
| `long_steps_10x_continuations` | Continue the cancelled 16GPU/100k case from its 50k checkpoint. |
| `long_steps_10x_extra` | Extra post-analysis diagnostics such as the pending 8-node/32GPU/3125-step run. |

The generator marked every case in the scratch matrices and wrote per-run
`myinput.json`, Slurm scripts, and `RUN_INFO.md` files. This benchmark folder
keeps only reference copies of matrices, summaries, and scripts.

## Restart Template Notes, 2026-05-24

The production Della/Tiger train templates are now 1-to-N-GPU safe: changing `#SBATCH --gres=gpu:a100:4` to `#SBATCH --gres=gpu:a100:1` makes the runtime launch one rank per node. The production templates archived in `reference_scripts/` are named without `restart`:

```text
DELLA_TIGER_train_1h.apptr.Ngpu.TF.sh
DELLA_TIGER_train_1h.apptr.Ngpu.PT.sh
ALCF_POLARIS_train_1h.apptr.Ngpu.TF.sh
ALCF_POLARIS_train_1h.apptr.Ngpu.PT.sh
```

The corresponding `.restart.{TF,PT}.sh` copies are kept in `reference_scripts/` as self-resubmitting checkpoint-restart references. They resubmit the same script with `sbatch`/`qsub` until the target step is reached, with duplicate-submit markers, max-chain limits, health gates, numeric checkpoint selection, and rollback guards. Chain state is recorded in `CHAIN_ATTEMPTS.txt`, `CHAIN_HISTORY.tsv`, per-job `.resubmitted_*` markers, and `HEALTH_GATE.tsv` when the health gate is evaluated. The production `train_1h.apptr.Ngpu.{TF,PT}.sh` scripts are not self-resubmitting; in the ALCHEMY pipeline, `TRAIN_MLMD_LEVEL_2.sh` owns chained resubmission and freeze/compress finalization. Neither production nor restart-reference scripts pass `--skip-neighbor-stat`. Restart checkpoint selection is deliberately numeric: TF selects from `model-compression/model.ckpt-*.index` and PT selects from `model-compression/model.ckpt-*.pt`, then chooses the second-highest checkpoint step when available. A rollback guard refuses a restart if the selected checkpoint is unexpectedly far behind the last recorded `lcurve.out` step. This guard was added after a PT chained-restart test reached step 19980 but accidentally restarted from `model.ckpt-9800.pt` because a string/filename ordering path was used in an intermediate test script.

## Usage

Preview what will be created:

```bash
python scripts/materialize_experiments.py --dry-run
```

Create the scratch tree:

```bash
python scripts/materialize_experiments.py
```

Then submit selectively from the generated scratch root:

```bash
cd /scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517
bash submit_sample_matched.sh
```

The runs in this benchmark snapshot have already been submitted and analyzed;
use these commands only as historical reference or as templates for a new
scratch-side experiment.

## Evaluation

Training `lcurve.out` is useful but not sufficient. For scientific decisions,
run the same `dp test` validation/test protocol on all completed checkpoints and
compare:

```text
best validation/test RMSE_f
best validation/test RMSE_e
best validation/test RMSE_v
wall time to best checkpoint
GPU-hours to best checkpoint
rank-batches ~= GPUs * steps * local batch size
```

Pseudo-validation aggregate RMSEs for the 10x completed matrix are reported
with explicit bootstrap percentiles. The bootstrap resamples validation systems
while keeping each system's frame count as the weighting factor for the
aggregate RMSE. The archived percentile columns are p0.135, p15.865, p50,
p84.135, and p99.865, so the median and both 1-sigma/3-sigma
normal-equivalent percentile bounds are visible directly.

Metric units follow the DeePMD `dp test` log output. Energy is
`Energy RMSE/Natoms` in eV/atom, force is `Force RMSE` in eV/A, and the third
reported pseudo-validation metric is `Virial RMSE/Natoms` in eV per atom, not
stress converted to GPa.

For this TF/Horovod setup, larger GPU counts are data-parallel large-batch
experiments. They are not expected to reduce per-step time automatically.

## Current Inference

Energy RMSE/atom (eV/atom) is the primary ranking metric in the completed 10x analysis.
The leading energy group is practically tight:

| Case | GPUs | Steps | Wall s | Energy RMSE/atom (eV/atom) | Force RMSE (eV/A) | Virial RMSE/Natoms (eV/atom) |
|---|---:|---:|---:|---:|---:|---:|
| `reuse_4gpu_10k_10x` | 4 | 100000 | 2109.329 | 0.177581 | 0.655888 | 0.075694 |
| `2gpu_5k_linear_10x` | 2 | 50000 | 1079.118 | 0.181000 | 0.686653 | 0.051985 |
| `reuse_1gpu_10k_10x` | 1 | 100000 | 2076.592 | 0.183390 | 0.692176 | 0.054289 |
| `4gpu_2500_linear_10x` | 4 | 25000 | 567.368 | 0.184682 | 0.701653 | 0.052845 |
| `8gpu_1250_sqrt_10x` | 8 | 12500 | 347.398 | 0.189304 | 0.791522 | 0.062366 |
| `16gpu_625_linear_10x` | 16 | 6250 | 275.955 | 0.197025 | 0.780122 | 0.083433 |

If wall time is the dominant constraint and GPU allocation is available,
`16gpu_625_linear_10x` is a reasonable fast option: about 7.5x faster than the
1GPU/100k case, 3.9x faster than the 2GPU/50k case, 2.1x faster than the
4GPU/25k case, and 1.26x faster than the 8GPU/1250 sqrt case. If GPU-hours and
robustness matter, the 2GPU/4GPU cases are better balanced. The conservative
default remains 1GPU for debugging/baselines and 4GPU for practical
acceleration, with `4gpu_2500_linear_10x` as the best "fast but not wasteful"
setting from the completed 10x matrix.

The 10x follow-up changed the early-read conclusions in two ways. First,
longer training rescued some short-step cases: the original `2gpu_5k_linear`
looked like a clear failure at 5k steps, but `2gpu_5k_linear_10x` is in the
leading energy group at 50k steps. Second, longer training exposed instability
in several large-GPU long-step schedules: `reuse_2gpu_10k_10x`,
`reuse_8gpu_10k_10x`, `8gpu_7k_linear_10x`, `16gpu_5k_linear_10x`, and
`4gpu_2500_sqrt_10x` all end as poor final checkpoints.

The practical trend is therefore not "more GPUs plus more steps is always
better." Scaled-step schedules such as 2GPU/50k, 4GPU/25k, 8GPU/12.5k sqrt,
and 16GPU/6.25k linear are the useful large-batch experiments. Update-matched
or walltime-matched large-GPU long-step schedules need validation-based
checkpoint selection and probably a redesigned learning-rate schedule before
they should be treated as production candidates.

## TF/PT Training-Curve and Validation Update, 2026-05-23

The later TF/PT reference tables and training-curve plots keep the practical
recommendation centered near 4 GPUs. Given the current amount of training data
and the benchmark evidence archived here, 4-GPU TF and 4-GPU PT are the best
practical region right now: they are strong enough in validation RMSE to be
production candidates, while avoiding the extra instability and GPU-hour cost
seen in several larger-GPU schedules.

The training curves suggest that TF improves training metrics faster than PT
early on, especially relative to the 1-GPU comparison. PT is not simply worse,
though: with enough optimizer steps, the PT curves catch or merge toward
comparable late-training quality. This is an optimizer-health observation, not
a model-selection rule. Validation RMSE remains the selection criterion;
training curves are diagnostics for convergence, drift, and schedule health.

Operationally, the 4-GPU evidence supports the hypothesis that production
training may often need fewer optimizer steps than the corresponding 1-GPU
run, possibly around half or quarter in favorable cases. Treat that as an
inference to validate with held-out/pseudo-validation RMSE for each new
dataset and schedule, not as a guaranteed scaling law. The useful scientific
question is whether the shorter 4-GPU run preserves validation energy/force
quality and downstream MD stability, not whether the training loss alone falls
quickly.

For TF multi-GPU production-style runs in this benchmark family, use `none`
worker scaling by default unless a test explicitly asks for `linear`, `sqrt`,
or another scaling choice. Given the evidence so far, `none` is clearly more
stable than `linear` and `sqrt` for the TF multi-GPU cases tested here. The
original 4GPU/100k `none` run stayed stable, and the seed02-seed05 4GPU
`none` repeats all reached 100k without catastrophic blowup. The raw final
rows are noisy, but their late training behavior and pseudo-validation metrics
are consistent enough to support 4GPU TF `none` as the current default region.
This is empirical benchmark evidence, not a theorem. It remains specific to
the present NH3/H2 data and Della DeePMD setup; revisit it when a new dataset,
optimizer schedule, or validation result justifies doing so.

There is also a comparability caveat. The exact TF and PT curves are not all
perfectly matched in schedule and data family. This matters most for the TF
1GPU representative curve, which is a linear-scaling long run rather than the
same fixed-100k `none` design used for the PT 1GPU/4GPU curves and the TF
4GPU curve. Use the TF-vs-PT plots to compare optimizer behavior, but use the
validation tables for production ranking.

Relevant archived artifacts:

- TF `none` seed-repeat and decay10k diagnostic summary:
  `reference_results/TF_NONE_DECAY10K_DIAGNOSTIC_SUMMARY_20260524.tsv`
- PT aggregate training plot:
  `reference_results/PT_100K_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png`
- PT aggregate summary table:
  `reference_results/PT_100K_TRAINING_EVOLUTION_SUMMARY_20260523.tsv`
- TF aggregate training plot:
  `reference_results/TF_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png`
- TF aggregate summary table:
  `reference_results/TF_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_SUMMARY_20260523.tsv`
- TF representative `none` training plot:
  `reference_results/TF_NONE_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png`
- TF-vs-PT 1/4 GPU plot:
  `reference_results/TF_VS_PT_1GPU_4GPU_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png`
- TF-vs-PT 1/4 GPU summary table:
  `reference_results/TF_VS_PT_1GPU_4GPU_TRAINING_EVOLUTION_SUMMARY_20260523.tsv`
- Validation summary tables:
  `reference_results/PT_TF_VALIDATION_REFERENCE_20260523.tsv`,
  `reference_results/PT_VALIDATION_REFERENCE_20260523.tsv`, and
  `reference_results/TF_VALIDATION_REFERENCE_20260523.tsv`
- Companion validation note:
  `reference_results/PT_TF_VALIDATION_REFERENCE_20260523.md`


## TF `none` Seed and Decay10k Diagnostic Update, 2026-05-24

The 2026-05-24 diagnostics add two useful checks to the 4GPU-centered
recommendation. First, the 4GPU TF `scale_by_worker = none` seed repeats
`seed02` through `seed05` all reached 100k steps without catastrophic blowup.
Their final training rows are noisy, but the late training behavior and
pseudo-validation aggregates are fairly consistent: validation energy
RMSE/atom spans 0.17305292-0.17605704 eV/atom and force RMSE spans
0.57159301-0.60205868 eV/A. This supports 4GPU TF `none` as a stable practical
region for the present NH3/H2 benchmark, while still being evidence so far
rather than a general theorem.

Second, the new 8GPU and 16GPU decay10k diagnostics show that learning-rate
schedule tuning helps, but not universally. The old `8gpu_100k_none__final`
run was a real blowup: final training total RMSE 421 and validation energy
RMSE/atom 17.914549 eV/atom. Repeating the same 8GPU TF/Horovod `none` 100k
design with only `decay_steps` changed from 100000 to 10000 removed that
failure. The `8gpu_100k_none_decay10k__final` run reached step 100000 with
final training total RMSE 1.09 and pseudo-validation RMSEs 0.15664999 eV/atom
for energy, 0.45800901 eV/A for force, and 0.031369228 eV/atom for virial.

The companion `16gpu_100k_none_decay10k` run did not share that success. It
timed out at the 1-hour wall limit after step 77270, with final training total
RMSE 1.65e3, energy RMSE 14.2 eV/atom, force RMSE 15.6 eV/A, and virial RMSE
73.8 eV/atom. Because the training loss had already blown up badly, no final
pseudo-validation was launched for that case.

The practical recommendation therefore stays conservative: for NH3/H2 DeePMD
production-style training on this Della TF/PT setup, default to 4GPU TF/PT with
`scale_by_worker = none`. Treat 8GPU TF `none` with `decay_steps = 10000` as a
promising diagnostic or follow-up candidate, not as the new default, and avoid
16GPU TF/PT production runs until a more reliable schedule is demonstrated by
training stability, pseudo-validation RMSE, and downstream MD behavior.
