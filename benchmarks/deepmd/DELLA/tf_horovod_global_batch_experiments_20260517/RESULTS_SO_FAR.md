# DeePMD TF/Horovod Global-Batch Results So Far

Snapshot time: 2026-05-17 17:57 EDT

This is a reference snapshot for the NH3/H2 DeePMD TF/Horovod global-batch
experiments. Runnable jobs and active I/O live in scratch, not in this
benchmark folder.

Scratch experiment root:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517
```

## Reference Files Here

| File | Purpose |
|---|---|
| `EXPERIMENT_MATRIX.tsv` | Original short-run and reused-run matrix. |
| `EXPERIMENT_MATRIX_10x_steps.tsv` | 10x-step follow-up matrix submitted on 2026-05-17. |
| `reference_results/TRAINING_SUMMARY.tsv` | Training log summary for completed short/reused runs. |
| `reference_results/PSEUDO_VALIDATION_SUMMARY.tsv` | Pseudo-validation summary with bootstrap percentile columns. |
| `reference_results/TRAINING_SUMMARY_10x_COMPLETED.tsv` | Training log summary for completed 10x-step follow-up runs. |
| `reference_results/PSEUDO_VALIDATION_SUMMARY_10x_COMPLETED.tsv` | 10x-step pseudo-validation summary with bootstrap percentile columns. |
| `reference_results/METRIC_UNITS.md` | Units and source conventions for training and pseudo-validation columns. |
| `reference_results/EXPERIMENT_MATRIX_10x_COMPLETED_FOR_PSEUDOVAL.tsv` | Completed-only 10x matrix used for freeze/compress/`dp test`. |
| `reference_results/EXTRA_PENDING_32GPU_20260517.tsv` | Extra pending 8-node/32GPU diagnostic submitted after the completed 10x matrix. |
| `reference_results/PSEUDO_VALIDATION_SYSTEMS.tsv` | Pseudo-validation systems used for `dp test`. |
| `reference_results/ANALYSIS_SUMMARY.md` | Scratch-side analysis note at the time of archiving. |
| `reference_results/TF_NONE_DECAY10K_DIAGNOSTIC_SUMMARY_20260524.tsv` | Curated 4GPU TF `none` seed-repeat plus 8GPU/16GPU decay10k diagnostic summary; no raw logs or checkpoints. |
| `reference_results/PT_TF_VALIDATION_REFERENCE_20260523.tsv` | Combined TF/PT validation reference table, sorted for model selection by validation energy RMSE per atom. |
| `reference_results/PT_VALIDATION_REFERENCE_20260523.tsv` | PT-only validation reference slice. |
| `reference_results/TF_VALIDATION_REFERENCE_20260523.tsv` | TF-only validation reference slice. |
| `reference_results/PT_100K_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` | PT 100k aggregate training-curve plot for optimizer-health diagnostics. |
| `reference_results/TF_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` | TF representative long-training aggregate plot for optimizer-health diagnostics. |
| `reference_results/TF_NONE_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` | TF representative `none`-schedule plot using the 1GPU baseline plus available 4GPU/8GPU TF `none` curves. |
| `reference_results/TF_VS_PT_1GPU_4GPU_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` | TF-vs-PT 1GPU/4GPU training-curve comparison plot, with schedule-comparability caveats. |
| `reference_results/TF_VS_PT_NONE_1GPU_4GPU_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` | Explicit TF-vs-PT `none`-schedule 1GPU/4GPU comparison; TF 1GPU is the shared one-worker baseline. |
| `reference_results/TF_LINEAR_VS_NONE_1GPU_4GPU_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png` | TF-only 1GPU baseline and 4GPU `linear`/`none` comparison showing the stable 4GPU `none` curve and the late drift in the comparable 4GPU `linear` curve. |
| TF multi-GPU scaling default | Given the evidence so far, 4GPU TF/PT with `scale_by_worker = none` remains the recommended production-style default; seed02-seed05 4GPU TF `none` repeats reached 100k without catastrophic blowup, 8GPU decay10k is promising, and 16GPU decay10k still failed. |
| `reference_scripts/` | Reference copies of materialization, summarization, submission, pseudo-validation, and restart-template scripts. |
| `reference_scripts/DELLA_TIGER_train_1h.apptr.Ngpu.{TF,PT}.sh` | Della/Tiger production 1-to-N-GPU TF/PT single-slice training templates; Level 2 owns chained resubmission and freeze/compress finalization. |
| `reference_scripts/ALCF_POLARIS_train_1h.apptr.Ngpu.{TF,PT}.sh` | Polaris production multi-GPU TF/PT single-slice training templates with the same Level-2-owned chaining model. |
| `reference_scripts/DELLA_TIGER_train_1h.apptr.Ngpu.restart.{TF,PT}.sh` | Self-resubmitting Della/Tiger checkpoint-restart references retained to document same-script `sbatch` chaining, health gates, numeric checkpoint selection, and rollback guards. |
| `reference_scripts/ALCF_POLARIS_train_1h.apptr.Ngpu.restart.{TF,PT}.sh` | Self-resubmitting Polaris checkpoint-restart references retained to document same-script `qsub` chaining and the same safety logic. |

## Metric Conventions

Pseudo-validation E/F/virial values are parsed from DeePMD `dp test` logs, not
from `lcurve.out`. The reported aggregate is a frame-count-weighted mean of
per-system `dp test` RMSE values.

Units:

- `Energy RMSE/atom` means DeePMD `Energy RMSE/Natoms`, in eV/atom.
- `Force RMSE` means DeePMD `Force RMSE`, in eV/A.
- `Virial RMSE/Natoms` means DeePMD `Virial RMSE/Natoms`, in eV per atom; it
  is not stress in GPa.
- `Total RMSE` is the final training total from `lcurve.out`; it is not a
  pseudo-validation metric.
- Percentile tuples are ordered as p0.135/p15.865/p50/p84.135/p99.865 for
  3-sigma-style summaries, or p15.865/p50/p84.135 where the compact table only
  shows the 1-sigma normal-equivalent interval and median.

## Completed Tests

These completed tests combine reused 10k scaling runs, new short 15-minute
runs, and pseudo-validation.

| Group | Cases | Status |
|---|---|---|
| update-matched reused runs | 1, 2, 4, 8, 16 GPU at 10k steps | complete |
| sample-matched short runs | 2GPU/5k, 4GPU/2500, 8GPU/1250, 16GPU/625 | complete |
| LR sensitivity | 4GPU/2500 and 8GPU/1250 with `linear`, `sqrt`, `none` worker scaling | complete |
| walltime-matched larger GPU runs | 8GPU/7k, 16GPU/5k | complete |
| 1GPU longer baseline | 1GPU/20k | complete |
| pseudo-validation | 127 systems, 2547 frames per model | complete |

Pseudo-validation used DFT-labeled systems from:

```text
v7_i34/md/*/pre/recal/set_train/deepmd
v7_i34/md/*/pre/recal/set_test/deepmd
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i16/md/*/pre/recal/set_train/deepmd
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i16/md/*/pre/recal/set_test/deepmd
```

This is a pseudo-validation set, not a fully independent validation set,
because the configurations trace back to trajectories generated by a related
`pv_comp.pb` model before VASP DFT recalculation.

## Pseudo-Validation Ranking

Frame-weighted aggregate over all 2547 pseudo-validation frames, sorted by
force RMSE (eV/A). Percentiles are bootstrap percentiles over validation systems
while preserving frame weighting.

| Case | GPUs | Steps | Train wall s | GPU s | Force RMSE (eV/A) | Force p15.865/p50/p84.135 | Energy RMSE/atom (eV/atom) | Energy p15.865/p50/p84.135 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `reuse_4gpu_10k` | 4 | 10000 | 218.502 | 874.0 | 0.762202 | 0.732706/0.762741/0.793384 | 0.146282 | 0.139755/0.146287/0.152944 |
| `reuse_16gpu_10k` | 16 | 10000 | 424.447 | 6791.2 | 0.763084 | 0.740084/0.763010/0.785432 | 0.196480 | 0.190570/0.196428/0.202415 |
| `reuse_8gpu_10k` | 8 | 10000 | 308.830 | 2470.6 | 0.766043 | 0.742524/0.765267/0.789443 | 0.193723 | 0.186876/0.193743/0.200385 |
| `8gpu_7k_linear` | 8 | 7000 | 239.202 | 1913.6 | 0.788976 | 0.764488/0.788997/0.813762 | 0.192334 | 0.185434/0.192371/0.199429 |
| `16gpu_5k_linear` | 16 | 5000 | 237.428 | 3798.8 | 0.862008 | 0.829275/0.861369/0.893505 | 0.220959 | 0.213332/0.220713/0.228387 |
| `reuse_2gpu_10k` | 2 | 10000 | 214.899 | 429.8 | 0.874601 | 0.836857/0.874540/0.913035 | 0.203460 | 0.196000/0.203749/0.211293 |
| `1gpu_20k_linear` | 1 | 20000 | 418.889 | 418.9 | 0.891402 | 0.851336/0.891500/0.932288 | 0.205130 | 0.196556/0.205106/0.213498 |
| `8gpu_1250_linear` | 8 | 1250 | 55.391 | 443.1 | 0.911180 | 0.873015/0.910038/0.947943 | 0.200326 | 0.192376/0.200399/0.208621 |
| `4gpu_2500_linear` | 4 | 2500 | 60.668 | 242.7 | 0.954160 | 0.914467/0.953225/0.992468 | 0.193705 | 0.184241/0.193811/0.203115 |
| `8gpu_1250_sqrt` | 8 | 1250 | 45.063 | 360.5 | 1.016572 | 0.977164/1.016353/1.056029 | 0.209289 | 0.198768/0.209226/0.219542 |
| `16gpu_625_linear` | 16 | 625 | 46.854 | 749.7 | 1.057012 | 1.019428/1.057253/1.095656 | 0.209170 | 0.200165/0.209099/0.218003 |
| `4gpu_2500_sqrt` | 4 | 2500 | 65.859 | 263.4 | 1.177871 | 1.143500/1.177742/1.212897 | 0.221385 | 0.209395/0.221589/0.233239 |
| `reuse_1gpu_10k` | 1 | 10000 | 208.078 | 208.1 | 1.182635 | 1.149174/1.182001/1.216910 | 0.215918 | 0.204952/0.215795/0.226693 |
| `4gpu_2500_none` | 4 | 2500 | 60.411 | 241.6 | 1.602351 | 1.559470/1.603452/1.646232 | 0.262680 | 0.250110/0.262506/0.275063 |
| `8gpu_1250_none` | 8 | 1250 | 46.209 | 369.7 | 1.672235 | 1.627938/1.672125/1.718869 | 0.255131 | 0.244872/0.255087/0.265231 |
| `2gpu_5k_linear` | 2 | 5000 | 113.763 | 227.5 | 3.534171 | 3.468584/3.535570/3.598634 | 1.664869 | 1.542206/1.661571/1.787408 |

## Inferences So Far

The best completed run is `reuse_4gpu_10k`. It has the best force RMSE within
the present uncertainty and the best energy RMSE/atom, while costing much less
GPU time than the 8GPU/10k or 16GPU/10k update-matched runs.

The 8GPU and 16GPU 10k update-matched cases have similar force RMSE to
4GPU/10k, but their energy errors are worse and their GPU-second costs are much
higher. They do not currently justify routine use for normal training.

The `8gpu_7k_linear` case is the strongest new walltime-matched large-GPU run.
It is close in force RMSE to the 8GPU/10k and 16GPU/10k references, but still
worse than 4GPU/10k by energy RMSE/atom and GPU cost.

Linear DeePMD `scale_by_worker` learning-rate scaling is better than `sqrt` or
`none` in the short 4GPU/8GPU tests. The `none` cases are poor diagnostic
controls, not promising production settings.

The `2gpu_5k_linear` run is an outlier failure by both training behavior and
pseudo-validation. It should not guide scheduling choices unless the 10x follow
up shows that the short result was only an early-training artifact.

The current practical recommendation is:

1. Keep 1GPU for cheap debugging and serial baselines.
2. Prefer 4GPU update-matched training when wall clock matters.
3. Use 8GPU only when shorter wall clock is worth extra GPU cost.
4. Avoid routine 16GPU training unless the training schedule is redesigned and
   judged by validation RMSE per wall time and per GPU-hour.

## 10x-Step Follow-Up Completed Update

Snapshot time: 2026-05-17 23:05 EDT

The completed 10x follow-up was summarized from:

```text
TRAINING_SUMMARY_10x_COMPLETED.tsv
pseudo_validation_10x_20260517/PSEUDO_VALIDATION_SUMMARY.tsv
```

All 15 completed 10x models were frozen, compressed, and tested on the same 127
pseudo-validation systems. The intentionally cancelled original
`reuse_16gpu_10k_10x` was excluded; its continuation
`reuse_16gpu_10k_10x_from50k` was included.

Frame-weighted aggregate over all 2547 pseudo-validation frames, sorted by
energy RMSE/atom (eV/atom). The table keeps force, virial, and final training
total RMSE in view, but the main ranking criterion here is energy. Percentile
tuples are p0.135/p15.865/p50/p84.135/p99.865.

| Case | GPUs | Steps | Total RMSE | Energy RMSE/atom (eV/atom) | Energy percentiles | Force RMSE (eV/A) | Force percentiles | Virial RMSE/Natoms (eV/atom) | Virial percentiles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reuse_4gpu_10k_10x` | 4 | 100000 | 0.997 | 0.177581 | 0.158734/0.171740/0.177876/0.183535/0.194067 | 0.655888 | 0.605329/0.636972/0.655933/0.675700/0.721113 | 0.075694 | 0.053935/0.066356/0.075006/0.084782/0.110012 |
| `2gpu_5k_linear_10x` | 2 | 50000 | 1.03 | 0.181000 | 0.160456/0.174224/0.181103/0.187784/0.199375 | 0.686653 | 0.637698/0.668910/0.686414/0.704991/0.745269 | 0.051985 | 0.046309/0.050066/0.051989/0.053983/0.057959 |
| `reuse_1gpu_10k_10x` | 1 | 100000 | 1.3 | 0.183390 | 0.161084/0.176372/0.183571/0.190311/0.203226 | 0.692176 | 0.644828/0.675363/0.692220/0.710254/0.750057 | 0.054289 | 0.048049/0.052093/0.054339/0.056585/0.061425 |
| `4gpu_2500_linear_10x` | 4 | 25000 | 1.32 | 0.184682 | 0.162389/0.177629/0.184766/0.191526/0.203885 | 0.701653 | 0.644943/0.681923/0.701438/0.721664/0.766574 | 0.052845 | 0.046603/0.050787/0.052836/0.054943/0.059203 |
| `8gpu_1250_sqrt_10x` | 8 | 12500 | 0.86 | 0.189304 | 0.168001/0.182505/0.189484/0.196028/0.208640 | 0.791522 | 0.724657/0.768007/0.790953/0.815488/0.872743 | 0.062366 | 0.055839/0.060024/0.062325/0.064619/0.069578 |
| `16gpu_625_linear_10x` | 16 | 6250 | 1.51 | 0.197025 | 0.174570/0.190047/0.197057/0.203656/0.217087 | 0.780122 | 0.713963/0.755912/0.779989/0.804913/0.862264 | 0.083433 | 0.059381/0.073027/0.082599/0.093787/0.122674 |
| `4gpu_2500_none_10x` | 4 | 25000 | 1.89 | 0.197189 | 0.173207/0.189677/0.197146/0.204454/0.218877 | 0.850935 | 0.785511/0.827003/0.850462/0.875389/0.930778 | 0.063014 | 0.056610/0.060879/0.062979/0.065112/0.069569 |
| `8gpu_1250_none_10x` | 8 | 12500 | 1.34 | 0.199872 | 0.177518/0.192313/0.199796/0.207358/0.222800 | 0.852042 | 0.759669/0.819235/0.852226/0.886065/0.951832 | 0.064143 | 0.056563/0.061421/0.064087/0.066789/0.072202 |
| `reuse_8gpu_10k_10x` | 8 | 100000 | 11.4 | 0.294202 | 0.260374/0.282922/0.294551/0.305806/0.328378 | 3.035107 | 2.821946/2.963069/3.032865/3.107247/3.257685 | 0.545935 | 0.529625/0.540270/0.545819/0.551606/0.564146 |
| `reuse_16gpu_10k_10x_from50k` | 16 | 100000 | 21 | 2.276191 | 1.647704/2.061689/2.272114/2.481947/2.894414 | 2.070921 | 1.917512/2.019597/2.070609/2.123732/2.227362 | 0.864581 | 0.549384/0.706606/0.844860/1.016184/1.448211 |
| `8gpu_1250_linear_10x` | 8 | 12500 | 75.1 | 2.584377 | 2.381804/2.516315/2.585582/2.653173/2.784179 | 0.754503 | 0.667178/0.725601/0.754387/0.784838/0.849322 | 0.505609 | 0.427487/0.478904/0.504894/0.531922/0.593989 |
| `16gpu_5k_linear_10x` | 16 | 50000 | 98.1 | 4.295180 | 3.978407/4.198841/4.297243/4.392560/4.573574 | 1.133693 | 1.058028/1.108595/1.133836/1.159057/1.209318 | 0.535195 | 0.419680/0.492882/0.533575/0.577200/0.673368 |
| `4gpu_2500_sqrt_10x` | 4 | 25000 | 171 | 4.303055 | 3.645689/4.078259/4.299717/4.531679/4.998352 | 3.767129 | 3.322353/3.620220/3.767266/3.913400/4.210296 | 5.121701 | 4.433761/4.906494/5.124210/5.342970/5.787742 |
| `reuse_2gpu_10k_10x` | 2 | 100000 | 491 | 12.611698 | 10.610701/11.930820/12.603242/13.307674/14.716390 | 13.010718 | 11.562020/12.527966/13.009773/13.478294/14.335430 | 16.416686 | 14.824678/15.874769/16.420473/16.956311/17.938089 |
| `8gpu_7k_linear_10x` | 8 | 70000 | 396 | 17.579198 | 16.676149/17.253399/17.570189/17.896970/18.550904 | 4.923031 | 4.564900/4.806698/4.924098/5.038410/5.267157 | 3.791055 | 3.192417/3.592369/3.790783/3.989618/4.380441 |

The 10x results reinforce the conservative scheduling recommendation. The
best pseudo-validation energy RMSE/atom is the 4-GPU update-matched case, but
2GPU/50k, 1GPU/100k, and 4GPU/25k are statistically close by energy. These
same cases also stay near the top by force and virial RMSE. The larger
GPU counts do not show a clean quality win after 10x more steps, and several
large-GPU/long-step cases are clear failures by energy, force, virial, and
training total RMSE.

For a fixed 1-hour wall-clock budget, the evidence still does not support the
simple equivalence "10 nodes for 1 hour equals 4 GPUs for 10 hours or 1 GPU for
40 hours." More GPUs increase data throughput, but model quality depends on
the global-batch training schedule, learning-rate scaling, optimizer dynamics,
and validation behavior.

If wall time is the main constraint and GPU allocation/cost is secondary,
`16gpu_625_linear_10x` is competitive. It reaches energy RMSE/atom 0.197025,
force RMSE 0.780122, and training total RMSE 1.51 in 275.955 s. Relative to
the nearby energy group, that is about 7.5x faster than `reuse_1gpu_10k_10x`,
3.9x faster than `2gpu_5k_linear_10x`, 2.1x faster than
`4gpu_2500_linear_10x`, and 1.26x faster than `8gpu_1250_sqrt_10x`. Energy is
a little worse than the 1/2/4GPU group, but not catastrophically; force is
similar to the 8GPU case and not far from the 4GPU case. Virial RMSE/Natoms is
somewhat worse than the 2/4/8GPU short-step cases, but still nowhere near the
failure cases.

The practical split is therefore:

1. If wall-clock time dominates, `16gpu_625_linear_10x` is a reasonable
   fast-training option.
2. If GPU-hours or robustness dominate, the 2GPU/50k and 4GPU/25k cases look
   better.
3. If one wants the most balanced "fast but not wasteful" setting,
   `4gpu_2500_linear_10x` is probably the sweet spot.

## 10k Versus 10x Trend Update

The 10x follow-up shows that short-run behavior is not always predictive of
final model quality. Longer training improves the normal 1GPU baseline and
rescues some short-step cases, but it also exposes unstable long-step
large-GPU schedules.

| Case family | GPUs | Short steps | 10x steps | Energy short -> 10x | Force short -> 10x | Main read |
|---|---:|---:|---:|---:|---:|---|
| `reuse_1gpu_10k` -> `reuse_1gpu_10k_10x` | 1 | 10000 | 100000 | 0.216 -> 0.183 | 1.183 -> 0.692 | improved standard baseline |
| `reuse_2gpu_10k` -> `reuse_2gpu_10k_10x` | 2 | 10000 | 100000 | 0.203 -> 12.612 | 0.875 -> 13.011 | failed with long update-matched training |
| `reuse_4gpu_10k` -> `reuse_4gpu_10k_10x` | 4 | 10000 | 100000 | 0.146 -> 0.178 | 0.762 -> 0.656 | force improves; energy not monotonic |
| `reuse_8gpu_10k` -> `reuse_8gpu_10k_10x` | 8 | 10000 | 100000 | 0.194 -> 0.294 | 0.766 -> 3.035 | degraded under long update-matched training |
| `2gpu_5k_linear` -> `2gpu_5k_linear_10x` | 2 | 5000 | 50000 | 1.665 -> 0.181 | 3.534 -> 0.687 | short-run failure was early-training artifact |
| `4gpu_2500_linear` -> `4gpu_2500_linear_10x` | 4 | 2500 | 25000 | 0.194 -> 0.185 | 0.954 -> 0.702 | improved balanced fast option |
| `4gpu_2500_sqrt` -> `4gpu_2500_sqrt_10x` | 4 | 2500 | 25000 | 0.221 -> 4.303 | 1.178 -> 3.767 | failed when trained longer |
| `8gpu_1250_linear` -> `8gpu_1250_linear_10x` | 8 | 1250 | 12500 | 0.200 -> 2.584 | 0.911 -> 0.755 | energy failed despite acceptable force |
| `8gpu_1250_sqrt` -> `8gpu_1250_sqrt_10x` | 8 | 1250 | 12500 | 0.209 -> 0.189 | 1.017 -> 0.792 | stable 8GPU scaled-step choice |
| `16gpu_625_linear` -> `16gpu_625_linear_10x` | 16 | 625 | 6250 | 0.209 -> 0.197 | 1.057 -> 0.780 | improved wall-clock-only option |
| `8gpu_7k_linear` -> `8gpu_7k_linear_10x` | 8 | 7000 | 70000 | 0.192 -> 17.579 | 0.789 -> 4.923 | failed badly when trained longer |
| `16gpu_5k_linear` -> `16gpu_5k_linear_10x` | 16 | 5000 | 50000 | 0.221 -> 4.295 | 0.862 -> 1.134 | failed by energy |

The main takeaways are:

1. Validation-based checkpoint selection matters. The 4GPU update-matched case
   improves in force from 10k to 100k, but its energy RMSE is better at 10k
   than at the final 100k checkpoint.
2. Scaled-step large-batch training is much safer than update-matched
   long-step large-GPU training.
3. The failed 10x runs did not look like cleanly improving trajectories. Their
   training curves generally reached better regions earlier and then drifted or
   blew up by the final checkpoint.
4. For production, the comparison should be energy-focused pseudo-validation
   RMSE per wall time and per GPU-hour, followed by downstream MD stability,
   not final training loss alone.

## Historical 10x-Step Submission Snapshot

The 10x-step follow-up was generated in:

```text
runs/long_steps_10x/
```

It repeats the completed matrix with 10x more optimizer steps, except
`1gpu_20k_linear`, which was intentionally skipped. Every 10x job is capped at
1 hour wall time.

Submitted Slurm jobs:

| Job ID | Case | GPUs | Steps | Priority at setup | State at snapshot |
|---:|---|---:|---:|---|---|
| 8370701 | reuse_1gpu_10k_10x | 1 | 100000 | core | pending |
| 8370702 | reuse_2gpu_10k_10x | 2 | 100000 | secondary | pending |
| 8370703 | reuse_4gpu_10k_10x | 4 | 100000 | core | pending |
| 8370704 | reuse_8gpu_10k_10x | 8 | 100000 | core | running |
| 8370705 | reuse_16gpu_10k_10x | 16 | 100000 | secondary | running |
| 8370706 | 2gpu_5k_linear_10x | 2 | 50000 | diagnostic | pending |
| 8370707 | 4gpu_2500_linear_10x | 4 | 25000 | secondary | pending |
| 8370708 | 4gpu_2500_sqrt_10x | 4 | 25000 | secondary | pending |
| 8370709 | 4gpu_2500_none_10x | 4 | 25000 | diagnostic | pending |
| 8370710 | 8gpu_1250_linear_10x | 8 | 12500 | secondary | pending |
| 8370711 | 8gpu_1250_sqrt_10x | 8 | 12500 | secondary | pending |
| 8370712 | 8gpu_1250_none_10x | 8 | 12500 | diagnostic | pending |
| 8370713 | 16gpu_625_linear_10x | 16 | 6250 | secondary | pending |
| 8370714 | 8gpu_7k_linear_10x | 8 | 70000 | core | pending |
| 8370715 | 16gpu_5k_linear_10x | 16 | 50000 | secondary | pending |

These jobs test whether the short-run trends hold when the models get more
training, and whether the apparent large-GPU behavior is just early-training
dynamics. Because the cap is 1 hour, the 100k-step cases may stop at the wall
limit before reaching their scripted final step; that outcome is itself useful
for the "1 hour on many GPUs versus many hours on fewer GPUs" question.

## Remaining Follow-Up

Two jobs were submitted separately after the completed 10x matrix and were
still pending at this update:

| Job ID | Case | Purpose |
|---:|---|---|
| 8375105 | `dpgb16g_100k_90m` | Fresh 16GPU/100k/90m diagnostic. |
| 8381309 | `32gpu_3125_linear_10x` | Extra 8-node/32GPU wall-time diagnostic, linear scaling, 3125 steps = 100000/32. |

When either run finishes:

1. regenerate the completed-run training summary if it should be included;
2. freeze/compress the new checkpoint;
3. run the same pseudo-validation set;
4. compare RMSEs with explicit bootstrap percentile columns;
5. decide whether it changes the large-GPU conclusion.


## TF `none` Seed and Decay10k Diagnostic Update

Snapshot time: 2026-05-24

The newest TF `none` diagnostics support the 4GPU-centered recommendation but
do not justify moving the default to larger GPU counts. The 4GPU seed-repeat
runs `seed02__final`, `seed03__final`, `seed04__final`, and `seed05__final`
all reached 100000 steps without catastrophic blowup. Their raw final training
rows remain noisy, but the late training behavior was stable enough for a
production-style benchmark read, and their aggregate pseudo-validation metrics
cluster tightly: validation energy RMSE/atom 0.17305292-0.17605704 eV/atom,
force RMSE 0.57159301-0.60205868 eV/A, and virial RMSE/Natoms
0.045853057-0.048237282 eV/atom. This is evidence so far, not a theorem about
all datasets or schedules.

The 8GPU decay10k diagnostic shows that the old 8GPU TF `none` failure was
strongly learning-rate-schedule dependent. The old `8gpu_100k_none__final`
run had final training total RMSE 421 and validation energy RMSE/atom
17.914549 eV/atom. Changing only `decay_steps` from 100000 to 10000 produced
`8gpu_100k_none_decay10k__final`, which completed 100000 steps with final
training total RMSE 1.09 and validation RMSEs 0.15664999 eV/atom for energy,
0.45800901 eV/A for force, and 0.031369228 eV/atom for virial.

The 16GPU companion prevents overgeneralizing that result.
`16gpu_100k_none_decay10k` timed out at step 77270 with training already blown
up: final training total RMSE 1.65e3, energy RMSE 14.2 eV/atom, force RMSE
15.6 eV/A, and virial RMSE 73.8 eV/atom. No final pseudo-validation was run.

Current recommendation: keep 4GPU TF/PT with `scale_by_worker = none` as the
recommended default for this NH3/H2 Della benchmark family. Treat 8GPU TF
`none` with `decay_steps = 10000` as a promising follow-up candidate, and
treat 16GPU as not production-ready until a schedule passes training stability,
pseudo-validation, and downstream MD checks. See
`reference_results/TF_NONE_DECAY10K_DIAGNOSTIC_SUMMARY_20260524.tsv` for the
curated TSV record.
