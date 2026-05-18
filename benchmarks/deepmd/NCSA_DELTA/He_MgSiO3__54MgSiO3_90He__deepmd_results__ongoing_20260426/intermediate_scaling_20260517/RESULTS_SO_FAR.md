# Results So Far

Snapshot time: 2026-05-17 19:30 CDT on NCSA Delta.

## Completed Validation

The completed validation used copied local `CONFIG=71MgSiO3_5He` systems from
`pre/recal/{set_train,set_test}` and did not modify any `qmd_data/v1_i*` source data.

Held-out set:

- Dataset: `v1_i2` test split.
- Frames: 125.
- Metric emphasized here: energy RMSE per atom, because the energy controls the
  thermodynamic properties we care about most.
- Error bars: asymmetric bootstrap intervals on the reported energy RMSE/atom.
  The 1-sigma interval uses the 16/84 percentiles. The 3-sigma interval uses the
  0.135/99.865 percentiles.

| rank | model | params | E RMSE/atom | -1 sigma | +1 sigma | -3 sigma | +3 sigma | training GPU time |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `balanced_10x` | 26.80M | 0.060336 | 0.001987 | 0.001752 | 0.005753 | 0.005367 | 46h07m |
| 2 | `big` | 23.91M | 0.102139 | 0.003013 | 0.002629 | 0.009012 | 0.008146 | 22h53m |
| 3 | `fit_deep2x` | 5.28M | 0.241183 | 0.003114 | 0.002942 | 0.009412 | 0.008930 | 12h47m |
| 4 | `fit_deep10x` | 26.77M | 0.266868 | 0.003078 | 0.002908 | 0.009120 | 0.008865 | 28h52m |
| 5 | `base` | 2.67M | 0.383706 | 0.004058 | 0.003861 | 0.012287 | 0.011730 | 10h58m |

Inference:

- `balanced_10x` is clearly the best completed model for held-out energy on this
  validation set.
- Width-only `big` is the practical runner-up: worse energy than `balanced_10x`, but
  much cheaper than `balanced_10x`.
- Pure fitting-net depth scaling is not competitive with width-only or balanced scaling
  for energy here. `fit_deep10x` is especially unattractive because it costs nearly as
  many parameters as `balanced_10x` but gives much worse held-out energy.
- The ranking gaps are much larger than the bootstrap error bars, so the completed-model
  ranking is stable for this validation sample.
- These are interpolation-style validation results within the copied `71MgSiO3_5He`
  train/test distribution. They do not by themselves prove transfer to other compositions,
  pressure-temperature regions, or unusual structural states.

## Intermediate Jobs Started

The new intermediate jobs test whether the balanced improvement is smooth with model
size and whether a cheaper balanced model captures most of the `balanced_10x` energy gain.

| case | job id | state at snapshot | architecture | submitted descriptor | submitted fitting net |
|---|---:|---|---|---|---|
| `big2x` | 18313921 | RUNNING | width-only | `[35,70,140]` | `[340,340,340]` |
| `balanced_2x` | 18313922 | RUNNING | balanced width+depth | `[30,60,120,120]` | `[320,320,320,320]` |
| `big5x` | 18313923 | RUNNING | width-only | `[56,112,224]` | `[540,540,540]` |
| `balanced_5x` | 18313924 | RUNNING | balanced width+depth | `[45,90,180,180,180]` | `[480,480,480,480,480]` |
| `fit_deep5x` | 18313925 | RUNNING | fitting-depth-only | `[25,50,100]` | `[240] x 40` |

Dependency policy:

- The 5x jobs were submitted with Slurm `after:18313921:18313922`, not `afterok`.
- This means the 5x jobs became eligible once both 2x jobs started.
- This matches the intended ordering: launch 2x cases first, then allow 5x cases to start
  without waiting for the 2x cases to finish.

## Wrapper Corrections

Two preliminary 2x submission attempts failed before DeePMD training started:

- `18313765`, `se_big2x`, 9 seconds.
- `18313766`, `se_bal2x`, 10 seconds.

Cause:

- The scripts tried to locate `../shared` relative to the Slurm-copied script path under
  `/var/spool/slurmd`.

Fix:

- The final submitted scripts hardcode each run directory in `HERE=...`.
- The failed preliminary jobs should not be interpreted as model/config failures.

## Next Update When Jobs Finish

For each intermediate run, collect:

- Slurm elapsed wall time and GPU time.
- Final `lcurve.out` force/energy/virial training behavior.
- Frozen-model validation on the same `71MgSiO3_5He` held-out test split.
- Energy RMSE/atom with asymmetric bootstrap 1-sigma and 3-sigma intervals.
- Updated ranking against base, `big`, `fit_deep2x`, `fit_deep10x`, and `balanced_10x`.
