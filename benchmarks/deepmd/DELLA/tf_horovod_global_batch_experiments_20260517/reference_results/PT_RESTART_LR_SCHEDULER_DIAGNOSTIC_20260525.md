# PT Restart Learning-Rate Scheduler Diagnostic

Date: 2026-05-25

This note records a DeePMD PyTorch restart issue observed in the Della NH3/H2
4GPU checkpoint-chain tests. It explains why the PT restart-chain losses were
not comparable to the clean 4GPU PT baseline, even when checkpoint selection,
rollback guards, and health gates were fixed.

## Short Version

Do not trust chained `dp --pt train --restart ...` production runs in the
tested DeePMD PT environment until the scheduler behavior is patched and
revalidated.

The restarted PT learning rate is not monotonic. It jumps upward after later
restarts, and the jump is visible directly in raw `lcurve.out`, not only in
plots. The same chained-restart pattern in TF keeps decaying monotonically.

## Affected Environment

Observed in:

```text
DeePMD version: 3.1.3
Backend: PyTorch
Source commit: b2c8511e
PT version: 2.6.0
Environment: ALCHEMY_env__PT on Della
```

The diagnostic run was:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517/runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2
```

Input schedule:

```json
"learning_rate": {
  "type": "exp",
  "start_lr": 0.001,
  "decay_steps": 100000,
  "scale_by_worker": "none"
}
```

The DeePMD-normalized `out.json` added:

```json
"stop_lr": 5e-08,
"smooth": false
```

## Raw Evidence

The chain restarted from numerically sensible second-latest checkpoints:

| chain slice | mode | checkpoint | lcurve step before restart |
|---:|---|---:|---:|
| 1 | fresh | none | 20320 |
| 2 | restart | `model.ckpt-20200.pt` | 40710 |
| 3 | restart | `model.ckpt-40600.pt` | 61180 |
| 4 | restart | `model.ckpt-61000.pt` | 84030 |
| 5 | restart | `model.ckpt-83900.pt` | 100000 |

Checkpoint contents confirmed that optimizer state was stored, including the
optimizer LR and Adam step counter:

| checkpoint | checkpoint train step | checkpoint optimizer LR |
|---:|---:|---:|
| `20200` | 20200 | `1.3526702575e-04` |
| `40600` | 40601 | `1.7938322258e-05` |
| `61000` | 61002 | `2.7598606387e-06` |
| `83900` | 83903 | `2.0811207130e-05` |
| `100000` | 100004 | `2.0101716360e-04` |

The problematic jumps are visible in `lcurve.out`:

| region | before restart | after restart |
|---|---:|---:|
| near 61000 | step 61180, LR `2.8e-06` | restarted at 61000, LR `2.1e-05` |
| near 83900 | step 84030, LR `2.1e-05` | restarted at 83900, LR `2.0e-04` |

The second jump is large enough to explain the visible PT loss degradation:
the rows immediately after restarting from `model.ckpt-83900.pt` have total
training loss around `7-14`, whereas rows just before the restart were mostly
around `1-3`.

## Root Cause Inference

The installed PT trainer rebuilds the scheduler with both:

```python
last_epoch=self.start_step - 1
```

and:

```python
lambda step: self.lr_schedule.value(step + self.start_step) / initial_lr
```

where:

```python
initial_lr = self.lr_schedule.value(self.start_step)
```

This effectively counts the restart offset twice. In early restarts this can
look almost harmless. In later restarts, once `step + self.start_step` passes
the schedule end and the schedule clips to `stop_lr`, the ratio can invert the
intended decay and raise the optimizer LR. That matches the observed pattern:
late PT restarts resume at much higher LR and then stay high.

The local source path inspected was:

```text
/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/ALCHEMY_env__PT/lib/python3.11/site-packages/deepmd/pt/train/training.py
```

This is distinct from the experimental/non-DDP `pt_expt` trainer, whose local
source already has a comment warning that the lambda must not add the restart
step again. The active multi-GPU PT run used the normal `deepmd/pt` trainer,
not `pt_expt`.

## Why This Is Not Just Plotting

The LR increase is in raw `lcurve.out` and checkpoint optimizer state. The
plot only revealed it.

The same plotting workflow also showed TF restart chains. TF chains have small
step rollbacks from using the second-latest checkpoint, but TF LR remains
monotonically decreasing. Therefore the PT behavior is not caused by the plot,
the 10-minute wall-time slicing, or the second-latest checkpoint strategy
alone.

## Relation To Official DeePMD Usage

The launch style is consistent with the DeePMD PyTorch parallel-training docs:
the docs describe PyTorch distributed training through `torchrun`, e.g.

```text
torchrun --nproc_per_node=4 --no-python dp --pt train input.json
```

Reference:

```text
https://docs.deepmodeling.org/projects/deepmd/en/stable/train/parallel-training.html
```

So the problem is not that PT multi-GPU training was launched in a completely
unsupported way. The issue is specifically the restart/scheduler behavior in
this installed PT trainer.

## Practical Guidance

- For PT production training, prefer uninterrupted jobs or long enough wall
  times to finish in one scheduler allocation.
- Do not use PT chained restart as a production-safe path until a patched
  scheduler restart has been tested against raw `lcurve.out`.
- Keep the PT restart-chain plots as failure-mode diagnostics, not as model
  quality references.
- When testing a patch, verify raw `lcurve.out` around every restart boundary:
  LR should not increase except for a tiny display/rounding artifact.
- Also inspect checkpoint optimizer state directly. The checkpoint should
  store an LR consistent with the next resumed `lcurve.out` rows.

## Candidate Fix To Test

The intended scheduler rebuild should use one global notion of step, not both
`last_epoch=start_step-1` and `step + start_step`.

A safer shape is:

```python
base_lr = self.lr_schedule.value(0)
self.scheduler = torch.optim.lr_scheduler.LambdaLR(
    self.optimizer,
    lambda step: self.lr_schedule.value(step) / base_lr,
    last_epoch=self.start_step - 1,
)
```

This should be tested in a copied/temporary environment before any production
script relies on it. Do not patch the shared conda environment silently.

