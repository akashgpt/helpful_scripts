# He_MgSiO3 54MgSiO3_90He DPA-2 current_auto 1M decay10k no-val

## Status snapshot

- Last updated: 2026-05-28 19:33 CDT on NCSA Delta login node `dt-login02`
- Job ID: `18590160`
- Slurm state after submit check: `RUNNING` on `gpue06`
- Job-start files now present in the variant directory: `input.json`, `log.train`, `slurm-18590160.out`
- `squeue` snapshot:

```text
JOBID PARTITION NAME USER ST TIME TIME_LIMIT NODES NODELIST(REASON)
18590160 gpuH200x8 dpa2_cur_auto_1M_d10k agupta46 R 0:51 2-00:00:00 1 gpue06
```

- `sacct` snapshot:

```text
JobID JobName Partition Account AllocTRES State ExitCode Elapsed Start End
18590160 dpa2_cur_auto_1M_d10k gpuH200x8 bguf-delt+ billing=3000,cpu=2,gres/gpu:h200=1,gres+ RUNNING 0:0 00:00:33 2026-05-28T19:32:46 Unknown
```

Early `log.train` check at 2026-05-28 19:33 CDT reached DeePMD startup on
`gpue06`, detected `NVIDIA H200`, and was calculating neighbor statistics.

## Working directory

```text
/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_current_auto_1M_H200_decay10k_no_val
```

## What was submitted

- Submitted from inside the variant directory with `sbatch sub.sh`.
- Partition: `gpuH200x8`
- GPUs: one H200 (`--gpus-per-node=1`, `--gpus-per-task=1`)
- Time limit: 48 hours
- Environment: `ALCHEMY_env__PT`
- DeePMD command: `dp --pt train input.json`, or `dp --pt train --restart <latest model.ckpt-*.pt> input.json` if a checkpoint exists.

## Config summary

- Base shared JSON: `../shared/train_dpa2_current_auto_200k.json`
- New shared JSON: `../shared/train_dpa2_current_auto_1M_H200_decay10k_no_val.json`
- Intentional JSON change only:
  - `training.numb_steps`: `200000` -> `1000000`
- Confirmed unchanged:
  - `learning_rate.decay_steps = 10000`
  - no `training.validation_data`
  - no three-body, warmup, clipping, model, loss, or training-data changes
- The job wrapper copies the shared JSON to local `input.json` at job start.

## Verification completed before submit

```text
python -m json.tool ../shared/train_dpa2_current_auto_1M_H200_decay10k_no_val.json
bash -n sub.sh
```

Structured JSON comparison against `train_dpa2_current_auto_200k.json`:

```text
changed_paths=training.numb_steps
old_training.numb_steps=200000
new_training.numb_steps=1000000
new_learning_rate.decay_steps=10000
new_has_validation_data=False
```

## How to check

```bash
squeue -j 18590160 -o '%.18i %.9P %.32j %.8u %.2t %.10M %.10l %.6D %R'
sacct -j 18590160 --format=JobID,JobName%32,Partition,Account,AllocTRES%40,State,ExitCode,Elapsed,Start,End
tail -f /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_current_auto_1M_H200_decay10k_no_val/log.train
```

## How to resume or rerun

If the job exits before 1M steps and leaves `model.ckpt-*.pt`, resubmit from the
variant directory:

```bash
cd /work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench/variant_train_dpa2_PT_current_auto_1M_H200_decay10k_no_val
sbatch sub.sh
```

The wrapper will restart from the latest checkpoint using version sort.

## Next step

Watch the early `log.train` output for DeePMD startup errors, then monitor
checkpoint creation and final completion.

## Later snapshot

2026-05-29 10:18 CDT:

- `sacct`: job `18590160` still `RUNNING`, elapsed `14:45:24`, node `gpue06`.
- Latest visible checkpoints include `model.ckpt-711000.pt` through `model.ckpt-715000.pt`.
- This indicates the run is progressing normally toward 1M steps.

## Follow-Up After Completion

Once this current-auto 1M control and the doc-medium 1M warmup 57-val run are both complete, set up a network-size scaling test:

- increase network size for the `current_auto` variant;
- increase network size for the `doc_medium` variant;
- keep the comparison controlled against these completed 1M baselines;
- validate the final checkpoints on the 57 in-domain MgSiO3 systems before drawing conclusions.
