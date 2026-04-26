# DeePMD-kit TF/Horovod Apptainer Benchmark on Della

**Date:** 2026-04-22  
**Cluster:** Della, Princeton GPU nodes, A100 80 GB GPUs  
**Benchmark folder:** `$VASP_DATA/benchmarks/deepmd/DELLA/tf_horovod_apptainer300cuda126_20260422`  
**Working run folder:** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422`

## Purpose

Test whether the Apptainer image
`/scratch/gpfs/BURROWS/akashgpt/softwares/APPTAINER_REPO/deepmd-kit_3.0.0_cuda126.sif`
can run DeePMD-kit TensorFlow/Horovod training across multiple GPUs on Della, and record:

- 10,000-step training-loss behavior for 1, 2, and 4 GPUs.
- GPU use and memory use during those runs.
- CPU allocation sensitivity for 4-GPU training using 1000-step runs.
- The launcher details needed to make the container work reliably under Slurm.

## Environment

| Component | Value |
|---|---|
| DeePMD-kit | 3.0.0 |
| TensorFlow | 2.17.0 |
| Horovod | 0.28.1 |
| Container | `deepmd-kit_3.0.0_cuda126.sif` |
| GPUs | Della A100 80 GB |
| Slurm partition/QOS | `gputest`, `gpu-test` |

Horovod build check inside the image showed MPI and NCCL support. Gloo was not available. The practical working launcher is host `srun` launching `apptainer exec --nv`, not `horovodrun` or container-side `mpirun`.

Important launcher settings:

```bash
export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export DP_INFER_BATCH_SIZE=32768
export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=2
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/deepmd-kit
export NCCL_DEBUG=WARN

srun --mpi=pmix --ntasks="${nproc}" --cpu-bind=cores --kill-on-bad-exit=1 \
	apptainer exec --nv "${image}" env ... dp train --mpi-log=workers --skip-neighbor-stat myinput.json
```

Do not use per-task GPU binding such as `--gpus-per-task=1 --gpu-bind=single:1` for this DeePMD/Horovod setup. DeePMD expects each local rank to see all local GPUs and then binds internally. With Slurm per-task GPU binding, ranks saw only one visible GPU and DeePMD failed with a local-rank/GPU-count mismatch.

`XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/deepmd-kit` is needed because TensorFlow otherwise could not find `libdevice.10.bc`.

## 10k GPU Loss Scaling

These runs use the same input, same dataset, and 10,000 training steps. The purpose is not pure walltime acceleration, but whether increasing GPU count changes training-loss behavior at fixed optimizer-step count.

| GPUs | Job ID | Slurm elapsed | DeePMD train time | DeePMD wall time | Avg active GPU util | Final RMSE | Mean RMSE, steps 9000-10000 | Mean RMSE_f, steps 9000-10000 | Best late RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7261183 | 00:05:14 | 0.0194 s/batch | 208.078 s | 86.7% | 8.41 | 3.013 | 1.285 | 1.150 at step 9510 |
| 2 | 7261184 | 00:06:28 | 0.0200-0.0209 s/batch | 214.899-215.193 s | 85.7%, 87.1% | 3.25 | 2.274 | 1.041 | 0.836 at step 9390 |
| 4 | 7261185 | 00:06:48 | 0.0202-0.0211 s/batch | 217.331-227.180 s | 83.3%, 83.5%, 83.5%, 84.6% | 3.26 | 2.346 | 0.925 | 0.643 at step 9430 |

Late-window details:

| GPUs | Mean RMSE_e | Mean RMSE_f | Mean RMSE_v | Best late RMSE_e | Best late RMSE_f | Best late RMSE_v |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0843 | 1.2847 | 0.0676 | 0.00283 | 0.842 | 0.0235 |
| 2 | 0.0485 | 1.0412 | 0.0593 | 0.000535 | 0.533 | 0.0236 |
| 4 | 0.0709 | 0.9245 | 0.0526 | 0.00164 | 0.432 | 0.0172 |

### 10k Loss Scaling Interpretation

- Multi-GPU TF/Horovod did not reduce per-optimizer-step time in this setup. Per-step timing stayed near 0.02 s/batch for 1, 2, and 4 GPUs.
- Multi-GPU did improve the late-window training loss relative to 1 GPU.
- 2 GPUs had the best late-window total RMSE and energy RMSE.
- 4 GPUs had the best late-window force and virial RMSE and the best single late-window RMSE.
- Final-step RMSE is noisy because individual minibatches can spike. The 9000-10000 mean and best-late metrics are more useful for comparing training behavior.
- GPU utilization was real on all ranks. The 4-GPU run kept all GPUs around 83-85% active utilization during training.

## 4-GPU CPU Sweep

These are 1000-step runs with 4 GPU ranks and varying total allocated CPUs. Since 4-GPU Horovod uses 4 ranks, the tested total CPU counts map to `cpus-per-task` as:

| Total CPUs | Tasks | CPUs/task |
|---:|---:|---:|
| 4 | 4 | 1 |
| 8 | 4 | 2 |
| 16 | 4 | 4 |
| 32 | 4 | 8 |

| Total CPUs | Job ID | Slurm elapsed | Avg train time | Rank wall-time range | Avg active GPU util | Final RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 7261214 | 00:04:35 | 0.0309-0.0319 s/batch | 38.164-41.636 s | 53.2% | 2.76 |
| 8 | 7261213 | 00:02:49 | 0.0210-0.0219 s/batch | 26.367-29.059 s | 65.1% | 2.76 |
| 16 | 7261212 | 00:02:41 | 0.0201-0.0208 s/batch | 24.786-29.336 s | 66.4% | 2.76 |
| 32 | 7261215 | 00:02:40 | 0.0200-0.0208 s/batch | 24.783-35.052 s | 67.7% | 2.76 |

### CPU Sweep Interpretation

- 4 total CPUs, meaning 1 CPU per rank/GPU, is too low. It slows training to about 0.031 s/batch and leaves GPUs under-utilized around 53%.
- 8 total CPUs, meaning 2 CPUs per rank/GPU, gets most of the benefit.
- 16 total CPUs is the practical sweet spot for this 4-GPU TF/Horovod setup.
- 32 total CPUs gives little additional benefit over 16 and can show more rank wall-time imbalance.

Recommended default for similar Della A100 TF/Horovod runs: 4 GPUs, 4 tasks, 4 CPUs/task, 120 GB memory, 60-minute test walltime for benchmarks.

## Quick 500-Step Sanity Checks

Before the 10k loss runs, shorter checks were used to validate the launcher and GPU visibility.

| Config | Job ID | Result |
|---|---:|---|
| 1 GPU | 7260110 | Completed. Average training time 0.0197 s/batch, wall time 14.540 s. |
| 2 GPUs | 7260352 | Completed. Horovod size 2, both GPUs visible. Average training time 0.0199-0.0207 s/batch. |
| 2 GPUs with monitor | 7260575 | Completed. Both GPUs active during training. GPU0 average memory about 4603 MiB, GPU1 about 2814 MiB; average GPU utilization about 76% and 74% over the active training window. |

These short runs confirmed that 2 GPUs were actually used, but they also showed no step-time speedup. The longer 10k runs above are the better comparison for loss behavior.

## Stored Artifacts

This folder stores raw artifacts needed for reproducibility and later inspection. It deliberately omits `model-compression/` checkpoint directories because they are large and not needed for benchmark interpretation.

```text
tf_horovod_apptainer300cuda126_20260422/
|-- SUMMARY.md
|-- loss_scaling_10k/
|   |-- 1gpu/
|   |-- 2gpu/
|   `-- 4gpu/
|-- cpu_sweep_4gpu_1000/
|   |-- cpu_total4/
|   |-- cpu_total8/
|   |-- cpu_total16/
|   `-- cpu_total32/
`-- quick_checks_500step/
    |-- 1gpu/
    |-- 2gpu/
    `-- 2gpu_mem/
```

Each result subdirectory contains some combination of:

- `myinput.json`: exact training input.
- `run_srun_train_mem.sbatch` or `run_srun_train.sbatch`: exact submission script.
- `lcurve.out`: training loss curve.
- `slurm-<jobid>.out`: full Slurm/DeePMD log.
- `gpu_mem_util_<jobid>.csv`: `nvidia-smi` monitor output, where collected.

## Full Run Locations

The complete original run directories, including checkpoints, remain under:

```text
/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/
```

Key subdirectories:

- `10k_gpu_scaling_loss_rerun/`
- `4gpu_cpu_sweep_1000/`
- `1gpu/`, `2gpu/`, `2gpu_mem/` for the 500-step launcher checks.
