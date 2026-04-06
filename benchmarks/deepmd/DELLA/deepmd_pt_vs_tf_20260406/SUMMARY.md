# DeePMD-kit Benchmark: PyTorch DDP vs TensorFlow (Single-GPU)

**Date:** 2026-04-06
**Cluster:** Della (Princeton), A100 80GB GPUs
**Author:** Akash Gupta (with Claude Code assistance)

## Purpose

Compare training performance of DeePMD-kit using:
- **TensorFlow backend** (single-GPU only, `ALCHEMY_env`)
- **PyTorch backend** (single and multi-GPU via DDP, `ALCHEMY_env__PT`)

The goal is to determine whether PyTorch DDP multi-GPU training provides a
net speedup over the existing TF single-GPU workflow, despite PT being slower
per-batch on a single GPU.

## Environment

| Component        | TF Environment (`ALCHEMY_env`) | PT Environment (`ALCHEMY_env__PT`) |
|------------------|-------------------------------|-------------------------------------|
| DeePMD-kit       | v3.1.3                        | v3.1.3                              |
| Backend          | TensorFlow 2.21               | PyTorch 2.6.0 (conda-forge)        |
| CUDA             | 12.8 (system module)          | 12.8 (system module)                |
| GCC              | 14 (gcc-toolset/14)           | 14 (gcc-toolset/14)                 |
| Multi-GPU        | Not available (horovod fails) | PyTorch DDP via torchrun            |
| Install script   | `install_deepmd-kit_w_plumed_lmp_and_others___for_ALCHEMY.sh` | `install_deepmd-kit_PT_w_plumed_lmp_and_others___for_ALCHEMY.sh` |

## Model Configuration

- **Descriptor:** `se_e2_a` (smooth edition, two-body, all atoms)
- **rcut:** 6.0 A
- **sel:** [800, 150]
- **Descriptor neurons:** [25, 50, 100]
- **Fitting neurons:** [240, 240, 240]
- **Training steps:** 500 (benchmark only)
- **Batch size:** auto
- **Training systems:** 627
- **Dataset:** NH3-H2 mixture QMD data

## Results

### 1-GPU (PT single-GPU, varying CPUs)

| CPUs | OMP_NUM_THREADS | s/batch | vs TF baseline |
|------|-----------------|---------|----------------|
| 1    | 1 (default)     | 0.0353  | 1.61x slower   |
| 4    | 4               | 0.0341  | 1.56x slower   |
| 8    | 8               | 0.0330  | 1.51x slower   |

### 2-GPU DDP (varying CPUs per GPU)

| CPUs/GPU | OMP_NUM_THREADS | s/batch | Effective throughput* |
|----------|-----------------|---------|----------------------|
| 1        | 1 (default)     | 0.0442  | 0.99x                |
| 2        | 2               | 0.0358  | 1.22x                |
| 4        | 4               | 0.0366  | 1.20x                |
| 8        | 8               | 0.0376  | 1.16x                |

### 4-GPU DDP (varying CPUs per GPU)

| CPUs/GPU | OMP_NUM_THREADS | s/batch | Effective throughput* |
|----------|-----------------|---------|----------------------|
| 1        | 1 (default)     | 0.0422  | 2.08x                |
| 2        | 2               | 0.0367  | 2.39x                |
| 4        | 4               | 0.0376  | 2.33x                |
| 8        | 8               | 0.0381  | 2.30x                |

### TF Baseline

| Config | s/batch | Wall time (500 batches) |
|--------|---------|------------------------|
| TF 1-GPU, 1 CPU | 0.0219 | 13.7 s |

*Effective throughput = N_GPUs * (TF_baseline_s_per_batch / PT_s_per_batch).
This accounts for data parallelism: each GPU processes its own batch, so total
data processed per step scales linearly with GPU count.

## Key Findings

### 1. PT is ~1.5-1.6x slower per-batch than TF on single GPU
For the `se_e2_a` descriptor, PT single-GPU (0.0353 s/batch) is 61% slower
than TF (0.0219 s/batch). This is a known trade-off — PT's flexibility and
DDP support come at the cost of single-GPU throughput for this model type.

### 2. More CPUs provide modest benefit for single-GPU (~7%)
Going from 1 to 8 CPUs on a single GPU improves from 0.0353 to 0.0330 s/batch.
The benefit is small because the bottleneck is GPU compute, not CPU data loading
for this model.

### 3. DDP scales well — 4 GPUs achieves ~2.4x TF-baseline throughput
With 4 GPUs at 2 CPUs/GPU, 0.0367 s/batch translates to 2.39x effective
throughput vs TF single-GPU. This is genuine speedup for training.

### 4. For DDP, 2 CPUs/GPU is the sweet spot
- **2-GPU DDP:** Best at 2 CPUs/GPU (0.0358 s/batch). Adding more CPUs
  actually *hurts* slightly (0.0366 at 4, 0.0376 at 8) — likely due to
  OMP thread contention with PyTorch's own threading.
- **4-GPU DDP:** Best at 2 CPUs/GPU (0.0367 s/batch), with 4 and 8
  CPUs/GPU showing no further gain (0.0376, 0.0381).
- **Key insight:** Going from 1 to 2 CPUs/GPU gives the biggest jump
  (0.0442 → 0.0358 for 2-GPU = 19% improvement; 0.0422 → 0.0367 for
  4-GPU = 13% improvement). Beyond 2 CPUs/GPU, returns diminish or reverse.

### 5. LAMMPS works directly with PT models
LAMMPS is built with DeePMD C API (`deepmd_c`) which supports the PT backend
natively. No model conversion from `.pt` to `.pb` is needed.

## Section B: Network Size Scaling (10x, 100x)

To test whether the PT/TF training gap narrows with larger networks, we scaled
the neuron widths by ~3x (10x params) and ~10x (100x params):

| Scale | Desc. neurons | Fitting neurons | Params |
|-------|---------------|-----------------|--------|
| 1x (baseline) | [25, 50, 100] | [240, 240, 240] | ~1M |
| 10x | [75, 150, 300] | [720, 720, 720] | 9.2M |
| 100x | [250, 500, 1000] | [2400, 2400, 2400] | 102.4M |

All runs: 500 steps, batch_size auto, 2 CPUs/GPU for DDP.

### Results: PT/TF Single-GPU Ratio

| Scale | TF 1-GPU (s/batch) | PT 1-GPU (s/batch) | PT/TF ratio |
|-------|--------------------|--------------------|-------------|
| 1x    | 0.0219             | 0.0353             | 1.61x       |
| 10x   | 0.0545             | 0.0651             | 1.19x       |
| 100x  | 0.2165             | 0.2341             | 1.08x       |

### Results: PT DDP Effective Throughput vs TF 1-GPU

| Scale | PT 1-GPU | PT 2-GPU DDP | PT 4-GPU DDP |
|-------|----------|--------------|--------------|
| 1x    | 0.62x    | 1.22x        | 2.39x        |
| 10x   | 0.84x    | 1.56x        | 3.09x        |
| 100x  | 0.92x    | 1.68x        | 3.45x        |

*Effective throughput = N_GPUs × (TF_s_per_batch / PT_s_per_batch)

### Results: Full Timing Table

| Config | Backend | GPUs | s/batch | PT/TF ratio | Eff. throughput |
|--------|---------|------|---------|-------------|-----------------|
| tf_1gpu_10x | TF | 1 | 0.0545 | — | 1.00x (baseline) |
| pt_1gpu_10x | PT | 1 | 0.0651 | 1.19x | 0.84x |
| pt_2gpu_10x | PT | 2 | 0.0697 | 1.28x | 1.56x |
| pt_4gpu_10x | PT | 4 | 0.0706 | 1.30x | 3.09x |
| tf_1gpu_100x | TF | 1 | 0.2165 | — | 1.00x (baseline) |
| pt_1gpu_100x | PT | 1 | 0.2341 | 1.08x | 0.92x |
| pt_2gpu_100x | PT | 2 | 0.2575 | 1.19x | 1.68x |
| pt_4gpu_100x | PT | 4 | 0.2509 | 1.16x | 3.45x |

### Key Findings: Network Size Scaling

**6. PT/TF gap vanishes for large networks**
The single-GPU PT/TF ratio drops from 1.61x (1M params) → 1.19x (9.2M) →
1.08x (102.4M). At 100x network size, PT is only 8% slower than TF on a
single GPU. This confirms that PyTorch's dispatch overhead (JIT tracing,
tensor allocation, CUDA kernel launch) dominates for small networks but
becomes negligible when GPU compute dominates.

**7. PT DDP advantage grows with network size**
With 4-GPU DDP, effective throughput vs TF 1-GPU increases from 2.39x (1x)
→ 3.09x (10x) → 3.45x (100x). For large networks, PT 4-GPU DDP provides
a genuine ~3.5x speedup over TF single-GPU — approaching the theoretical
4x linear scaling.

**8. At 100x, PT 2-GPU already beats TF 1-GPU by 1.68x**
For large networks, even 2 GPUs with PT DDP substantially outperform
TF single-GPU, making the PT backend the clear choice for training
large models.

## Recommendations

- **For small networks (~1M params):** TF single-GPU is faster. Use
  `ALCHEMY_env` if you only need 1 GPU.
- **For medium networks (~10M params):** PT 4-GPU DDP gives ~3.1x
  throughput vs TF 1-GPU. PT is worthwhile if you have multi-GPU access.
- **For large networks (>100M params):** PT is nearly as fast as TF
  per-GPU (only 8% slower). PT 4-GPU DDP gives ~3.5x throughput.
  Use `ALCHEMY_env__PT` with DDP for production training.
- **Optimal DDP config:** 2 CPUs/GPU is the sweet spot (from Section A).
- **Batch size note:** With DDP, effective batch_size = batch_size * N_GPUs.
  Reduce `decay_steps` by ~N_GPUs in `input.json` for equivalent convergence.

## File Organization

```
benchmarks/deepmd_pt_vs_tf_20260406/
├── SUMMARY.md                  # This file
├── sbatch_files/               # All submission scripts
│   ├── tf_1gpu_baseline.sbatch       # Section A: 1x network
│   ├── pt_1gpu.sbatch
│   ├── pt_1gpu_2cpu.sbatch
│   ├── pt_1gpu_4cpu.sbatch
│   ├── pt_1gpu_8cpu.sbatch
│   ├── pt_1gpu_16cpu.sbatch
│   ├── pt_2gpu.sbatch
│   ├── pt_2gpu_2cpupergpu.sbatch
│   ├── pt_2gpu_4cpupergpu.sbatch
│   ├── pt_2gpu_8cpupergpu.sbatch
│   ├── pt_4gpu.sbatch
│   ├── pt_4gpu_2cpupergpu.sbatch
│   ├── pt_4gpu_4cpupergpu.sbatch
│   ├── pt_4gpu_8cpupergpu.sbatch
│   ├── tf_1gpu_10x.sbatch            # Section B: network size scaling
│   ├── pt_1gpu_10x.sbatch
│   ├── pt_2gpu_10x.sbatch
│   ├── pt_4gpu_10x.sbatch
│   ├── tf_1gpu_100x.sbatch
│   ├── pt_1gpu_100x.sbatch
│   ├── pt_2gpu_100x.sbatch
│   └── pt_4gpu_100x.sbatch
├── myinput.json                # 1x training input
├── myinput_10x.json            # 10x training input
└── myinput_100x.json           # 100x training input
```

## Source Data Locations

- **PT benchmark runs (1x):** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/pt_bench_20260405/`
- **TF baseline run (1x):** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_bench_20260403/1gpu_disp10_skipns__baseline__ALCHEMY_env/`
- **Network size scaling runs (10x, 100x):** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/netsize_bench_20260406/`
- **PT install script:** `/scratch/gpfs/BURROWS/akashgpt/softwares/installing_MLMD_related_stuff/install_deepmd-kit_PT_w_plumed_lmp_and_others___for_ALCHEMY.sh`
- **TF install script:** `/scratch/gpfs/BURROWS/akashgpt/softwares/installing_MLMD_related_stuff/install_deepmd-kit_w_plumed_lmp_and_others___for_ALCHEMY.sh`

## Why No TensorFlow in the PT Environment?

TensorFlow and conda-forge PyTorch cannot coexist for C++ compilation due to
a protobuf ABI conflict. TF headers use `FullTypeDef` inheriting from
`google::protobuf::Message`, but conda-forge protobuf has an incompatible
`GetClassData()` pure virtual function. Since conda-forge PyTorch is required
(for CXX11 ABI compatibility with GCC 14), TF must be excluded. See the
install script header for full details.
