# LAMMPS Benchmark: PyTorch vs TensorFlow Backend (DeePMD-kit)

**Date:** 2026-04-06
**Cluster:** Della (Princeton), A100 80GB GPUs
**Author:** Akash Gupta (with Claude Code assistance)

## Purpose

Compare LAMMPS MD inference performance using DeePMD-kit with:
- **TensorFlow backend** (`ALCHEMY_env`, compressed `.pb` model)
- **PyTorch backend** (`ALCHEMY_env__PT`, compressed `.pth` model)

Both use model compression for fair comparison. Multi-GPU uses MPI domain
decomposition (1 MPI rank per GPU).

## System

- **Base composition:** 40 H2 + 40 NH3 (240 atoms)
- **Conditions:** ~6000 K, ~3350 GPa (planetary interior)
- **Descriptor:** `se_e2_a`, rcut=6.0, sel=[800,150]
- **Descriptor neurons:** [25, 50, 100]; **Fitting neurons:** [240, 240, 240]
- **Configuration file:** `conf.lmp` (from ongoing simulation)

Two benchmark sets were run:
1. **Production-like** (Section A): 240 atoms, 50000 MTMB steps, with PLUMED
2. **Size-scaling** (Section B): 240 / 1920 / 15360 atoms, 1000 NVE steps, no PLUMED

## Environment

| Component        | TF (`ALCHEMY_env`)   | PT (`ALCHEMY_env__PT`)          |
|------------------|---------------------|---------------------------------|
| DeePMD-kit       | v3.1.3              | v3.1.3                          |
| Backend          | TensorFlow 2.21     | PyTorch 2.6.0 (conda-forge)    |
| Model            | pv_comp.pb (compressed) | model_comp.pth (compressed)  |
| CUDA             | 12.8                | 12.8                            |
| LAMMPS           | stable_2Aug2023_update3 | stable_2Aug2023_update3     |
| PLUMED           | 2.8.2               | 2.8.2                           |
| Multi-GPU        | MPI domain decomposition | MPI domain decomposition   |

## Results

### Section A: Production-like (240 atoms, 50000 MTMB steps, with PLUMED)

| Config | Backend | GPUs | Loop time (s) | ms/step | Speedup vs TF-1GPU |
|--------|---------|------|---------------|---------|---------------------|
| tf_1gpu | TF (compressed .pb) | 1 | 404.1 | 8.08 | 1.00x |
| tf_2gpu | TF (compressed .pb) | 2 | 294.6 | 5.89 | 1.37x |
| tf_4gpu | TF (compressed .pb) | 4 | 227.1 | 4.54 | 1.78x |
| pt_1gpu | PT (compressed .pth) | 1 | 1426.1 | 28.52 | 0.28x |
| pt_2gpu | PT (compressed .pth) | 2 | 986.2 | 19.72 | 0.41x |
| pt_4gpu | PT (compressed .pth) | 4 | 762.5 | 15.25 | 0.53x |

### Section B: Size-scaling (240 / 1920 / 15360 atoms, 1000 NVE steps, no PLUMED)

| Atoms | Backend | GPUs | Loop time (s) | ms/step | PT/TF ratio | GPU scaling |
|-------|---------|------|---------------|---------|-------------|-------------|
| 240   | TF      | 1    | 3.81          | 3.81    | —           | 1.00x       |
| 240   | TF      | 2    | 2.87          | 2.87    | —           | 1.33x       |
| 240   | TF      | 4    | 2.88          | 2.88    | —           | 1.32x       |
| 240   | PT      | 1    | 11.92         | 11.92   | 3.13x       | 1.00x       |
| 240   | PT      | 2    | 9.55          | 9.55    | 3.33x       | 1.25x       |
| 240   | PT      | 4    | 9.15          | 9.15    | 3.18x       | 1.30x       |
| 1920  | TF      | 1    | 18.57         | 18.57   | —           | 1.00x       |
| 1920  | TF      | 2    | 9.96          | 9.96    | —           | 1.86x       |
| 1920  | TF      | 4    | 6.50          | 6.50    | —           | 2.86x       |
| 1920  | PT      | 1    | 41.49         | 41.49   | 2.24x       | 1.00x       |
| 1920  | PT      | 2    | 24.25         | 24.25   | 2.44x       | 1.71x       |
| 1920  | PT      | 4    | 16.62         | 16.62   | 2.56x       | 2.50x       |
| 15360 | TF      | 1    | 154.6         | 154.6   | —           | 1.00x       |
| 15360 | TF      | 2    | 76.1          | 76.1    | —           | 2.03x       |
| 15360 | TF      | 4    | 39.3          | 39.3    | —           | 3.93x       |
| 15360 | PT      | 1    | 321.3         | 321.3   | 2.08x       | 1.00x       |
| 15360 | PT      | 2    | 151.1         | 151.1   | 1.98x       | 2.13x       |
| 15360 | PT      | 4    | 77.2          | 77.2    | 1.96x       | 4.16x       |

**PT/TF ratio summary by system size:**

| Atoms | 1 GPU | 2 GPU | 4 GPU | Average |
|-------|-------|-------|-------|---------|
| 240   | 3.13x | 3.33x | 3.18x | 3.21x   |
| 1920  | 2.24x | 2.44x | 2.56x | 2.41x   |
| 15360 | 2.08x | 1.98x | 1.96x | 2.01x   |

## Key Findings

### 1. PT/TF gap is system-size dependent: 3.2x → 2.0x
The performance gap between PT and TF narrows significantly with system size:
- **240 atoms:** PT is ~3.2x slower (dispatch/JIT overhead dominates)
- **1920 atoms:** PT is ~2.4x slower (transitional regime)
- **15360 atoms:** PT is ~2.0x slower (GPU compute dominates)

This confirms that PyTorch's per-call dispatch overhead (JIT tracing, tensor
allocation, CUDA kernel launch) is the primary source of the gap at small
system sizes. As the system grows and GPU compute dominates, the gap narrows
to ~2x — consistent with literature reports of 1.5-2.5x.

### 2. Multi-GPU scaling improves with system size
For the small 240-atom system, neither backend scales beyond ~1.3x on 2-4 GPUs
(MPI communication overhead dominates). For larger systems:
- **1920 atoms:** TF 1.86x/2.86x, PT 1.71x/2.50x (2/4 GPU)
- **15360 atoms:** TF 2.03x/3.93x, PT 2.13x/4.16x (2/4 GPU)

At 15360 atoms, both backends achieve near-linear scaling, with PT slightly
exceeding TF's GPU scaling efficiency (4.16x vs 3.93x on 4 GPUs).

### 3. TF is ~3.5x faster at production size (240 atoms, with PLUMED)
In Section A (production-like conditions), TF achieves 8.08 ms/step vs PT's
28.52 ms/step on a single GPU — a 3.53x difference. Even with 4 GPUs, PT
(15.25 ms/step) is still 1.89x slower than TF on 1 GPU.

### 4. At large system sizes, PT 4-GPU matches TF 2-GPU
For 15360 atoms, PT on 4 GPUs (77.2s) is comparable to TF on 2 GPUs (76.1s).
This means that for large systems, the GPU cost of using the PT backend is
roughly 2x (need twice as many GPUs for equivalent throughput).

### 5. Model note
The PT model was trained for only 100 steps (for benchmarking purposes).
The TF model (`pv_comp.pb`) is a fully converged, compressed model from
production training. While the model weights differ, the architecture is
identical (`se_e2_a` with same sel/rcut/neuron), so the inference speed
comparison is valid — inference time depends on architecture, not weights.

## Recommendations

- **For LAMMPS production runs with small systems (<1000 atoms):** Use
  `ALCHEMY_env` (TF backend). The 3-3.5x speed advantage makes TF the clear
  choice where dispatch overhead dominates.
- **For LAMMPS production runs with large systems (>10000 atoms):** TF is
  still faster (~2x), but the gap is smaller. PT is viable if the workflow
  benefit (no backend conversion) justifies the ~2x cost.
- **For multi-GPU training:** Use `ALCHEMY_env__PT` (PT backend with DDP).
  See the companion training benchmark for details.
- **Workflow:** Train with PT (multi-GPU DDP) → convert model to .pb →
  Run LAMMPS with TF. Currently `dp convert-backend` requires both TF and
  PT in the same environment. As a workaround, train with TF for now if
  LAMMPS performance is critical.
- **Multi-GPU scaling:** For large systems (15360 atoms), both backends
  scale near-linearly. Multi-GPU runs are worthwhile for systems >1000 atoms.
  For small systems (240 atoms), multi-GPU MPI decomposition provides
  diminishing returns.

## File Organization

```
benchmarks/lammps_pt_vs_tf_20260406/
├── SUMMARY.md                  # This file
├── in.lammps.bench             # LAMMPS input file (50000 MTMB steps)
├── plumed.dat                  # PLUMED input for metadynamics
└── sbatch_files/
    ├── tf_1gpu.sbatch          # Section A: production-like (with PLUMED)
    ├── tf_2gpu.sbatch
    ├── tf_4gpu.sbatch
    ├── pt_1gpu.sbatch
    ├── pt_2gpu.sbatch
    ├── pt_4gpu.sbatch
    ├── tf_1gpu_240_noplumed.sbatch   # Section B: size-scaling
    ├── tf_2gpu_240_noplumed.sbatch
    ├── tf_4gpu_240_noplumed.sbatch
    ├── pt_1gpu_240_noplumed.sbatch
    ├── pt_2gpu_240_noplumed.sbatch
    ├── pt_4gpu_240_noplumed.sbatch
    ├── tf_1gpu_2x2x2.sbatch
    ├── tf_2gpu_2x2x2.sbatch
    ├── tf_4gpu_2x2x2.sbatch
    ├── pt_1gpu_2x2x2.sbatch
    ├── pt_2gpu_2x2x2.sbatch
    ├── pt_4gpu_2x2x2.sbatch
    ├── tf_1gpu_4x4x4.sbatch
    ├── tf_2gpu_4x4x4.sbatch
    ├── tf_4gpu_4x4x4.sbatch
    ├── pt_1gpu_4x4x4.sbatch
    ├── pt_2gpu_4x4x4.sbatch
    └── pt_4gpu_4x4x4.sbatch
```

## Source Data Locations

- **Benchmark runs:** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/md/ZONE_1/40H2_40NH3/lmp_bench_20260406/`
- **TF compressed model:** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train/model-compression/pv_comp.pb`
- **PT compressed model:** `.../lmp_bench_20260406/pt_model_gen/model_comp.pth`
- **Previous benchmarks (dp_plmd_della env):** `.../40H2_40NH3/slurm-5925*.out`

## Comparison with Literature

The PT/TF LAMMPS inference gap is **system-size dependent**, ranging from
~3.2x (240 atoms) to ~2.0x (15360 atoms). The large-system result (~2x)
is **consistent with** the 1.5-2.5x range reported in DeePMD-kit GitHub
discussions (issues #3986, #4010, #3853). The elevated gap at small system
sizes is explained by PT dispatch overhead, which is well-documented:

1. **System-size dependence (confirmed):** At 240 atoms, PT's per-call
   dispatch overhead (JIT tracing, tensor allocation, CUDA kernel launch)
   dominates, inflating the gap to 3.2x. At 15360 atoms, GPU compute
   dominates and the gap converges to ~2.0x. This is the primary factor.

2. **se_e2_a descriptor:** This is the classic "smooth edition" descriptor.
   TF's graph-mode and XLA optimizations are particularly effective for this
   descriptor type. Newer descriptors (DPA-1, DPA-2) may show smaller gaps.

3. **Network size vs overhead ratio:** With fitting neurons [240,240,240]
   and descriptor neurons [25,50,100], the neural network compute per
   evaluation is moderate. For larger networks (e.g., DPA-2 with attention),
   GPU compute dominates further and the PT/TF gap should shrink even more.

4. **Model compression maturity:** TF model compression is more mature and
   may produce more efficient inference kernels. PT compression was added
   later in the v3 development cycle.

5. **Environment tuning:** Neither run set `DP_INTRA_OP_PARALLELISM_THREADS`,
   `DP_INTER_OP_PARALLELISM_THREADS`, or `OMP_NUM_THREADS`. Tuning these
   may improve both backends but potentially PT more.

6. **Multi-GPU scaling at large sizes:** Both backends achieve near-linear
   scaling at 15360 atoms (TF 3.93x, PT 4.16x on 4 GPUs), which is
   expected for systems large enough to fully utilize each GPU.

References:
- deepmodeling/deepmd-kit#3986, #4010, #3853
- https://docs.deepmodeling.com/projects/deepmd/en/stable/backend.html

## Related

- Training benchmarks: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/benchmarks/deepmd_pt_vs_tf_20260406/`
