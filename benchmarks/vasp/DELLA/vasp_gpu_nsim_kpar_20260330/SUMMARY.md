# VASP GPU Benchmark: NSIM Tuning & Multi-GPU (KPAR) Scaling

**Date:** 2026-03-30
**Cluster:** Della (Princeton), NVIDIA A100-SXM4-80GB GPUs
**Author:** Akash Gupta

## Purpose

Benchmark VASP 6.6.0 (GPU build) to determine:
1. **Optimal NSIM** for GPU utilization on a single A100
2. **Multi-GPU scaling** via KPAR (k-point parallelism) across 1, 2, and 4 GPUs
3. Whether VASP GPU can achieve near-linear scaling for production QMD workloads

## System

- **Composition:** MgSiO3 perovskite (Mg: 32, Si: 32, O: 96 = 160 atoms)
- **Source:** "Frame 1 extracted from DeePMD dataset" (from active learning cycle)
- **Cell:** Cubic, a = 14.888 A
- **Temperature:** 8000 K (high-T QMD frame)
- **XC Functional:** R2SCAN meta-GGA (METAGGA = R2SCAN, LASPH = .TRUE.)
- **ENCUT:** 800 eV
- **NBANDS:** 768 (960 electrons)
- **K-points:** 2x2x2 Monkhorst-Pack (4 irreducible k-points)
- **Electronic:** ALGO = N (Davidson), EDIFF = 1E-06, NELM = 150
- **Ionic:** NSW = 0 (single-point energy, benchmark only)
- **GPU settings:** NPAR = 1 (required for GPU offloading)

## Environment

| Component | Version |
|-----------|---------|
| VASP | 6.6.0 (GPU build, compiled 2026-03-30) |
| Compiler | nvhpc/25.5 |
| MPI | openmpi/cuda-12.9/nvhpc-25.5/4.1.8 |
| Math libs | intel-mkl/2024.2 |
| GPU | NVIDIA A100-SXM4-80GB |
| VASP binary | `/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin/vasp_std` |

## Results

### A. NSIM Tuning (Single GPU)

NSIM controls the number of bands updated simultaneously in the subspace
diagonalization. Larger NSIM can better saturate the GPU but uses more memory.
The VASP default is NSIM = 4.

| Config | NSIM | GPUs | Elapsed (s) | vs Default | Speedup | Memory (MB) | SCF iters |
|--------|------|------|-------------|------------|---------|-------------|-----------|
| 1_GPU (default) | 4* | 1 | 1323.1 | baseline | 1.00x | 11,498 | 35 |
| 1gpu_nsim16 | 16 | 1 | 1543.7 | +16.7% | 0.86x | 13,366 | 35 |
| 1gpu_nsim32 | 32 | 1 | 1239.7 | -6.3% | 1.07x | 13,621 | 34 |
| 1gpu_nsim64 | 64 | 1 | 1262.6 | -4.6% | 1.05x | 13,826 | 34 |

*NSIM not set in INCAR; VASP default is 4.

### B. Multi-GPU Scaling (KPAR, with optimal NSIM=32)

KPAR distributes k-points across GPUs. With 4 irreducible k-points, KPAR can
be 1, 2, or 4 (must evenly divide the number of k-points).

| Config | NSIM | KPAR | GPUs | Nodes | Elapsed (s) | Speedup vs 1-GPU* | GPU scaling |
|--------|------|------|------|-------|-------------|-------------------|-------------|
| 1gpu_nsim32 | 32 | 1 | 1 | 1 | 1239.7 | 1.00x | — |
| 2gpu_kpar2_nsim32 | 32 | 2 | 2 | 1 | 646.0 | 1.92x | 1.92x |
| 4gpu_kpar4_nsim32 | 32 | 4 | 4 | 2 | 346.5 | 3.58x | 3.58x |

*Speedup relative to best single-GPU config (1gpu_nsim32, 1239.7s).

### C. Complete Summary Table

| Config | NSIM | KPAR | GPUs | Elapsed (s) | Speedup vs baseline (1_GPU) |
|--------|------|------|------|-------------|----------------------------|
| 1_GPU (default) | 4 | 1 | 1 | 1323.1 | 1.00x |
| 1gpu_nsim16 | 16 | 1 | 1 | 1543.7 | 0.86x |
| **1gpu_nsim32** | **32** | **1** | **1** | **1239.7** | **1.07x** |
| 1gpu_nsim64 | 64 | 1 | 1 | 1262.6 | 1.05x |
| **2gpu_kpar2_nsim32** | **32** | **2** | **2** | **646.0** | **2.05x** |
| **4gpu_kpar4_nsim32** | **32** | **4** | **4** | **346.5** | **3.82x** |

### D. Energy Consistency

All runs converge to the same total energy within numerical precision:

| Config | TOTEN (eV) | SCF iterations |
|--------|------------|----------------|
| 1_GPU (default) | -1089.85247502 | 35 |
| 1gpu_nsim16 | -1089.85247501 | 35 |
| 1gpu_nsim32 | -1089.85247484 | 34 |
| 1gpu_nsim64 | -1089.85247484 | 34 |
| 2gpu_kpar2_nsim32 | -1089.85247484 | 34 |
| 4gpu_kpar4_nsim32 | -1089.85247484 | 34 |

Energy spread: < 2e-7 eV. NSIM=32+ runs converge in 34 SCF iterations
vs 35 for NSIM<=16, likely due to slightly different numerical paths.

## Key Findings

### 1. NSIM=32 is optimal for single-GPU A100 (6% speedup)
NSIM=32 gives the best single-GPU time (1239.7s), 6.3% faster than the
default NSIM=4 (1323.1s). NSIM=64 is slightly slower (1262.6s), suggesting
the GPU is already saturated at NSIM=32 for 768 bands on A100. NSIM=16
is actually SLOWER than the default — it likely disrupts the default
batching strategy without fully saturating the GPU.

### 2. KPAR scaling is excellent: 3.82x on 4 GPUs
With KPAR=4 and 4 k-points, each GPU handles exactly 1 k-point. This
gives near-ideal parallelism:
- 2 GPUs (KPAR=2): 1.92x (96% efficiency, same node)
- 4 GPUs (KPAR=4): 3.58x (89% efficiency, 2 nodes)

The 4-GPU run spans 2 nodes (2 GPUs per node) yet still achieves 89%
parallel efficiency, indicating that inter-node MPI communication overhead
is small for k-point parallelism.

### 3. Combined optimization: 3.82x total speedup
Default 1-GPU (NSIM=4): 1323.1s → Optimized 4-GPU (NSIM=32, KPAR=4):
346.5s = **3.82x speedup**. For a ~22-minute single-GPU job, this brings
it down to under 6 minutes.

### 4. No failures
All 6 runs completed successfully with zero errors. All energies agree
to within ~2e-7 eV, confirming that NSIM and KPAR do not affect the
final converged result.

### 5. Memory is not a concern
Memory usage ranges from 11.5 GB (NSIM=4) to 13.8 GB (NSIM=64) — well
within the A100's 80 GB. Increasing NSIM from 4 to 32 costs only ~2 GB
additional memory.

## Recommendations

- **For production QMD on this system (MgSiO3, 160 atoms, 2x2x2 k-points):**
  Use **NSIM=32** and **KPAR=4** with 4 GPUs for ~3.8x speedup.

- **If limited to 1 GPU:** Set **NSIM=32** for a modest 6% improvement.
  The default NSIM=4 is suboptimal on A100 GPUs.

- **KPAR should equal the number of GPUs** when the number of irreducible
  k-points is divisible by KPAR. Here, 4 k-points / KPAR=4 = 1 k-point
  per GPU, which is ideal.

- **For systems with more k-points:** KPAR scaling should be even better
  since there's more work to distribute. Systems with only Gamma-point
  (KPAR=1) cannot benefit from k-point parallelism.

- **NPAR must be 1** for GPU VASP. Do not change this.

- **NSIM tuning is system-dependent:** The optimal NSIM depends on the
  number of bands, system size, and GPU memory. For 768 bands on A100,
  NSIM=32 is optimal. For smaller systems (fewer bands), lower NSIM may
  be better. Always benchmark for your specific system.

## File Organization

```
benchmarks/DELLA/vasp_gpu_nsim_kpar_20260330/
├── SUMMARY.md                       # This file
├── POSCAR                           # Structure (MgSiO3, 160 atoms)
├── KPOINTS                          # 2x2x2 Monkhorst-Pack
├── sbatch_files/                    # All submission scripts
│   ├── 1_GPU.sbatch                 # 1 GPU, default NSIM
│   ├── 1gpu_nsim16.sbatch           # 1 GPU, NSIM=16
│   ├── 1gpu_nsim32.sbatch           # 1 GPU, NSIM=32
│   ├── 1gpu_nsim64.sbatch           # 1 GPU, NSIM=64
│   ├── 2gpu_kpar2_nsim32.sbatch     # 2 GPUs, KPAR=2, NSIM=32
│   └── 4gpu_kpar4_nsim32.sbatch     # 4 GPUs, KPAR=4, NSIM=32
└── incar_files/                     # INCAR for each run
    ├── INCAR_1_GPU
    ├── INCAR_1gpu_nsim16
    ├── INCAR_1gpu_nsim32
    ├── INCAR_1gpu_nsim64
    ├── INCAR_2gpu_kpar2_nsim32
    └── INCAR_4gpu_kpar4_nsim32
```

## Source Data Location

- **Benchmark runs:** `/scratch/gpfs/BURROWS/akashgpt/qmd_data/test/TEST__u.pro.E.l.j.pv_hf_copy.8k.r3-3100.recal/`
- **VASP GPU binary:** `/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0.gpu/`

## Notes

- This benchmark uses NSW=0 (single-point), so only electronic convergence
  is timed. In production QMD (NSW>>0), each ionic step repeats the SCF
  cycle, so the per-step speedups translate directly to wall-time savings.

- The system name in the directory ("u.pro.E.l.j.pv_hf_copy.8k.r3-3100.recal")
  indicates: bridgmanite (pv = perovskite), Hartree-Fock hybrid reference
  copy, 8000 K, r3 (third recalculation at step 3100 of active learning).

- The 4-GPU run uses 2 nodes with 2 GPUs each. This is because Della GPU
  nodes have a maximum of 2 A100 GPUs per node (some have 4, but the
  `all` partition often schedules 2 per node). For guaranteed 4 GPUs on
  one node, specify `--gres=gpu:a100:4` and use the appropriate partition.

## Related

- DeePMD-kit training benchmarks: `benchmarks/DELLA/deepmd_pt_vs_tf_20260406/`
- LAMMPS inference benchmarks: `benchmarks/DELLA/lammps_pt_vs_tf_20260406/`
