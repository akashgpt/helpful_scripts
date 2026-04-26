# VASP 6.6.0 Multi-GPU Scaling Benchmark: NCSA DELTA vs PU Stellar

**Date:** 2026-04-06
**Author:** Akash Gupta (with Claude Code analysis assist)
**Test system:** MgSiO Pv, 160 atoms, 768 bands, 4 k-points (2x2x2 Monkhorst-Pack), R2SCAN meta-GGA, ENCUT=800 eV
**Source data:** `/work/nvme/bguf/akashgpt/qmd_data/test/TEST__u.pro.E.l.j.pv_hf_copy.8k.r3-3100.recal/`

---

## 1. Executive Summary

Single-GPU performance on NCSA DELTA (A100-SXM4 80 GB) is **~8% faster** than PU Stellar (A100-SXM4 40 GB).
However, **multi-GPU (KPAR) scaling on DELTA is completely broken** -- 4 GPUs provide essentially zero
wall-clock speedup over 1 GPU. Stellar achieves near-linear scaling (3.8x with 4 GPUs). The root cause
is **GPU-rank binding failure** in the MPI launch configuration on DELTA, not a hardware or compilation issue.

---

## 2. Hardware Comparison

| Property          | PU Stellar                         | NCSA DELTA                          |
| ----------------- | ---------------------------------- | ----------------------------------- |
| GPU model         | NVIDIA A100-SXM4 40 GB             | NVIDIA A100-SXM4 80 GB             |
| GPUs per node     | 2                                  | 4 (gpuA100x4 partition)            |
| CPU               | AMD EPYC 7H12 (2x 64-core)        | AMD EPYC 7763 (2x 64-core Milan)   |
| Interconnect      | InfiniBand HDR100                  | HPE Slingshot 11                    |
| Scheduler         | Slurm + OpenMPI                    | Slurm + Cray MPICH / HPC-X         |

---

## 3. Timing Results

### 3.1 GPU Runs -- Wall-Clock (Elapsed) Time

| Config               | Stellar (s) | DELTA (s)  | Stellar speedup | DELTA speedup |
| -------------------- | ----------- | ---------- | --------------- | ------------- |
| 1 GPU, NSIM=16       | 1520        | 1428       | 1.0x            | 1.0x          |
| 1 GPU, NSIM=32       | 1516        | 1405       | 1.0x            | 1.0x          |
| 2 GPU, KPAR=2        | 763         | 1311-1405  | **2.0x**        | **1.0-1.1x**  |
| 4 GPU, KPAR=4        | 397         | 1354       | **3.8x**        | **1.05x**     |

### 3.2 CPU Time vs Real Time (the Smoking Gun)

The ratio of CPU time to real (wall-clock) time reveals the problem. Healthy GPU runs have
ratio ~1.0 (CPU busy while GPU computes). Ratio << 1.0 means ranks are idle/waiting.

| Run                  | CPU time (s) | Real time (s) | Ratio | Status       |
| -------------------- | ------------ | ------------- | ----- | ------------ |
| DELTA 1 GPU          | 1416         | 1402          | 1.01  | Healthy      |
| DELTA 2 GPU          | 843          | 1405          | 0.60  | **Broken**   |
| DELTA 4 GPU          | 379          | 1354          | 0.28  | **Broken**   |
| Stellar 1 GPU        | 1530         | 1520          | 1.01  | Healthy      |
| Stellar 2 GPU        | 822          | 763           | 1.08  | Healthy      |
| Stellar 4 GPU        | 416          | 397           | 1.05  | Healthy      |

Key observation: On DELTA, the CPU time scales correctly with KPAR (the work is split), but
the wall time stays constant -- meaning the MPI ranks are being **serialized on a single GPU**
rather than running in parallel on separate GPUs.

### 3.3 CPU vs GPU Comparison (NCSA DELTA)

| Config                                    | Resources         | Elapsed (s) | Speedup vs 128-core CPU |
| ----------------------------------------- | ----------------- | ----------- | ----------------------- |
| CPU 64 cores (8 MPI x 8 OMP)             | 64 cores, 1 node  | >3600 (DNF) | <0.63x                 |
| CPU 96 cores (12 MPI x 8 OMP)            | 96 cores, 1 node  | 2182        | 1.03x                  |
| CPU 128 cores (16 MPI x 8 OMP)           | 128 cores, 1 node | 2256        | 1.0x (baseline)        |
| CPU 256 cores (32 MPI x 8 OMP)           | 256 cores, 2 nodes| 1513        | 1.49x                  |
| **1 GPU** (A100-SXM4 80 GB)              | 16 cores + 1 GPU  | **1405**    | **1.61x**              |

Key observations:
- **1 GPU beats 256 CPU cores across 2 nodes** (1405 s vs 1513 s) using far fewer resources.
- CPU scaling saturates early: 96-core and 128-core are essentially identical (~2200 s).
  Doubling to 256 cores (2 nodes) only gives 1.49x -- poor CPU scaling for this system.
- The 64-core run was killed at the 1-hour time limit before completing.
- A single GPU delivers 1.6x speedup over a full 128-core DELTA node.

### 3.4 CPU Count per GPU -- Does It Matter? (NCSA DELTA)

Tested 1, 2, 4, and 16 CPUs per GPU to determine optimal allocation.
System: 160 atoms, 768 bands, 4 k-points, 1 GPU.

| CPUs per GPU | OMP Threads | Elapsed (s) |
| ------------ | ----------- | ----------- |
| 1            | 1           | 1409        |
| 2            | 2           | 1400        |
| 4            | 4           | 1530        |
| 16           | 16          | 1405        |

**CPU count has negligible impact on single-GPU performance.** 1 CPU performs
identically to 16 CPUs. This means multi-GPU packing (MULTI_sub_vasp_GPU.sh)
can use `--cpus-per-task=1`, minimizing resource requests and improving
scheduling priority.

### 3.5 Multi-GPU Packing (Independent VASP Runs, 1 GPU Each)

Tested packing multiple independent VASP calculations onto GPUs using
`MULTI_sub_vasp_GPU.sh` with GNU parallel + per-GPU CUDA_VISIBLE_DEVICES binding.
Each run: 1 MPI rank, 1 OMP thread, 1 GPU, KPAR=1.

**Test 1: 1 node x 4 GPUs, 8 runs (4 concurrent + 4 queued)**

| Run    | Node     | GPU | Elapsed (s) |
| ------ | -------- | --- | ----------- |
| run_01 | gpua097  | 0   | 1418        |
| run_02 | gpua097  | 1   | 1409        |
| run_03 | gpua097  | 2   | 1412        |
| run_04 | gpua097  | 3   | 1416        |
| run_05 | gpua097  | 0   | 1407        |
| run_06 | gpua097  | 1   | 1460        |
| run_07 | gpua097  | 2   | 1411        |
| run_08 | gpua097  | 3   | 1447        |

**Test 2: 2 nodes x 4 GPUs, 8 runs (all concurrent)**

| Run    | Node     | GPU | Elapsed (s) |
| ------ | -------- | --- | ----------- |
| run_01 | gpua053  | 0   | 1412        |
| run_02 | gpua053  | 1   | 1415        |
| run_03 | gpua053  | 2   | 1427        |
| run_04 | gpua053  | 3   | 1409        |
| run_05 | gpua084  | 0   | 1409        |
| run_06 | gpua084  | 1   | 1408        |
| run_07 | gpua084  | 2   | 1408        |
| run_08 | gpua084  | 3   | 1408        |

Key observations:
- **No contention** between concurrent GPU runs on the same node (~1410 s each,
  matching the single-GPU baseline of ~1405 s).
- **Multi-node distribution works correctly** via `mpirun --host $target_node`.
- **1 CPU per GPU is sufficient** -- total node usage is just 4 CPUs + 4 GPUs.
- Memory usage: ~13 GB per run (~52 GB total for 4 concurrent runs on 1 node).
- GNU parallel correctly queues overflow runs (test 1: runs 5-8 waited for GPUs 0-3
  to free up, then ran on the same GPUs).

### 3.6 When GPU Is Slower Than CPU

Tested on NH3/H2 system: 220 atoms, 560 bands, **1 k-point (gamma only)**.

| Config                           | Elapsed (s) |
| -------------------------------- | ----------- |
| CPU 128 cores (16 MPI x 8 OMP)  | ~300        |
| **1 GPU** (A100-SXM4 80 GB)     | **~650**    |

GPU is **2x slower** for this system because:
- Only 1 k-point → no KPAR parallelism across GPUs.
- 560 bands → matrices too small to saturate the A100.
- CPU-GPU data transfer overhead dominates the compute savings.

**Rule of thumb:** GPU VASP benefits from larger systems (more bands) and/or
multiple k-points. For small gamma-point calculations, CPU is faster.

### 3.7 CPU Baselines by Cluster (for Reference)

| Config                          | Elapsed (s) | Cluster |
| ------------------------------- | ----------- | ------- |
| 96 cores (12 MPI x 8 OMP)      | 2182        | DELTA   |
| 128 cores (16 MPI x 8 OMP)     | 2256        | DELTA   |
| 256 cores / 2 nodes (DELTA)    | 1513        | DELTA   |
| 96 cores (12 MPI x 8 OMP)      | 2130        | Stellar |

---

## 4. Root Cause Analysis: GPU-Rank Binding Failure on DELTA

### 4.1 The Problem

On DELTA, all MPI ranks end up sharing the same GPU (GPU 0) instead of each being assigned
its own distinct GPU. This is evidenced by:

1. The `mpirun --bind-to none` command in the HPC-X runs provides **no GPU affinity**.
   All ranks see all GPUs via `CUDA_VISIBLE_DEVICES` but VASP defaults to GPU 0.

2. The rank-binding wrapper (`rankbindfix2`) attempted to set `CUDA_VISIBLE_DEVICES` per
   rank using `OMPI_COMM_WORLD_LOCAL_RANK`, but the inherited `CUDA_VISIBLE_DEVICES=0,1`
   was already pre-filtered by Slurm -- causing rank 0 -> GPU 0, rank 1 -> GPU 1 **in the
   remapped namespace** (which may still map to the same physical GPU depending on how
   Slurm cgroups are configured).

3. The cray-mpich/srun attempt with `--gpu-bind=closest` showed
   `SLURM_STEP_GPUS=0` for **all ranks** -- Slurm's step-level GPU allocation was restricting
   all ranks to a single GPU.

### 4.2 Why Stellar Works

Stellar uses a clean **OpenMPI + srun** stack:

```bash
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
srun vasp_std
```

With `--gres=gpu:N` and `--ntasks-per-node=N`, Slurm+OpenMPI automatically maps each rank
to its own GPU. No wrapper or explicit GPU binding is needed.

### 4.3 Why DELTA Is Different

DELTA uses HPE Slingshot + Cray MPICH (or the HPC-X compatibility layer), which has a
different GPU binding model:

- Cray MPICH does not automatically set `CUDA_VISIBLE_DEVICES` per rank.
- HPC-X `mpirun` with `--bind-to none` leaves GPU selection to the application.
- VASP (6.6.0) relies on the MPI layer or Slurm to handle GPU affinity -- it does not
  set `CUDA_VISIBLE_DEVICES` internally.
- Slurm on DELTA may use GPU cgroups differently, so `CUDA_VISIBLE_DEVICES` numbering
  inside the job step may not match physical GPU IDs.

---

## 5. What Was Tried (and What Failed)

### 5.1 Attempt 1: Plain `mpirun --bind-to none` (HPC-X, 2026-04-04)

- **Runs:** `gpu_6.6.0_{1,2,4}gpu_*__20260404__190644`
- **Result:** 1 GPU works fine. 2 GPU and 4 GPU "complete" but with **no scaling** (wall time ~1350-1405s for all).
- **Diagnosis:** No GPU affinity set. All ranks share GPU 0.

### 5.2 Attempt 2: `srun` with HPC-X binary (2026-04-05)

- **Runs:** `*srunfix__20260405__102950` (in `reference__NCSA_DELTA__failed_runs/`)
- **Result:** 1-GPU run was `CANCELLED DUE to SIGNAL Terminated`. 2-GPU run also killed by srun after DAV step 5.
- **Diagnosis:** The HPC-X-linked VASP binary is not compatible with `srun` launch (which uses PMI2/Cray's PMI). The binary was compiled against HPC-X's OpenMPI and expects `mpirun`.

### 5.3 Attempt 3: `mpirun` with rank-binding wrapper (`rankbind`, 2026-04-05)

- **Runs:** `*rankbind__20260405__103419` (in `reference__NCSA_DELTA__failed_runs/`)
- **Result:** 2-GPU run appeared to start but log truncated -- likely timed out without producing OUTCAR.
- **Diagnosis:** Wrapper set `CUDA_VISIBLE_DEVICES` per rank, but the assignment may have been incorrect (both ranks could have ended up on the same physical GPU due to Slurm cgroup GPU remapping).

### 5.4 Attempt 4: Improved rank-binding wrapper (`rankbindfix2`, 2026-04-05)

- **Runs:** `*rankbindfix2__20260405__103612` and `*rankbindfix2__purge__20260405__110522`
- **Result:** 2-GPU completed in 1311-1319s. Slight improvement over 1405s but still **far** from the expected ~700s.
- **Diagnosis:** The wrapper log shows:
  ```
  Rank 0: inherited_CUDA_VISIBLE_DEVICES=0,1 final=0
  Rank 1: inherited_CUDA_VISIBLE_DEVICES=0,1 final=1
  ```
  This *looks* correct, but `SLURM_JOB_GPUS=0,2` (physical GPU IDs 0 and 2) while `CUDA_VISIBLE_DEVICES=0,1`
  (Slurm-remapped IDs). The wrapper correctly picks device 0 and 1 in the remapped space, but the performance
  suggests the ranks are still contending -- possibly because the HPC-X mpirun internally overrides
  `CUDA_VISIBLE_DEVICES` after the wrapper sets it, or because GPU-GPU communication (NCCL/NVLink) is not
  working correctly with this binding approach.

### 5.5 Attempt 5: Cray MPICH (PrgEnv-nvidia) + srun with `--gpu-bind=closest` (2026-04-05)

- **Runs:** `*craympich_exclusive__20260405__160441` (in `reference__NCSA_DELTA__failed_runs/`)
- **Result:** **PMI2 initialization failure:**
  ```
  Fatal error in PMPI_Init_thread: MPIR_pmi_init(115): PMI2_Job_GetId returned 14
  ```
- **Diagnosis:** The cray-mpich binary was launched without `--mpi=pmi2` in the srun flags, or the PMI
  environment was not properly set up. The `--exclusive` sbatch flag also changed the cgroup GPU assignment
  which may have interfered.

### 5.6 Attempt 6: Cray MPICH + srun + `--mpi=pmi2` + GTL (2026-04-05)

- **Runs:** `*craympich_pmi2_gtl__20260405__165015`
- **Result for 1 GPU:** Success! Elapsed 1405s (matches HPC-X 1-GPU performance).
- **Result for 2 GPU:** **CUDA illegal address error** after DAV step 4:
  ```
  Accelerator Fatal Error: call to cuStreamSynchronize returned error 700
  (CUDA_ERROR_ILLEGAL_ADDRESS): Illegal address during kernel execution
  File: mpi.f90  Function: m_sumb_d:755  Line: 1864
  ```
- **Diagnosis:** The Cray MPICH build with `MPICH_GPU_SUPPORT_ENABLED=1` enables GPU-aware MPI
  (direct GPU-to-GPU transfers via GTL). This triggers a code path in VASP's MPI communication
  (`m_sumb_d`) that performs GPU-direct operations. The illegal address error suggests either:
  - The VASP binary was not correctly linked against Cray's GPU Transport Layer (GTL) libraries.
  - The GPU memory pointers passed to MPI are not registered with the GPU-aware MPI layer.
  - A bug in the interplay between NCCL (which VASP uses internally) and Cray's GTL.

### 5.7 Attempt 7: Cray MPICH + 2 GPU with rank wrapper (2026-04-06, most recent)

- **Runs:** `*craympich_pmi2_gtl_rankfix__20260405__171207`
- **Result:** Timed out (30 min limit). Log shows `SLURM_STEP_GPUS=0` for **both ranks**.
- **Diagnosis:** Even with `--gpus-per-task=1 --gpu-bind=closest`, srun assigned GPU 0 to the step
  for both ranks. This confirms that the Slurm+Cray MPICH GPU assignment on DELTA is not behaving
  as expected for multi-GPU-per-node configurations.

---

## 6. Recommended Next Steps

### 6.1 Priority: Fix GPU Binding with HPC-X `mpirun` (Most Likely to Work)

Since the HPC-X binary already works for 1 GPU and the computation itself is correct (energies match),
the path of least resistance is to fix the GPU binding for the HPC-X launch:

```bash
# Option A: Use --map-by with GPU binding in OpenMPI/HPC-X
mpirun --bind-to none --map-by ppr:1:numa:PE=16 \
       -x CUDA_VISIBLE_DEVICES \
       --mca mpi_cuda_support 0 \
       -np 4 \
       bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec vasp_std'

# Option B: Explicit per-rank CUDA_VISIBLE_DEVICES via wrapper
# (This is what rankbindfix2 tried -- but ensure the wrapper is invoked
#  AFTER mpirun sets OMPI_COMM_WORLD_LOCAL_RANK and BEFORE VASP initializes CUDA)
```

### 6.2 Alternative: Build VASP Against Native OpenMPI (Not Cray MPICH)

Load a standalone OpenMPI module (not HPC-X, not Cray MPICH) and rebuild VASP:

```bash
module purge
module load nvhpc/25.3
module load openmpi/4.1.6    # or whatever standalone OpenMPI is available
module load intel-oneapi-mkl/2024.2.2
```

This would make the launch semantics identical to Stellar, where `srun vasp_std` just works.

### 6.3 Alternative: Debug Cray MPICH GTL Build

If GPU-aware MPI is desired for future large-scale runs:

1. Rebuild VASP with `-DUSENCCL` **disabled** to avoid NCCL/GTL conflicts.
2. Ensure `MPICH_GPU_SUPPORT_ENABLED=1` and `MPICH_OFI_NIC_POLICY=GPU` are set.
3. Use `srun --mpi=pmi2 --gpu-bind=map_gpu:0,1,2,3` explicitly.
4. Test with a simple GPU-aware MPI hello-world program first to confirm GTL is working.

### 6.4 Diagnostic: Verify GPU Assignment at Runtime

Add this to the submission script before launching VASP to confirm each rank sees a different GPU:

```bash
# Quick GPU visibility check
mpirun --bind-to none -np ${MPI_RANKS} bash -c '
    echo "Rank $OMPI_COMM_WORLD_RANK: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    nvidia-smi -L 2>/dev/null | head -1
'
```

---

## 7. Reference: Submission Scripts

### 7.1 Stellar -- Working Multi-GPU Script (4 GPU)

```bash
#!/bin/bash
#SBATCH --job-name=qmd_4gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --mem=128G
#SBATCH --time=1:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module purge
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
module load intel-mkl/2024.2

srun vasp_std
```

### 7.2 Stellar -- Working 2-GPU Script

```bash
#!/bin/bash
#SBATCH --job-name=qmd_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --mem=128G
#SBATCH --time=1:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module purge
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
module load intel-mkl/2024.2

srun vasp_std
```

### 7.3 Stellar -- Working 1-GPU Script

```bash
#!/bin/bash
#SBATCH --job-name=qmd_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --time=1:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module purge
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
module load intel-mkl/2024.2

srun vasp_std
```

### 7.4 DELTA -- HPC-X 1-GPU Script (Working)

```bash
#!/bin/bash
#SBATCH --job-name=qmd_gpu660_delta
#SBATCH --partition=gpuA100x4
#SBATCH --account=bguf-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch"

module reset
module load nvhpc-hpcx-cuda12/25.3 intel-oneapi-mkl/2024.2.2
ulimit -s unlimited

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

mpirun --bind-to none -np 1 vasp_std__NCSA_DELTA_GPU
```

### 7.5 DELTA -- HPC-X Multi-GPU Script (Broken Scaling)

Same as 7.4 but with `--ntasks-per-node=4 --gpus-per-node=4 -np 4` and KPAR=4 in INCAR.
This runs and produces correct energies but with no speedup.

### 7.6 DELTA -- Cray MPICH + GTL (Crashes on 2+ GPUs)

```bash
#!/bin/bash
#SBATCH --partition=gpuA100x4
#SBATCH --account=bguf-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch"

module purge
module load PrgEnv-nvidia/8.6.0 craype-x86-milan cudatoolkit/25.3_12.8 aws-ofi-nccl

export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
ulimit -s unlimited

srun --mpi=pmi2 --ntasks=2 --cpus-per-task=16 --gpus-per-task=1 \
     --gpu-bind=closest --cpu-bind=cores \
     vasp_std__NCSA_DELTA_GPU_craympich
```

Result: 1 GPU works, 2 GPU crashes with `CUDA_ERROR_ILLEGAL_ADDRESS` in `m_sumb_d`.

---

## 8. VASP Compilation Details

### 8.1 VASP Build on DELTA

Two binaries were built:

1. **HPC-X build** (`vasp_std__NCSA_DELTA_GPU`):
   - Modules: `nvhpc-hpcx-cuda12/25.3`, `intel-oneapi-mkl/2024.2.2`
   - Built: 2026-04-04
   - MPI: HPC-X OpenMPI (bundled with nvhpc-hpcx-cuda12)

2. **Cray MPICH build** (`vasp_std__NCSA_DELTA_GPU_craympich`):
   - Modules: `PrgEnv-nvidia/8.6.0`, `craype-x86-milan`, `cudatoolkit/25.3_12.8`
   - Compile flags: `-DUSENCCL`, `-DNVCUDA`, `-DACC_OFFLOAD`, GPU target `cc80,cuda12.8`
   - MPI: Cray MPICH 8.1.32

### 8.2 VASP Build on Stellar

- Modules: `nvhpc/25.5`, `openmpi/cuda-12.9/nvhpc-25.5/4.1.8`, `intel-mkl/2024.2`
- Built: 2026-03-28
- MPI: OpenMPI 4.1.8 (CUDA-aware)

### 8.3 INCAR Settings (Common to All Runs)

```
NPAR   = 1        # Must be 1 for GPU VASP
NBANDS = 768
NSIM   = 32       # Larger value saturates GPU
KPAR   = {1,2,4}  # Matches number of GPUs
ALGO   = N
LREAL  = A
ENCUT  = 800
EDIFF  = 1E-06
METAGGA = R2SCAN
LASPH  = .TRUE.
ISYM   = 0
ISMEAR = -1
SIGMA  = 0.689387
```

---

## 9. Conclusions

1. **DELTA A100-80GB hardware is faster per-GPU than Stellar A100-40GB** (~8% faster single-GPU).
2. **Multi-GPU KPAR scaling is completely broken on DELTA** due to GPU-rank binding issues in the MPI launch layer.
3. **The computation itself is correct** -- energies match across all runs and both clusters.
4. **The HPC-X `mpirun` + rank-wrapper approach got closest** to working but still showed severe contention.
5. **The Cray MPICH + GTL approach crashes** on multi-GPU due to illegal GPU memory access in VASP's MPI communication layer.
6. **The most promising path forward** is either (a) fixing the HPC-X GPU binding or (b) building VASP against a standalone OpenMPI installation on DELTA to replicate the Stellar environment.

---

## 10. Related Resource Test

A follow-up benchmark focused on a larger `ZONE_3/71MgSiO3_5He` production-like frame now
lives under:

- `resource_tests/GPU_VASP_RESOURCE_TEST__ZONE3_71MgSiO3_5He__20260421_194804/`

That benchmark is complementary to the scaling study in this document:

- this file explains why DELTA multi-GPU launch behavior can fail in general
- the `resource_tests` benchmark shows which layouts are actually usable for one large
  `360`-atom, `1536`-band R2SCAN frame under the tested DELTA setup
