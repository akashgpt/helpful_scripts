# VASP 6.6.0 GPU Compilation on NCSA DELTA

## Working Build: HPC-X OpenMPI

This is the build used in all successful benchmarks.

- **Makefile:** `makefile.include__NCSA_DELTA_GPU`
- **Build dir:** `/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0.gpu/build_delta_gpu_rh9_nvhpc_hpcx/`
- **Binary:** `vasp_std__NCSA_DELTA_GPU` (also `vasp_gam`, `vasp_ncl`)
- **Built:** 2026-04-04

### Modules

```bash
module reset
module load nvhpc-hpcx-cuda12/25.3 intel-oneapi-mkl/2024.2.2
```

### Build command

```bash
cp makefile.include__NCSA_DELTA_GPU makefile.include
make DEPS=1 -j 8 all
# Rename binaries to avoid overwriting
for v in std gam ncl; do
    cp bin/vasp_${v} bin/vasp_${v}__NCSA_DELTA_GPU
done
```

---

## Broken Build: Cray MPICH (PrgEnv-nvidia)

Built to test if Cray MPICH would fix multi-GPU scaling. **Crashes on 2+ GPUs** with
`CUDA_ERROR_ILLEGAL_ADDRESS` in `m_sumb_d` (GPU-aware MPI + NCCL conflict).

- **Makefile:** `makefile.include__NCSA_DELTA_GPU_craympich`
- **Build dir:** `/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0.gpu/build_delta_gpu_rh9_prgenv_nvidia_craympich/`
- **Binary:** `vasp_std__NCSA_DELTA_GPU_craympich`
- **Built:** 2026-04-05

### Modules

```bash
module purge
module load PrgEnv-nvidia/8.6.0 craype-x86-milan cudatoolkit/25.3_12.8 aws-ofi-nccl
```

### Status

- 1 GPU: works (same performance as HPC-X build)
- 2+ GPUs: crashes with illegal GPU memory access in VASP MPI layer
- Likely cause: conflict between NCCL (`-DUSENCCL`) and Cray's GPU Transport Layer (GTL)

---

## Untested Build: HPC-X Portable

A "portable" build with different optimization flags. Not benchmarked.

- **Makefile:** `makefile.include__NCSA_DELTA_GPU_portable`
- **Binary:** `vasp_std__NCSA_DELTA_GPU_portable`
- **Built:** 2026-04-04

---

## Summary

| Build | Binary suffix | 1 GPU | Multi-GPU | Recommended |
|---|---|---|---|---|
| HPC-X OpenMPI | `__NCSA_DELTA_GPU` | Working | Broken scaling (binding issue, not build issue) | **Yes** |
| Cray MPICH | `__NCSA_DELTA_GPU_craympich` | Working | Crashes | No |
| HPC-X Portable | `__NCSA_DELTA_GPU_portable` | Untested | Untested | No |
