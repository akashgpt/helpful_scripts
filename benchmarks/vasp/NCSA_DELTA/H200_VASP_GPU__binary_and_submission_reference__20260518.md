# NCSA Delta — H200 VASP GPU: binary compatibility & submission reference

**Date:** 2026-05-18
**Author:** Akash Gupta (with Claude Code analysis assist)
**Scope:** Which VASP GPU binary + Slurm settings to use on the `gpuH200x8`
(NVIDIA H200, Hopper/sm_90) partition on NCSA Delta, and how it relates to
the existing `gpuA100x4` (A100, Ampere/sm_80) setup.

---

## 1. Key takeaway

**One binary serves both partitions.**
`/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin/vasp_std__NCSA_DELTA_GPU`
runs correctly on **both**:

- `gpuA100x4` — A100, sm_80 (native), and
- `gpuH200x8` — H200, sm_90 (via the NVHPC OpenACC runtime's launch-time JIT).

To move a job between A100 and H200 you change **only** `#SBATCH --partition`
(`gpuA100x4` ⇄ `gpuH200x8`). Account (`bguf-delta-gpu`), modules, launch
command, and the binary path stay identical.

The `..._portable` build (`cuobjdump`: sm_80 + sm_86 + sm_90 native cubins)
also runs on H200, but it is **not required** and is **not** the validated
choice. Standardize on `vasp_std__NCSA_DELTA_GPU`.

---

## 2. The non-obvious part (read before "fixing" this)

`cuobjdump --list-elf vasp_std__NCSA_DELTA_GPU` shows **only `sm_80`** and
`--list-ptx` reports **no PTX**. A naïve reading says "Ampere-only, will fail
on Hopper" — **this is wrong**. It is an **NVHPC OpenACC** build (device code
is carried as NVHPC's own embedded IR in `pgcudafat*` objects, which the
CUDA-toolkit `cuobjdump` does not recognize as PTX). The NVHPC runtime
JIT-compiles to the live device (sm_90) at launch, so it runs on H200.

Do not swap this binary for a "portable"/multi-arch build on the assumption
that the standard one cannot run on Hopper — it can, and it is the
empirically validated one.

---

## 3. Evidence (empirically validated, not theoretical)

Successful H200 runs using `vasp_std__NCSA_DELTA_GPU` with
`#SBATCH --partition=gpuH200x8`:

| Source run | System | Outcome |
| --- | --- | --- |
| `He_MgSiO3/sim_data_ML/setup_MLMD/benchmarking_tests/ENCUT_test/72_MgSiO3__360_atoms__and_He/71MgSiO3_5He/ENCUT_0800/` | 360 atoms, ENCUT 800 | `done_RUN_VASP`, exit 0, OUTCAR complete (CPU 1682 s) |
| `.../benchmarking_tests/KSPACING_test/72_MgSiO3__360_atoms__and_He/71MgSiO3_5He/KSPACING_0p20_H200/` | 360 atoms | `done_RUN_VASP`, exit 0 |

`log.run_sim` of the ENCUT_0800 run shows `Host: gpue06.delta.ncsa.illinois.edu`;
`scontrol show node gpue06` → `Gres=gpu:h200:8`, partition `gpuH200x8`. So it
genuinely executed on H200 hardware (360 atoms — larger than typical 200-atom
production cells).

---

## 4. Validated submission settings (single-GPU)

Template: [`submission_scripts/NCSA_DELTA__submission_scripts/1gpu_h200/sub_vasp_h200.sh`](submission_scripts/NCSA_DELTA__submission_scripts/1gpu_h200/sub_vasp_h200.sh)

```
#SBATCH --account=bguf-delta-gpu     # AllowAccounts=ALL on gpuH200x8
#SBATCH --partition=gpuH200x8        # <-- only line that differs vs A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=180G                   # H200 nodes ~2 TB host RAM; 120-180 G used in benchmarks
#SBATCH --time=04:00:00              # gpuH200x8 MaxTime = 2-00:00:00

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VASP_BIN=/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin/vasp_std__NCSA_DELTA_GPU
module reset
module load nvhpc-hpcx-cuda12/25.3 intel-oneapi-mkl/2024.2.2
ulimit -s unlimited
mpirun --bind-to none -np 1 "$VASP_BIN"
```

Notes:

- **GPU memory:** H200 = 141 GB/GPU vs A100 = 40 GB on `gpuA100x4`. Large
  cells (≈200+ atoms, R2SCAN, ENCUT 800, generous NBANDS) that OOM on a
  single A100 fit comfortably on one H200 — H200 is the simplest fix for
  single-GPU GPU-OOM, no NBANDS/NSIM reduction or multi-GPU split needed.
- **Partition limits:** `gpuH200x8` — 8 nodes × 8 H200, `AllowAccounts=ALL`,
  `MaxTime=2-00:00:00`, `MaxNodes=1`.
- `bguf-delta-gpu` is valid on `gpuH200x8` (no separate H200 account).

---

## 5. Related references

- Compilation of the portable build: [`compilation/makefile.include__NCSA_DELTA_GPU_portable`](compilation/makefile.include__NCSA_DELTA_GPU_portable) and [`compilation/README__compilation.md`](compilation/README__compilation.md).
- A100 multi-GPU scaling caveats: [`README__VASP_GPU_SCALING_BENCHMARK.md`](README__VASP_GPU_SCALING_BENCHMARK.md) (KPAR multi-GPU is broken on Delta A100; single-GPU is healthy — same single-GPU guidance applies on H200).
- Project submission scripts that use this binary on `gpuH200x8`:
  `He_MgSiO3/sim_data_ML/setup_MLMD/submission_scripts/{RUN_VASP_gpu.sh,RUN_VASP_gpu_xtra.sh,MULTI_RUN_VASP_gpu.sh}`.
