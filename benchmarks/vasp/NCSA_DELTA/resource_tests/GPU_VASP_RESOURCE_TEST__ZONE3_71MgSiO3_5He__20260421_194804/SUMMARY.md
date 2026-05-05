# GPU VASP Resource Smoke Test: ZONE 3 `71MgSiO3_5He`

**Date:** 2026-04-21  
**Cluster:** NCSA DELTA  
**Observed GPU in logs:** NVIDIA A100-SXM4-40GB (`40960 MiB`)  
**Source benchmark directory:** `/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/setup_MLMD/benchmarking_tests/GPU_VASP_RESOURCE_TEST__ZONE3_71MgSiO3_5He__20260421_194804`

## Purpose

This benchmark is a **resource smoke test** for one production-like MLMD frame before
launching long VASP trajectories. The goal is not electronic convergence quality by itself,
but to learn which GPU/rank layouts are even viable for this large frame under the tested
DELTA launch stack.

The test uses a Zone 3 frame from:

- `/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/v1_i1/md/ZONE_3/71MgSiO3_5He/pre/recal/1`

## System and Common Settings

- Composition: `71 MgSiO3 + 5 He = 360 atoms`
- K-points: `4` irreducible k-points
- `NBANDS = 1536`
- `ENCUT = 800 eV`
- `METAGGA = R2SCAN`
- `NSIM = 32`
- `NPAR = 1`
- `NELM = 8`
- Launch style: `mpirun --bind-to none` with `nvhpc-hpcx-cuda12/25.3`

Because `NELM = 8`, these jobs are **not production convergence runs**. They are short,
bounded tests meant to expose memory pressure and launch-path failures early.

## Results

The machine-readable table is in [benchmark_results.tsv](./benchmark_results.tsv).

| Test | GPUs | Ranks | CPUs/task | KPAR | ALGO | Status | Elapsed (s) | Max GPU mem (MiB) | Main outcome |
| ---- | ---- | ----- | --------- | ---- | ---- | ------ | ----------- | ----------------- | ------------ |
| `g01_r01_c01_kpar1_algoN` | 1 | 1 | 1 | 1 | `N` | failed | NA | 36053 | CUDA OOM |
| `g01_r01_c08_kpar1_algoN` | 1 | 1 | 8 | 1 | `N` | failed | NA | 36369 | CUDA OOM |
| `g02_r02_c01_kpar1_algoN` | 2 | 2 | 1 | 1 | `N` | done | 609.102 | 34687 | stable smoke-test completion |
| `g02_r02_c01_kpar2_algoExact` | 2 | 2 | 1 | 2 | `Exact` | failed | NA | 34997 | wavefunction allocation failure |
| `g02_r02_c01_kpar2_algoN` | 2 | 2 | 1 | 2 | `N` | failed | NA | 34997 | CUDA OOM |
| `g02_r02_c08_kpar2_algoN` | 2 | 2 | 8 | 2 | `N` | failed | NA | 35283 | CUDA OOM |
| `g04_r04_c01_kpar1_algoN` | 4 | 4 | 1 | 1 | `N` | done | 319.817 | 20059 | stable smoke-test completion |
| `g04_r04_c01_kpar4_algoN` | 4 | 4 | 1 | 4 | `N` | failed | NA | 38581 | CUDA OOM |

## Key Findings

### 1. Single-GPU runs are not viable on the observed 40 GB A100s

Both 1-GPU tests failed with CUDA out-of-memory even though CPU allocation changed from
`1` to `8` cores per task. The peak sampled memory was already high in both cases:

- `g01_r01_c01_kpar1_algoN`: `36053 MiB`
- `g01_r01_c08_kpar1_algoN`: `36369 MiB`

Scientific meaning: for this `360`-atom, `1536`-band R2SCAN frame, the bottleneck is GPU
memory capacity, not host CPU count.

### 2. `KPAR = 1` with multiple GPUs is stable

Two layouts completed the 8-step smoke test:

- `2 GPU, 2 ranks, KPAR=1`: `609.102 s`
- `4 GPU, 4 ranks, KPAR=1`: `319.817 s`

This is a `1.90x` runtime reduction from `2` to `4` GPUs for the same bounded test.
The final total energies also agree to within about `7e-8 eV`:

- `g02_r02_c01_kpar1_algoN`: `-2830.53717305 eV`
- `g04_r04_c01_kpar1_algoN`: `-2830.53717312 eV`

That is a strong sign that the stable paths are numerically consistent, even though the
smoke test stops at `NELM=8` and prints the expected non-convergence warning afterward.

### 3. `KPAR > 1` is not usable in this tested setup

Every `KPAR > 1` case failed:

- `g02_r02_c01_kpar2_algoN`: CUDA OOM
- `g02_r02_c08_kpar2_algoN`: CUDA OOM
- `g04_r04_c01_kpar4_algoN`: CUDA OOM
- `g02_r02_c01_kpar2_algoExact`: failed with
  `Could not allocate wavefunction on mpi rank 0 of size: 4993793 MB`

Changing `ALGO` from `N` to `Exact` did **not** rescue the 2-GPU `KPAR=2` case. Instead,
it failed earlier with an extreme wavefunction allocation request, which is a different
symptom but still points to an unusable configuration for this frame.

### 4. CPU count per task does not fix the memory problem

The two 2-GPU `KPAR=2` cases with `1` and `8` CPUs per task both failed with OOM, and the
two 1-GPU cases also both failed. In simple terms, extra CPU threads did not buy enough
GPU-memory relief to change the outcome.

### 5. `hcoll` warnings are present but not the main failure signal here

Several multi-rank logs include OpenMPI `hcoll` initialization warnings, but the
`2 GPU, KPAR=1` and `4 GPU, KPAR=1` jobs still completed. For this benchmark, the useful
separation is:

- non-fatal `hcoll` warnings in some successful jobs
- fatal GPU-memory or wavefunction-allocation failures in the unstable jobs

## Recommendation

For this exact `ZONE_3/71MgSiO3_5He` frame on the observed DELTA A100-40GB nodes:

- Do **not** use `1 GPU`
- Prefer `KPAR = 1`
- Use at least `2 GPUs`
- Prefer `4 GPUs` when available, since the smoke-test runtime drops from `609.1 s` to `319.8 s`
- In batched production wrappers, make the OpenMPI host slot count match the per-frame
  rank/GPU count. For the `2 GPU, 2 rank` case, use a host form like
  `--host "${target_node}:2"` rather than `--host "$target_node"`, because OpenMPI may
  otherwise assume only one slot is available on that host.

If future work targets a different GPU memory class, especially 80 GB devices, the
single-GPU and `KPAR > 1` cases should be re-benchmarked rather than assumed.

## Included Files

This benchmark mirror intentionally keeps the files that explain or reproduce the result:

- `benchmark_config.json`
- `submit_all.sh`
- `summarize_results.sh`
- `benchmark_results.tsv`
- per-run `INCAR`, `KPOINTS`, `POSCAR`, `sub_vasp_gpu.sh`
- per-run `log.run_sim`, `gpu_memory_trace.csv`, `slurm-*.out`, `slurm-*.err`
- per-run `OUTCAR`

Heavy restart and wavefunction artifacts such as `WAVECAR`, `CHG`, and `vasprun.xml` were
left out on purpose so this benchmark folder stays focused on resource behavior.
