# VASP Benchmark Summary

## Benchmark Location

- Source benchmark directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/test/TEST__u.pro.E.l.j.pv_hf_copy.8k.r3-3100.recal`
- Analysis date: `2026-04-04`
- Scope note: folders `2`, `3`, `deepmd`, and `*111_to_222*` are intentionally ignored

## What Was Benchmarked

These runs benchmark one static VASP SCF calculation rather than MD throughput.
The completed benchmark cases use the same 160-atom Mg-Si-O perovskite
snapshot:

- Composition: `Mg32 Si32 O96`
- Cell: cubic, `14.88805580139160 A`
- `ENCUT = 800 eV`
- `ALGO = N`
- `NBANDS = 768`
- `ISTART = 0`
- `ICHARG = 2`
- `NSW = 0`

Because `NSW = 0`, the timings are dominated by the electronic SCF work and the
parallelization strategy rather than ionic motion.

## Completed Benchmark Results

| Case | Key settings | Elapsed time (s) | Speedup vs CPU 2x2x2 base |
| --- | --- | ---: | ---: |
| `1` | CPU VASP 6.4.3, 96 cores, `2x2x2`, `NPAR=12` | 2130.112 | 1.00 |
| `cpu_6.6.0_96cores` | CPU VASP 6.6.0, 96 cores, `2x2x2`, `NPAR=12` | 2182.418 | 0.98 |
| `1_KPAR` | CPU VASP 6.4.3, 96 cores, `2x2x2`, `KPAR=4` | 1991.448 | 1.07 |
| `1_GPU` | 1 GPU, default `NSIM`, `2x2x2` | 1633.995 | 1.30 |
| `1gpu_nsim16` | 1 GPU, `NSIM=16`, `2x2x2` | 1551.378 | 1.37 |
| `1gpu_nsim32` | 1 GPU, `NSIM=32`, `2x2x2` | 1515.956 | 1.41 |
| `1gpu_nsim64` | 1 GPU, `NSIM=64`, `2x2x2` | 1558.601 | 1.37 |
| `2gpu_kpar2_nsim32` | 2 GPU, `KPAR=2`, `NSIM=32`, `2x2x2` | 797.933 | 2.67 |
| `4gpu_kpar4_nsim32` | 4 GPU, `KPAR=4`, `NSIM=32`, `2x2x2` | 435.392 | 4.89 |
| `1__KPOINTS_111` | CPU VASP 6.4.3, 96 cores, `1x1x1` | 543.216 | 3.92 |

## Main Inferences

### 1. K-point mesh is the strongest runtime lever

Switching from `2x2x2` to `1x1x1` reduced the elapsed time from `2130.112 s`
to `543.216 s`, about `3.9x` faster. The two runs used the same system size and
the same number of SCF iterations (`35`), so the cost difference is mainly from
the number of irreducible k-points:

- `2x2x2` completed with `NKPTS = 4`
- `1x1x1` completed with `NKPTS = 1`

For this single configuration, the final energies were:

- `2x2x2`: `-1089.85247589 eV`
- `1x1x1`: `-1089.85671189 eV`

This is a difference of about `4.24 meV` per 160-atom cell. That is small in
energy, but a production decision should still be checked using forces and
stress on multiple snapshots before replacing `2x2x2` with `1x1x1`.

### 2. CPU-side tuning gives only modest improvement here

Adding `KPAR=4` on the CPU run improved the `2x2x2` wall time from `2130.112 s`
to `1991.448 s`, about `7%` faster.

Changing from VASP `6.4.3` to `6.6.0` on the same 96-core CPU layout did not
improve performance in this benchmark:

- VASP `6.4.3`: `2130.112 s`
- VASP `6.6.0`: `2182.418 s`

### 3. GPU scaling is the best path for the `2x2x2` workload

For the GPU runs, `NPAR=1` was used and `NSIM` was swept. The best 1-GPU result
was `NSIM=32`:

- Default 1 GPU: `1633.995 s`
- 1 GPU, `NSIM=16`: `1551.378 s`
- 1 GPU, `NSIM=32`: `1515.956 s`
- 1 GPU, `NSIM=64`: `1558.601 s`

So `NSIM=32` looks best for this system size and workload, while `NSIM=64`
starts to lose a little efficiency again.

Scaling across GPUs was strong when paired with k-point parallelization:

- 1 GPU, `NSIM=32`: `1515.956 s`
- 2 GPU, `KPAR=2`, `NSIM=32`: `797.933 s`
- 4 GPU, `KPAR=4`, `NSIM=32`: `435.392 s`

This gives:

- `1.90x` speedup from 1 GPU to 2 GPU
- `3.48x` speedup from 1 GPU to 4 GPU

That scaling is physically reasonable because the `2x2x2` mesh has `4`
irreducible k-points, and `KPAR=4` maps very naturally onto 4 GPU groups.

### 4. The completed `2x2x2` CPU and GPU runs reached the same final energy

The final energies of the completed `2x2x2` cases agree within about `1e-6 eV`,
which indicates that the faster GPU and `KPAR` settings did not change the
answer in any meaningful way for this benchmark.

Representative final energies:

- CPU base: `-1089.85247589 eV`
- CPU `KPAR=4`: `-1089.85247589 eV`
- 1 GPU default: `-1089.85247502 eV`
- 1 GPU `NSIM=32`: `-1089.85247484 eV`
- 2 GPU `KPAR=2`, `NSIM=32`: `-1089.85247484 eV`
- 4 GPU `KPAR=4`, `NSIM=32`: `-1089.85247484 eV`

## Practical Recommendation

For this exact static `2x2x2` workload, the best completed setting is:

- `4 GPUs`
- `KPAR=4`
- `NSIM=32`

For lower-cost screening, `1x1x1` appears much cheaper and may be acceptable for
some tasks, but it should be validated with force and stress comparisons over
multiple representative snapshots before adoption in production workflows.
