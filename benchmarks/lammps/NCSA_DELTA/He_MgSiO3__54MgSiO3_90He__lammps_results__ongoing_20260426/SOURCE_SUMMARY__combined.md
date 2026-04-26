# LAMMPS backend/KOKKOS scaling benchmark — He/MgSiO3 (Mg/Si/O/He, MEDIUM_SIZE base = 54 MgSiO3 + 90 He)

Hardware: NCSA Delta, 1× A100 GPU (`gpuA100x4`), 1 MPI rank.
Timestep: 0.5 fs. NPT ensemble. Pair style: `deepmd` (compressed model, `se_e2_a` descriptor).
Identical `in.lammps.bench` for all runs (no plumed) — apples-to-apples timings.

## Tile sizes

The 360-atom base cell (14.01 × 14.01 × 13.28 Å, 4 atom types) was tiled with
`shared/tile_conf.py`:

| File              | nx,ny,nz | N atoms | Box (Å)               |
|-------------------|----------|--------:|-----------------------|
| `conf.lmp`        | 1×1×1    |    360  | 14.01 × 14.01 × 13.28 |
| `conf_2880.lmp`   | 2×2×2    |   2880  | 28.02 × 28.02 × 26.56 |
| `conf_28800.lmp`  | 4×4×5    |  28800  | 56.04 × 56.04 × 66.40 |

Step counts were scaled (5000/500/100 for production, 200/100/20 for relax) so each
production loop runs ~30 s–2 min; long enough to amortize startup/JIT warmup but
short enough to fit in 30-min SLURM windows.

## Production-loop results (the ms/step number is the headline metric)

| N atoms | Backend       | Binary                  | Loop time (s) | ms/step | timesteps/s | katom-step/s |
|--------:|---------------|-------------------------|--------------:|--------:|------------:|-------------:|
|     360 | TF            | `lmp_plmd_ncsa_delta`   |        40.80  |   8.16  |     122.6   |       44.1   |
|     360 | PT            | `lmp`                   |       182.97  |  36.59  |      27.3   |        9.8   |
|     360 | PT + KOKKOS   | `lmp_kk`                |       225.35  |  45.07  |      22.2   |        8.0   |
|    2880 | TF            | `lmp_plmd_ncsa_delta`   |        16.18  |  32.37  |      30.9   |       89.0   |
|    2880 | PT            | `lmp`                   |        56.73  | 113.45  |       8.8   |       25.4   |
|    2880 | PT + KOKKOS   | `lmp_kk`                |        69.65  | 139.29  |       7.2   |       20.7   |
|   28800 | TF            | `lmp_plmd_ncsa_delta`   |        27.73  | 277.28  |       3.6   |      103.9   |
|   28800 | PT            | `lmp`                   |       105.06  |1050.55  |       0.95  |       27.4   |
|   28800 | PT + KOKKOS   | `lmp_kk`                |       122.05  |1220.48  |       0.82  |       23.6   |

## Backend ratios

| N atoms | PT / TF | KOKKOS / PT |
|--------:|--------:|------------:|
|     360 |  4.48×  |   1.23×     |
|    2880 |  3.51×  |   1.23×     |
|   28800 |  3.79×  |   1.16×     |

## Interpretation

- **PT is ~3.5–4.5× slower than TF, at every system size.** This is consistent across
  the three tile sizes, so it is not a small-system artifact. DeePMD-kit 3.1.3's PT
  backend is simply less optimized than the TF backend for the `se_e2_a` descriptor.
  The PT path is required for some newer model features (spin, DPA-2, etc.), but for
  a plain `se_e2_a` production workload, TF remains the default for performance.
- **KOKKOS hurts at every size we tested (~16–23% slower than plain PT).** This is
  the most important finding from this round. We hoped KOKKOS would amortize at
  larger N where neighbor list and communication start to matter, but the gap stays
  ~20% across 360 → 28800 atoms. The reason: `pair_style deepmd` has no KOKKOS
  variant in DeePMD-kit, so the NN evaluator (which is ~all of the per-step cost,
  even at 28800 atoms — see below) stays on DeepMD's native PT-GPU kernels. KOKKOS
  only accelerates the integrator, neighbor build, and comm — but adds host↔device
  data-sync overhead. Net result: KOKKOS is a tax on this workload, not a benefit.
- **GPU saturation, by per-atom throughput.** TF's `katom-step/s` climbs from 44
  (N=360) to 104 (N=28800) — i.e., the A100 is under-utilized at small N and
  approaches saturation at 30k atoms. PT shows the same shape but plateaus at
  ~27 `katom-step/s` — so the PT backend leaves ~75% of A100 throughput on the
  table even at 28800 atoms.

## Recommendations

- **Default to TF for production MD** at any system size in this scaling range (and
  almost certainly larger).
- **Don't use `lmp_kk` with PT backend** for `pair_style deepmd` — it is a strict
  regression at every size we measured. Revisit only if/when DeePMD-kit ships a
  KOKKOS-aware pair_style.
- The PT-vs-TF gap is intrinsic to the DeePMD-kit PT backend's kernel maturity,
  not a symptom of small N or KOKKOS. If we want to shrink the gap we should track
  upstream PT-backend performance work (e.g. CUDA-graph capture, compiled Torch),
  not invest more time tuning KOKKOS flags here.

## Reproducibility

```
cd shared
python3 tile_conf.py -i conf.lmp -o conf_2880.lmp  --nx 2 --ny 2 --nz 2
python3 tile_conf.py -i conf.lmp -o conf_28800.lmp --nx 4 --ny 4 --nz 5
sbatch train_freeze_compress_pt.sh        # only if shared/model_comp.pth absent

cd ..
for d in variant_TF_N360 variant_TF_N2880 variant_TF_N28800 \
         variant_PT       variant_PT_N2880 variant_PT_N28800 \
         variant_PT_KK    variant_PT_KK_N2880 variant_PT_KK_N28800; do
    (cd "$d" && sbatch sub*.sh)
done
```

All variant sub scripts source `shared/common_env.sh`, which loads the right
modules and conda env (`ALCHEMY_env` for TF, `ALCHEMY_env__PT` for PT and KK).

## Build / input fixes from this round

- Delta's `cudatoolkit/25.3_12.8` does not ship cuFFT inside `$CUDA_HOME`; pointed
  KOKKOS+KSPACE at `/opt/nvidia/cuda-12.8/.../libcufft.so` in `build_lmp_kk.sh`.
- Stale `$CONDA_PREFIX/bin/nvcc` shadows the module nvcc; `build_lmp_kk.sh`
  prepends `$CUDA_HOME/bin` to `PATH` before invoking `cmake`/`nvcc_wrapper`.
- LAMMPS stable_2Aug2023 KOKKOS rejects `read_data` files containing an
  `Atom Type Labels` block (`src/atom.cpp:2205`). Only the symbolic labels were
  affected — atoms already use numeric types and no `pair_coeff` / plumed command
  references the labels — so the block was commented out in-place in
  `shared/conf.lmp`. One file serves both `lmp` and `lmp_kk`.
- SLURM on Delta copies the submit script to `/var/spool/slurmd/job<JOBID>`, so
  `$(dirname "$0")` lands in a root-owned spool dir. All sub scripts use
  `${SLURM_SUBMIT_DIR:-$(dirname "$0")}` and resolve from the real submit dir.
- `shared/in.lammps.bench` declares `NSTEPS_relax` and `NSTEPS_4training` as
  `index`-style variables so per-variant sub scripts can override via `-var`.

---

# Training-speed benchmark — se_e2_a (PT, TF) vs DPA-2 (PT)

Hardware: same node (1× A100, 1 MPI rank, 1 CPU). DeePMD-kit 3.1.3.
Training sets: 3 MgSiOH systems (162 atoms/frame), identical for all 3 runs.
Common settings: 1000 steps, `batch_size: auto` (all picked 1 frame), `start_lr` 1e-3,
`decay_steps` 500, identical loss prefactors.

Architectures:
- `se_e2_a` — `neuron=[25,50,100]`, `axis_neuron=16`, fitting `[240,240,240]` → **2.666 M params**
- `DPA-2`   — `repinit [25,50,100]/nsel=120/rcut=6`, `repformer nlayers=6 g1=64 g2=32 4-head attn / rcut=4`,
  fitting `[240,240,240]` → **0.920 M params**

## Results — 1000 steps (10 warm-up excluded)

| Model             | Backend | Params (M) | s/batch  | Relative |
|-------------------|---------|-----------:|---------:|---------:|
| `se_e2_a`         | TF      |      2.666 |  0.0402  |   1.00×  |
| `se_e2_a`         | PT      |      2.666 |  0.1043  |   2.60×  |
| `DPA-2`           | PT      |      0.920 |  0.1006  |   2.50×  |

## Results — 10000 steps (second, longer run, same configs)

| Model             | Backend | Params (M) | s/batch  | Relative | Wall time |
|-------------------|---------|-----------:|---------:|---------:|----------:|
| `se_e2_a`         | TF      |      2.666 |  0.0406  |   1.00×  |   8:06    |
| `se_e2_a`         | PT      |      2.666 |  0.1048  |   2.58×  |  18:02    |
| `DPA-2`           | PT      |      0.920 |  0.1003  |   2.47×  |  17:15    |

Per-batch timings are stable to ~1% between the 1k and 10k runs, confirming the
initial numbers were not warm-up artifacts. DPA-2 PT ≈ `se_e2_a` PT at batch=1 /
162-atoms — consistent with the overhead-limited regime explanation below.

## Interpretation

- **PT vs TF for `se_e2_a`: 2.6× slower.** Consistent with the ~3.5–4.5× inference
  gap we saw in LAMMPS above — same underlying issue (DeePMD-kit 3.1.3 PT kernels
  less mature than TF for `se_e2_a`).
- **DPA-2 PT ≈ `se_e2_a` PT per batch.** This was *not* expected a-priori — DPA-2
  is a substantially heavier architecture (repformer attention stack + two
  descriptor scales). Reasons the per-batch wall time lands the same here:
  1. **Batch is tiny** (1 frame × 162 atoms). GPU is severely under-utilized for
     both; per-step cost is dominated by Python/launch overhead, not FLOPs. In
     that regime, architectural complexity doesn't manifest — both are bound by
     kernel-launch latency. Expect the gap to open up at larger batch sizes or
     larger systems.
  2. **Our DPA-2 is modest** (`nlayers=6`, `g1_dim=64`, `g2_dim=32`). Realistic
     DPA-2 papers use `nlayers=12` and `g1_dim=128–256`, which is ~4–10× more
     FLOPs per step.
  3. **DPA-2 here has fewer params** (0.92 M < 2.67 M). Parameter count and
     per-step FLOPs don't track linearly across architectures: se_e2_a's fitting
     net (3× 240-wide dense) dominates its param count; DPA-2 spends most of its
     compute on attention kernels that use fewer weights but more arithmetic.
- **Not apples-to-apples in "accuracy per second".** The headline number you
  care about is training time to reach a target force RMSE, *not* s/batch.
  DPA-2 typically reaches a given RMSE with substantially less data (see
  literature note below), so even at equal per-batch cost its time-to-accuracy
  would already favor DPA-2. We did not measure RMSE here.

## Literature (se_e2_a vs DPA-2)

- **DPA-1** (Zhang et al. 2023, *Nat. Commun.*): attention-augmented descriptor
  with a type embedding. Comparable RMSE to `se_e2_a` with ~3–10× fewer training
  frames, and can be pre-trained across chemistries (where `se_e2_a` must be
  retrained per system). Inference is roughly comparable to `se_e2_a` at
  equivalent network size.
- **DPA-2** (Zhang et al. 2024, *npj Comput. Mater.*): two-scale descriptor
  (repinit + repformer attention). On OC2M-like multi-element benchmarks, DPA-2
  reaches `se_e2_a`-level force errors with roughly an order of magnitude fewer
  labeled frames, and matches/beats dedicated single-system `se_e2_a` models
  after fine-tuning from a pre-trained checkpoint. Cost per MD step is reported
  as ~5–20× higher than compressed `se_e2_a`, which is why DPA-2 + distillation
  (or DPA-2 → compressed se_e2_a) is the typical production path when MD
  throughput matters.
- **Practical takeaway** consistent with these papers: use DPA-2 / pre-training
  to reduce the amount of DFT labels you need; use compressed `se_e2_a` for
  the actual long MD runs.

## If we want a "realistic" DPA-2 cost number

The 0.1006 s/batch above is a floor, not a realistic estimate. A more
representative re-run would:
- scale the repformer to `nlayers=12`, `g1_dim=128`, `g2_dim=64`,
- pin `batch_size: 4` or 8 instead of `auto=1`,
- keep `numb_steps` at 500–1000.

Expect per-batch wall time to rise to the 0.5–1.5 s range, matching the ~5–20×
gap the DPA-2 paper reports. Not done in this round.

## Files

- `training_bench/shared/train_se_e2_a.json` — se_e2_a architecture, 1000 steps.
- `training_bench/shared/train_dpa2.json`   — DPA-2 architecture, 1000 steps.
- `training_bench/variant_train_se_e2_a_TF/sub.sh`, `..._PT/sub.sh`,
  `variant_train_dpa2_PT/sub.sh` — each copies the right json to `input.json`,
  activates the right env (`ALCHEMY_env` for TF, `ALCHEMY_env__PT` for PT), and
  runs `dp --{tf,pt} train input.json`. DPA-2 has no TF variant (DPA-2 is
  PT-only in DeePMD-kit 3.x).
