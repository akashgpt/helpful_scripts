# Stellar VASP benchmark — efficiency vs. knob analysis

## 0. System under test and why it caps scaling

- **Atoms:** 370 total (54 O + 316 H), constant across all 77 tests.
- **NBANDS:** 468 (20 early tests), 480 (55 tests, the bulk), 512 (2 tests: `test35`, `test40`). Same POSCAR/POTCAR/KPOINTS otherwise — only NBANDS varies.
- **K-points:** Γ-only (1×1×1, Monkhorst), so `KPAR = 1` always; no k-point parallelism is in play.
- **Why this matters for scaling:** VASP distributes bands across `mpi_ranks / NPAR` groups of `NPAR` ranks, so average load ≈ `NBANDS / mpi_ranks` bands per rank.
  - Healthy: **≥ 4 bands/rank** (enough work per rank, small overhead).
  - Stressed: **2–4 bands/rank** (comms and synchronization start to dominate).
  - Starving: **< 2 bands/rank** (rank holds ~1 band; parallelism breaks down).
- With ~480 bands, the ceiling on productive MPI ranks is roughly **120–240 ranks** for this workload. Beyond that, adding cores via MPI buys nothing; adding cores via OMP threads may still help, up to ~8–12 threads/rank on Stellar.

**Primary metric:** `steps_per_sec_per_core` = 1 / (median_real_per_ionic_step × total_cores).
Think of it as "MD steps produced per core per wall-clock second"; higher = more efficient.
Reported below as **steps · s⁻¹ · (1000 cores)⁻¹** so numbers are readable (×1000 vs. raw value).

**Throughput** = steps/sec = 1 / t_step. **Core-seconds/step** = t_step × cores; lower = cheaper per step in allocation terms.

Ideal strong scaling ⇒ `steps/sec/core` constant as cores grow. If it drops, each added core is delivering less than the previous one.

## 1. Strong-scaling: best config at each total-core count

For each core budget, take the best-performing (fastest per-step) run and show: time per step, total throughput, per-core efficiency, core-seconds per step, and the bands/rank load.

| total_cores | best layout (n×ntpn×cpt) | NPAR | NBANDS | mpi_ranks | bands/rank | t_step (s) | steps/s | steps/s/(1k cores) | core·s/step |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1×36×1 | 6 | 468 | 36 | 13.0 | 9.811 | 0.1019 |   2.83 | 353.2 |
| 96 | 1×96×1 | 6 | 468 | 96 | 4.88 | 4.254 | 0.2351 |   2.45 | 408.4 |
| 192 | 2×12×8 | 12 | 480 | 24 | 20.0 | 2.938 | 0.3403 |   1.77 | 564.2 |
| 384 | 4×8×12 | 24 | 480 | 32 | 15.0 | 2.149 | 0.4654 |   1.21 | 825.1 |
| 768 | 8×24×4 | 6 | 468 | 192 | 2.44 | 3.875 | 0.2581 |   0.34 | 2976.2 |

**Read this as**: the `steps/s/(1k cores)` column is absolute per-core efficiency. Dividing row values by the single-node (96-core) entry gives strong-scaling efficiency. The `bands/rank` column is the single best predictor of where efficiency collapses: the best layouts cluster around 15–20 bands/rank, while stressed ones (≤ 5) match poor efficiency.

## 2. Effect of MPI/OMP split at fixed core budget

Each row is one (total_cores, ntasks_per_node × cpus_per_task) combination. If a combo was tested multiple times (different NPAR / reruns), we report min/median/max of per-step time across those reruns.

### total_cores = 36

| layout (n×ntpn×cpt) | mpi_ranks | omp | bands/rank* | runs | t_step min / med / max (s) | best steps/s/(1k cores) |
|---|---:|---:|---:|---:|---|---:|
| 1×36×1 | 36 | 1 | 13 | 1 | 9.811 / 9.811 / 9.811 |   2.83 |

### total_cores = 96

| layout (n×ntpn×cpt) | mpi_ranks | omp | bands/rank* | runs | t_step min / med / max (s) | best steps/s/(1k cores) |
|---|---:|---:|---:|---:|---|---:|
| 1×96×1 | 96 | 1 | 4.88/5 | 8 | 4.254 / 5.250 / 7.465 |   2.45 |
| 1×24×4 | 24 | 4 | 19.5 | 9 | 4.675 / 4.706 / 4.728 |   2.23 |
| 1×12×8 | 12 | 8 | 39 | 1 | 4.926 / 4.926 / 4.926 |   2.11 |
| 1×48×2 | 48 | 2 | 9.75 | 1 | 5.658 / 5.658 / 5.658 |   1.84 |
| 1×1×96 | 1 | 96 | 480 | 1 | 64.726 / 64.726 / 64.726 |   0.16 |

### total_cores = 192

| layout (n×ntpn×cpt) | mpi_ranks | omp | bands/rank* | runs | t_step min / med / max (s) | best steps/s/(1k cores) |
|---|---:|---:|---:|---:|---|---:|
| 2×12×8 | 24 | 8 | 20 | 3 | 2.938 / 2.955 / 2.961 |   1.77 |
| 2×16×6 | 32 | 6 | 15 | 4 | 2.982 / 3.001 / 3.014 |   1.75 |
| 2×24×4 | 48 | 4 | 9.75/10 | 5 | 3.250 / 3.256 / 3.270 |   1.60 |
| 2×8×12 | 16 | 12 | 30 | 4 | 3.297 / 3.389 / 3.419 |   1.58 |
| 2×96×1 | 192 | 1 | 2.44 | 2 | 3.641 / 6.035 / 8.428 |   1.43 |
| 2×4×24 | 8 | 24 | 60 | 4 | 3.814 / 3.828 / 3.837 |   1.37 |
| 2×6×16 | 12 | 16 | 40 | 3 | 5.216 / 5.390 / 6.583 |   1.00 |

### total_cores = 384

| layout (n×ntpn×cpt) | mpi_ranks | omp | bands/rank* | runs | t_step min / med / max (s) | best steps/s/(1k cores) |
|---|---:|---:|---:|---:|---|---:|
| 4×8×12 | 32 | 12 | 15/16 | 4 | 2.149 / 2.167 / 2.316 |   1.21 |
| 4×4×24 | 16 | 24 | 30 | 3 | 2.457 / 2.459 / 2.461 |   1.06 |
| 4×16×6 | 64 | 6 | 7.5/8 | 4 | 2.545 / 2.562 / 2.592 |   1.02 |
| 4×24×4 | 96 | 4 | 4.88/5 | 6 | 2.939 / 2.954 / 3.002 |   0.89 |
| 4×32×3 | 128 | 3 | 3.75 | 3 | 3.613 / 3.615 / 3.632 |   0.72 |
| 4×6×16 | 24 | 16 | 20 | 3 | 3.809 / 3.885 / 4.149 |   0.68 |
| 4×48×2 | 192 | 2 | 2.5 | 5 | 5.167 / 5.184 / 5.196 |   0.50 |

### total_cores = 768

| layout (n×ntpn×cpt) | mpi_ranks | omp | bands/rank* | runs | t_step min / med / max (s) | best steps/s/(1k cores) |
|---|---:|---:|---:|---:|---|---:|
| 8×24×4 | 192 | 4 | 2.44 | 1 | 3.875 / 3.875 / 3.875 |   0.34 |

`*` bands/rank = NBANDS / mpi_ranks. When a cell lists multiple values (e.g. `15/16`) it's because different runs in that layout used different NBANDS (468 vs 480 vs 512).

## 3. Marginal effect of individual knobs

For each knob, we aggregate best-per-(total_cores, knob_value) runs, so we're measuring the knob's effect, not conflating it with core-count effects.

### 3.a  — ntasks_per_node (MPI ranks per node)

Rows = ntasks_per_node value. Columns = total_cores budget. Cell = best steps/s/(1k cores) observed for that (knob, cores) pair, with sample count in parens.

| ntasks_per_node |     36 |     96 |    192 |    384 |    768 |
|---|---|---|---|---|---|
| 1        |      — |  0.16(1) |      — |      — |      — |
| 4        |      — |      — |  1.37(4) |  1.06(3) |      — |
| 6        |      — |      — |  1.00(3) |  0.68(3) |      — |
| 8        |      — |      — |  1.58(4) |  1.21(4) |      — |
| 12       |      — |  2.11(1) |  1.77(3) |      — |      — |
| 16       |      — |      — |  1.75(4) |  1.02(4) |      — |
| 24       |      — |  2.23(9) |  1.60(5) |  0.89(6) |  0.34(1) |
| 32       |      — |      — |      — |  0.72(3) |      — |
| 36       |  2.83(1) |      — |      — |      — |      — |
| 48       |      — |  1.84(1) |      — |  0.50(5) |      — |
| 96       |      — |  2.45(8) |  1.43(2) |      — |      — |

### 3.b  — cpus_per_task (OMP threads per MPI rank)

Rows = cpus_per_task value. Columns = total_cores budget. Cell = best steps/s/(1k cores) observed for that (knob, cores) pair, with sample count in parens.

| cpus_per_task |     36 |     96 |    192 |    384 |    768 |
|---|---|---|---|---|---|
| 1        |  2.83(1) |  2.45(8) |  1.43(2) |      — |      — |
| 2        |      — |  1.84(1) |      — |  0.50(5) |      — |
| 3        |      — |      — |      — |  0.72(3) |      — |
| 4        |      — |  2.23(9) |  1.60(5) |  0.89(6) |  0.34(1) |
| 6        |      — |      — |  1.75(4) |  1.02(4) |      — |
| 8        |      — |  2.11(1) |  1.77(3) |      — |      — |
| 12       |      — |      — |  1.58(4) |  1.21(4) |      — |
| 16       |      — |      — |  1.00(3) |  0.68(3) |      — |
| 24       |      — |      — |  1.37(4) |  1.06(3) |      — |
| 96       |      — |  0.16(1) |      — |      — |      — |

### 3.c  — NPAR (band-parallel group size)

Rows = NPAR value. Columns = total_cores budget. Cell = best steps/s/(1k cores) observed for that (knob, cores) pair, with sample count in parens.

| NPAR     |     36 |     96 |    192 |    384 |    768 |
|---|---|---|---|---|---|
| 2        |      — |  1.85(2) |      — |      — |      — |
| 4        |      — |  2.34(1) |      — |      — |      — |
| 6        |  2.83(1) |  2.45(5) |  1.60(3) |  0.88(1) |  0.34(1) |
| 8        |      — |  2.22(1) |  1.76(5) |  0.88(2) |      — |
| 12       |      — |  2.39(2) |  1.77(2) |      — |      — |
| 16       |      — |      — |  1.74(4) |  1.20(7) |      — |
| 24       |      — |  2.22(2) |  1.76(6) |  1.21(7) |      — |
| 32       |      — |      — |  1.75(4) |  1.20(7) |      — |
| 48       |      — |  2.21(2) |      — |  0.50(1) |      — |
| 64       |      — |      — |      — |  1.12(2) |      — |
| 96       |      — |  1.40(2) |      — |  0.89(1) |      — |

### 3.d  — nodes (node count)

Rows = nodes value. Columns = total_cores budget. Cell = best steps/s/(1k cores) observed for that (knob, cores) pair, with sample count in parens.

| nodes    |     36 |     96 |    192 |    384 |    768 |
|---|---|---|---|---|---|
| 1        |  2.83(1) |  2.45(20) |      — |      — |      — |
| 2        |      — |      — |  1.77(25) |      — |      — |
| 4        |      — |      — |      — |  1.21(28) |      — |
| 8        |      — |      — |      — |      — |  0.34(1) |

## 4. NPAR / NCORE sensitivity at fixed (nodes, ntpn, cpt)

Isolates the NPAR / NCORE effect by restricting to groups with ≥ 2 NPAR/NCORE values tested under the same SLURM layout. Spread = (max − min) / min of t_step.

| layout (n×ntpn×cpt) | n_runs | NPAR values | NCORE values | t_step range (s) | spread |
|---|---:|---|---|---|---:|
| 1×24×4 | 9 | 6,8,12,24,48 | 24,48,96 | 4.675 – 4.728 | 1.1% |
| 1×96×1 | 8 | 2,4,6,12,24,48,96 | — | 4.254 – 7.465 | 75.5% |
| 2×4×24 | 4 | 8,16,24,32 | — | 3.814 – 3.837 | 0.6% |
| 2×6×16 | 3 | 6,12,24 | — | 5.216 – 6.583 | 26.2% |
| 2×8×12 | 4 | 8,16,24,32 | — | 3.297 – 3.419 | 3.7% |
| 2×12×8 | 3 | 8,12,24 | — | 2.938 – 2.961 | 0.8% |
| 2×16×6 | 4 | 8,16,24,32 | — | 2.982 – 3.014 | 1.1% |
| 2×24×4 | 5 | 6,8,16,24,32 | — | 3.250 – 3.270 | 0.6% |
| 4×4×24 | 3 | 16,24,32 | — | 2.457 – 2.461 | 0.1% |
| 4×6×16 | 3 | 16,24,32 | — | 3.809 – 4.149 | 8.9% |
| 4×8×12 | 4 | 16,24,32,64 | — | 2.149 – 2.316 | 7.8% |
| 4×16×6 | 4 | 16,24,32,64 | — | 2.545 – 2.592 | 1.9% |
| 4×24×4 | 6 | 6,8,16,24,32,96 | — | 2.939 – 3.002 | 2.2% |
| 4×32×3 | 3 | 16,24,32 | — | 3.613 – 3.632 | 0.5% |
| 4×48×2 | 5 | 8,16,24,32,48 | — | 5.167 – 5.196 | 0.6% |

## 4b. Efficiency vs. bands/rank (NBANDS / mpi_ranks)

This is the single knob where NBANDS and the atom count enter. Below ~4–5 bands/rank the run is starving for band-parallel work and efficiency collapses.

Binning 'ok' runs by bands/rank bucket; each row shows the best run in that bucket (lowest t_step).

| bands/rank range | runs | best layout seen | NBANDS | cores | mpi_ranks | t_step (s) | steps/s/(1k cores) |
|---|---:|---|---:|---:|---:|---:|---:|
| 2 – 5 (stressed) | 15 | 4×24×4 | 468 | 384 | 96 | 2.954 |   0.88 |
| 5 – 10 (healthy) | 16 | 4×16×6 | 480 | 384 | 64 | 2.545 |   1.02 |
| 10 – 20 (comfortable) | 22 | 4×8×12 | 480 | 384 | 32 | 2.149 |   1.21 |
| 20 – 100 (under-utilized) | 21 | 4×4×24 | 480 | 384 | 16 | 2.457 |   1.06 |
| ≥ 100 (single-rank) | 1 | 1×1×96 | 480 | 96 | 1 | 64.726 |   0.16 |

**Observed trend (this benchmark):** best-in-bucket efficiency flattens around `≥ 10 bands/rank`; drops markedly below ~5 bands/rank; and is essentially wasted below ~2.5 bands/rank (the 768-core `8×24×4` test has 468/192 ≈ 2.44 bands/rank — exactly where scaling falls off a cliff).

**Rule of thumb derived from this sweep (NBANDS ≈ 480, Γ-only, 370 atoms):**

- Target `mpi_ranks ≤ NBANDS / 10` for comfortable scaling → ~48 ranks.
- Hard ceiling at `mpi_ranks ≈ NBANDS / 2.5` → ~192 ranks. Past this, extra cores should be OMP threads, not new ranks.
- At 384-core budget: 32 MPI × 12 OMP (`4×8×12`) gives 15 bands/rank — best observed. 192 MPI × 2 OMP (`4×48×2`) gives 2.5 bands/rank — half the efficiency of best.
- To go beyond 4 nodes productively, either increase NBANDS (more bands to distribute) or increase the atom count (heavier per-rank work).

## 5. Absolute leaderboard by per-core efficiency

Top 10 by `steps_per_sec_per_core` (i.e. highest per-core throughput — the best cost per core-hour).

| rank | name | n×ntpn×cpt | NPAR | NBANDS | bands/rank | cores | t_step (s) | steps/s/(1k cores) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | a08318m_test16 | 1×36×1 | 6 | 468 | 13.0 | 36 | 9.811 |   2.83 |
| 2 | a08318m_test3 | 1×96×1 | 6 | 468 | 4.88 | 96 | 4.254 |   2.45 |
| 3 | a08318m_test70 | 1×96×1 | 12 | 480 | 5.0 | 96 | 4.365 |   2.39 |
| 4 | a08318m_test75 | 1×96×1 | 4 | 480 | 5.0 | 96 | 4.448 |   2.34 |
| 5 | a08318m_test13 | 1×24×4 | 6 | 468 | 19.5 | 96 | 4.675 |   2.23 |
| 6 | a08318m_test8 | 1×24×4 | 8 | 468 | 19.5 | 96 | 4.691 |   2.22 |
| 7 | a08318m_test14 | 1×24×4 | 24 | 468 | 19.5 | 96 | 4.692 |   2.22 |
| 8 | a08318m_test9 | 1×24×4 | 12 | 468 | 19.5 | 96 | 4.700 |   2.22 |
| 9 | a08318m_test | 1×24×4 | 6 | 468 | 19.5 | 96 | 4.706 |   2.21 |
| 10 | a08318m_test15 | 1×24×4 | None | 468 | 19.5 | 96 | 4.706 |   2.21 |

Worst 5 (pathological configs — low per-core efficiency):

| name | n×ntpn×cpt | NPAR | NBANDS | bands/rank | cores | t_step (s) | steps/s/(1k cores) |
|---|---|---:|---:|---:|---:|---:|---:|
| a08318m_test74 | 1×1×96 | 96 | 480 | 480.0 | 96 | 64.726 |   0.16 |
| a08318m_test7 | 8×24×4 | 6 | 468 | 2.44 | 768 | 3.875 |   0.34 |
| a08318m_test34 | 4×48×2 | 48 | 480 | 2.5 | 384 | 5.196 |   0.50 |
| a08318m_test31 | 4×48×2 | 16 | 480 | 2.5 | 384 | 5.193 |   0.50 |
| a08318m_test32 | 4×48×2 | 24 | 480 | 2.5 | 384 | 5.184 |   0.50 |

## 6. Interpretation — what each knob does to efficiency

The four knobs (`nodes`, `ntasks_per_node`, `cpus_per_task`, `NPAR`) interact, but the marginal-effect tables above isolate each one. Reading them together:

- **`nodes` (scale):** Per-core efficiency drops as node count rises once the first node is saturated. The 36-core run (1 node, under-subscribed) is actually 15% more efficient per used core than the 96-core run (1 node, fully subscribed) — the 60 idle cores on the under-subscribed node free up memory bandwidth, L3 cache, and UPI/NUMA traffic for the 36 active cores. This is a scheduling quirk, not a sign that 36 cores is the "right" budget: absolute throughput (steps/s) is still much lower. From the fully-loaded 1-node point onward (96 → 192 → 384 → 768 cores), per-core efficiency drops monotonically. Total throughput rises 1 → 4 nodes, then falls at 8 — communication dominates past 4 nodes for this 370-atom, Γ-point system.

- **`ntasks_per_node` vs `cpus_per_task` (MPI/OMP balance):** Within a node-count, the best layouts concentrate around `ntpn × cpt ≈ 96` with moderate OMP (cpt = 8–12). Very high MPI counts (`ntpn = 48`, `ntpn = 96` on ≥ 2 nodes) are inefficient; very high OMP (`cpt = 96`, pure OpenMP) is catastrophic. **Pure MPI is fine on one node; hybrid is needed past one node.**

- **`NPAR` / `NCORE`:** Effect is layout-dependent. On well-chosen layouts (§4: `1×24×4`, `2×12×8`, `4×8×12`) the spread across NPAR values is **1–4%** — minor. On a few layouts (`1×96×1` with NPAR=96 vs 6 → 75% spread; `2×6×16` → 26% spread) NPAR matters a lot — usually when NPAR is set to a value that doesn't divide `mpi_ranks` cleanly or forces extreme band-group sizes. Safe rule: use NPAR ≈ number of nodes, or NPAR such that `mpi_ranks / NPAR` ≈ OMP threads.

- **`total_cores`:** Throughput peaks at 384 cores (4 nodes) for this system. Beyond that, steps/sec drops even as cores are added. Cost-effectiveness (per-core efficiency) peaks at the smallest core count tested (36-core, 2.83 steps/s/(1k cores); but this is a partially-loaded node — see nodes bullet above) and falls from there — typical strong-scaling behavior.

## 7. Decision guide (for 370 atoms, NBANDS ≈ 480, Γ-only)

- If walltime is the constraint → use 384 cores, `4×8×12`, NPAR ≈ 24. bands/rank ≈ 15, the largest productive layout for this system size.
- If AU / core-hour cost is the constraint → use 1 node, `1×96×1`, NPAR ≈ 6-8. bands/rank ≈ 5, still close-to-ideal per-core efficiency.
- **Avoid:** pure-OMP (`1×1×96`), very-high-MPI on ≥ 2 nodes (`n×96×1`, `n×48×2`), `8×*×*` (past the scaling wall — bands/rank drops below 2.5).
- **Before scaling to a new system size:** estimate `NBANDS / target_mpi_ranks`. If it would drop below 10, either use fewer MPI ranks (more OMP threads) or accept reduced efficiency.
