# VASP parallel-layout benchmark on Stellar (Princeton)

This folder captures the outcome of a parallelization sweep that was run in
June 2024 to find good `(nodes, ntasks-per-node, cpus-per-task)` layouts and
`NPAR`/`NCORE` choices for VASP 6.3.2 on Stellar, using a representative
H2O+H2 Nose-Hoover NVT MD step as the workload.

## Source of the benchmark data

Raw runs live at:

```
/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2O_H2/sim_data/a08318m_test*
```

77 test directories (`a08318m_test`, `a08318m_test2` … `a08318m_test77`),
each identical in physics and differing only in SLURM / VASP parallel
settings. A reference directory `a08318m_OG` is a legacy UCLA (SGE) run and
is excluded from the Stellar benchmark.

Only sub-data relevant to benchmarking is summarized here; the parent folder
also contains per-run MD output (OUTCAR, XDATCAR, vasprun.xml, analysis/...)
that is NOT backed up to tiger/della — see the cross-cluster note in
`helpful_scripts/CLAUDE.md`.

## System under test

- H2O + H2 mixture, supercell cubic (scale 17.342 Å, 0.5/0.5/1.0 shape).
- **370 atoms total**: 54 O + 316 H.
- INCAR: `ENCUT = 500 eV`, Γ-only 1x1x1 KPOINTS, `GGA = PS` (PBEsol),
  `LREAL = A`, `NBANDS = 468–480`, `ISMEAR=-1` (Fermi smearing via TEBEG),
  NVT MD: `IBRION=0`, `POTIM=0.5 fs`, `TEBEG=TEEND=2000 K`, `SMASS=0`.
- Same POSCAR / INCAR / POTCAR / KPOINTS across all tests; only `NPAR` /
  `NCORE` and SLURM layout vary.

This is a representative "moderately large" supercell run: dense enough that
wavefunction, FFT, and band communication all matter; small enough that
scaling flattens quickly.

## Stellar compute node (as seen by these runs)

- 96 physical cores per node (2 x Intel Xeon Platinum 8360Y Ice Lake, 48c each).
- Confirmed indirectly: configurations `1 x 96 x 1`, `1 x 48 x 2`, `1 x 1 x 96`
  and `2 x 96 x 1` are all present in the sweep.
- Modules used at run time (from `RUN_VASP.sh`):
  `intel/2022.2.0`, `intel-mpi/intel/2021.7.0`, `hdf5/intel-2021.1/intel-mpi/1.10.6`.
- VASP binary: `~/softwares/vasp.6.3.2/bin/vasp_std`.

## How timings were measured

VASP writes per-loop timers to the OUTCAR:

- `LOOP+:  cpu time ... real time ...` — per ionic (MD) step
- `LOOP :  cpu time ... real time ...` — per electronic (SCF) iteration

`analyze_benchmarks.py` walks every test dir, parses SLURM params from
`RUN_VASP.sh`, the INCAR knobs (NPAR/NCORE/KPAR/NBANDS), and the full list
of LOOP+/LOOP real-times from OUTCAR, then reports the **median real time
per ionic step** (first step dropped — it includes one-off setup). Median
is preferred over mean because it is robust to occasional I/O stalls.

Runs that died with segfaults / "SICK JOB" before producing any ionic step
are flagged `failed_crash` in the CSV.

## Files in this directory

| File | What it is |
| --- | --- |
| `analyze_benchmarks.py` | The parser / aggregator. Rerun with `python analyze_benchmarks.py`. |
| `benchmarks.csv` | Full per-test table: SLURM layout, VASP knobs, step counts, per-step times. |
| `SUMMARY.md` | Auto-generated summary: best config per core count, full ranked table, failures, strong-scaling table. |
| `README.md` | This file — methodology, interpretation, recommendations. |

To reproduce: the analyser has no dependencies beyond the Python 3.10+
standard library (uses `statistics`, `csv`, `re`, `pathlib`). On Stellar:

```bash
module load anaconda3/2024.6        # or any python >= 3.10
python /scratch/gpfs/BURROWS/akashgpt/qmd_data/benchmarks/vasp/STELLAR/analyze_benchmarks.py
```

## Key findings (see `SUMMARY.md` for full tables)

### 1. Best layout at each total-core count (H2O+H2, 370 atoms, Γ-point)

| Total cores | Best config (`nodes x ntpn x cpt`) | Best median t / ionic step (s) |
| ---: | --- | ---: |
| 36 | `1 x 36 x 1` | 9.81 |
| 96 | `1 x 96 x 1` (pure MPI, 1 node) | **4.25** |
| 192 | `2 x 12 x 8` (MPI=24, OMP=8) | **2.94** |
| 384 | `4 x 8 x 12` (MPI=32, OMP=12) | **2.15** |
| 768 | `8 x 24 x 4` (MPI=192, OMP=4) | 3.88 — worse than 384 |

### 2. Sweet spot: 4 nodes with hybrid MPI+OpenMP

At 384 cores, the fastest layout is `4 x 8 x 12` (32 MPI ranks × 12 OpenMP
threads each), with `NPAR = 24`. This is **~2× faster than the best
single-node run** (2.15 s vs 4.25 s per ionic step) — strong-scaling
efficiency ≈ 0.50. Pure-MPI `4 x 48 x 2` (192 ranks) is considerably worse
(~5.2 s), showing that at this system size and ENCUT, MPI oversubscription
costs exceed FFT/BLAS threading benefits.

### 3. Scaling breaks past 4 nodes

`8 x 24 x 4` (768 cores, only one configuration tested) runs at 3.88 s /
step — **worse than 4 nodes at half the cores**. Strong scaling is
effectively dead past 384 cores for this 370-atom, Γ-only system.

### 4. Hybrid MPI/OpenMP matters; pure OpenMP is catastrophic

- `1 x 1 x 96` (pure OpenMP, 1 MPI rank) in `test74`: 64.7 s / step —
  more than **15× worse** than `1 x 96 x 1`. Do not do this.
- `1 x 48 x 2` vs `1 x 96 x 1`: the pure-MPI single-node config wins
  (4.25 s vs 5.66 s). Moderate OMP helps only at ≥ 2 nodes.
- `2 x 96 x 1` (pure MPI across 2 nodes) gave 3.64 s and 8.43 s in two
  runs (noisy), vs `2 x 12 x 8` at 2.94 s — hybrid is the right call.

### 5. NPAR / NCORE effect

Same SLURM layout (`1 x 24 x 4`, 96 cores) was repeated with different
NPAR/NCORE settings: median times cluster around **4.69–4.73 s**, a spread
of ~1%. For this system size and layout the NPAR choice is largely a
second-order knob compared to MPI/OMP partition.

### 6. Failed runs

- `a08318m_test69` (`1 x 1 x 96`, NPAR = 96): crashed during setup.
- `a08318m_test77` (`1 x 96 x 1`): `"I REFUSE TO CONTINUE WITH THIS SICK JOB"`
  — physics / INCAR inconsistency on that invocation, not a layout issue.

## Recommended defaults on Stellar for similar systems

For a **Γ-only, ~370-atom, ENCUT≈500 eV AIMD** workload:

- **1 node / 96 cores** for cheap jobs: `#SBATCH --nodes=1 --ntasks-per-node=96 --cpus-per-task=1`, `NPAR=6` or `NPAR=8`.
- **4 nodes / 384 cores** for fastest time-to-step: `--nodes=4 --ntasks-per-node=8 --cpus-per-task=12`, `NPAR=24`. Roughly 2× faster than 1 node.
- Do **not** go to 8+ nodes — throughput decreases.
- Do **not** use pure OpenMP (1 MPI rank + many threads).

For larger systems (more atoms, more k-points) the optimum will shift
toward more nodes; this sweep is specific to this workload.
