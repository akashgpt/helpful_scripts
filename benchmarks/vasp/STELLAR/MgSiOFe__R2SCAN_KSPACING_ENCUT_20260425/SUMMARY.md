# MgSiOFe__R2SCAN static convergence benchmark

- Source benchmark directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test`
- Manifest rows: `22`
- Converged runs analyzed: `21`
- Incomplete or non-converged rows skipped: `1`
- Shared runtime footprint: `96 MPI ranks, 1 thread/rank`
- Skipped rows:
  - `ENCUT` / `50GPa_3500K__1` / `ENCUT_1000`: no final EDIFF convergence marker

## KSPACING observations

- `1000GPa_13000K__1`: reference is `KSPACING_0p20`; loosest completed setting within 1 meV/atom in both metrics is `KSPACING_0p50` (0.50); max |dTOTEN| = `0.0321` meV/atom; max |dE_internal| = `0.0532` meV/atom; runtime span = `44.4` to `611.2` min.
- `50GPa_3500K__1`: reference is `KSPACING_0p20`; loosest completed setting within 1 meV/atom in both metrics is `KSPACING_0p50` (0.50); max |dTOTEN| = `0.0073` meV/atom; max |dE_internal| = `0.0094` meV/atom; runtime span = `122.9` to `537.2` min.

## ENCUT observations

- `1000GPa_13000K__1`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_1200` (1200 eV); max |dTOTEN| = `15.8067` meV/atom; max |dE_internal| = `15.9142` meV/atom; runtime span = `64.2` to `417.1` min.
- `50GPa_3500K__1`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0800` (800 eV); max |dTOTEN| = `6.7870` meV/atom; max |dE_internal| = `6.8025` meV/atom; runtime span = `152.0` to `987.2` min.

## Runtime notes

- Mean runtime across all converged runs: `303.3` min.
- Slowest converged run: `ENCUT_1200` at `987.2` min.
- Fastest converged run: `KSPACING_0p50` at `44.4` min.
