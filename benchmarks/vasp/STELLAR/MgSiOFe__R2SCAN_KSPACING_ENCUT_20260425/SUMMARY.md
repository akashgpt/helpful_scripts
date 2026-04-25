# MgSiOFe R2SCAN static convergence benchmark

- Source benchmark directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test`
- Manifest rows: `22`
- Converged runs analyzed: `19`
- Incomplete or non-converged rows skipped: `3`
- Shared runtime footprint: `96 MPI ranks, 1 thread/rank`
- ENCUT sweep: `400, 500, 600, 800, 1000, 1200 eV`
- KSPACING sweep: `0.20, 0.25, 0.30, 0.40, 0.50`
- Skipped rows:
  - `ENCUT` / `50GPa_3500K__1` / `ENCUT_1000`: electronic self-consistency not achieved
  - `KSPACING` / `50GPa_3500K__1` / `KSPACING_0p40`: no final EDIFF convergence marker
  - `KSPACING` / `50GPa_3500K__1` / `KSPACING_0p50`: no final EDIFF convergence marker

## KSPACING observations

- `1000GPa_13000K__1`: reference is `KSPACING_0p20`; loosest completed setting within 1 meV/atom in both metrics is `KSPACING_0p50` (0.50); max |dTOTEN| = `0.0321` meV/atom; max |dE_internal| = `0.0532` meV/atom; runtime span = `44.4` to `611.2` min.
- `50GPa_3500K__1`: reference is `KSPACING_0p20`; loosest completed setting within 1 meV/atom in both metrics is `KSPACING_0p30` (0.30); max |dTOTEN| = `0.0000` meV/atom; max |dE_internal| = `0.0000` meV/atom; runtime span = `534.9` to `537.2` min.

## ENCUT observations

- `1000GPa_13000K__1`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_1200` (1200 eV); max |dTOTEN| = `15.8067` meV/atom; max |dE_internal| = `15.9142` meV/atom; runtime span = `64.2` to `417.1` min.
- `50GPa_3500K__1`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0800` (800 eV); max |dTOTEN| = `6.7870` meV/atom; max |dE_internal| = `6.8025` meV/atom; runtime span = `152.0` to `987.2` min.

## Runtime notes

- Mean runtime across all completed runs: `322.2` min.
- Slowest run: `ENCUT_1200` at `987.2` min.
- Fastest run: `KSPACING_0p50` at `44.4` min.
