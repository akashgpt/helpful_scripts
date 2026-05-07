# He_MgSiO3 static ENCUT sweep static convergence benchmark

- Source benchmark directory: `/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/setup_MLMD/benchmarking_tests/ENCUT_test_KSPACING_0p50`
- Manifest rows: `24`
- Converged runs analyzed: `24`
- Incomplete or non-converged rows skipped: `0`
- Shared runtime footprint: `1 MPI ranks, 1 thread/rank`

## KSPACING observations


## ENCUT observations

- `32_MgSiO3__160_atoms__and_He, 1MgSiO3_155He`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0500` (500 eV); max |dTOTEN| = `7.9887` meV/atom; max |dE_internal| = `7.9833` meV/atom; runtime span = `8.8` to `12.0` min.
- `32_MgSiO3__160_atoms__and_He, 31MgSiO3_5He`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0800` (800 eV); max |dTOTEN| = `7.3356` meV/atom; max |dE_internal| = `7.3696` meV/atom; runtime span = `8.5` to `11.4` min.
- `72_MgSiO3__360_atoms__and_He, 1MgSiO3_355He`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0500` (500 eV); max |dTOTEN| = `8.0803` meV/atom; max |dE_internal| = `8.0757` meV/atom; runtime span = `12.3` to `14.4` min.
- `72_MgSiO3__360_atoms__and_He, 71MgSiO3_5He`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0800` (800 eV); max |dTOTEN| = `7.6209` meV/atom; max |dE_internal| = `7.6285` meV/atom; runtime span = `11.7` to `16.0` min.

## Runtime notes

- Mean runtime across all converged runs: `11.7` min.
- Slowest converged run: `ENCUT_1200` at `16.0` min.
- Fastest converged run: `ENCUT_0500` at `8.5` min.
