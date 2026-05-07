# He_MgSiO3 static ENCUT sweep static convergence benchmark

- Source benchmark directory: `/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/setup_MLMD/benchmarking_tests/ENCUT_test`
- Manifest rows: `24`
- Converged runs analyzed: `24`
- Incomplete or non-converged rows skipped: `0`
- Shared runtime footprint: `1 MPI ranks, 1 thread/rank`

## KSPACING observations


## ENCUT observations

- `32_MgSiO3__160_atoms__and_He, 1MgSiO3_155He`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0500` (500 eV); max |dTOTEN| = `8.2206` meV/atom; max |dE_internal| = `8.2150` meV/atom; runtime span = `13.0` to `18.6` min.
- `32_MgSiO3__160_atoms__and_He, 31MgSiO3_5He`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0800` (800 eV); max |dTOTEN| = `7.2246` meV/atom; max |dE_internal| = `7.2586` meV/atom; runtime span = `11.6` to `18.0` min.
- `72_MgSiO3__360_atoms__and_He, 1MgSiO3_355He`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0500` (500 eV); max |dTOTEN| = `7.8704` meV/atom; max |dE_internal| = `7.8646` meV/atom; runtime span = `18.3` to `40.3` min.
- `72_MgSiO3__360_atoms__and_He, 71MgSiO3_5He`: reference is `ENCUT_1200`; lowest completed setting within 1 meV/atom in both metrics is `ENCUT_0800` (800 eV); max |dTOTEN| = `7.9329` meV/atom; max |dE_internal| = `7.9412` meV/atom; runtime span = `19.4` to `39.4` min.

## Runtime notes

- Mean runtime across all converged runs: `21.2` min.
- Slowest converged run: `ENCUT_1200` at `40.3` min.
- Fastest converged run: `ENCUT_0500` at `11.6` min.
