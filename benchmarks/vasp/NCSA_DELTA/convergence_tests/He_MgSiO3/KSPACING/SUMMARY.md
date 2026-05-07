# He_MgSiO3 static KSPACING sweep static convergence benchmark

- Source benchmark directory: `/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/setup_MLMD/benchmarking_tests/KSPACING_test`
- Manifest rows: `20`
- Converged runs analyzed: `20`
- Incomplete or non-converged rows skipped: `0`
- Shared runtime footprint: `1 MPI ranks, 1 thread/rank`

## KSPACING observations

- `32_MgSiO3__160_atoms__and_He, 1MgSiO3_155He`: reference is `KSPACING_0p20`; loosest completed setting within 1 meV/atom in both metrics is `KSPACING_0p50` (0.50); max |dTOTEN| = `0.0002` meV/atom; max |dE_internal| = `0.0009` meV/atom; runtime span = `16.2` to `35.9` min.
- `32_MgSiO3__160_atoms__and_He, 31MgSiO3_5He`: reference is `KSPACING_0p20`; loosest completed setting within 1 meV/atom in both metrics is `KSPACING_0p50` (0.50); max |dTOTEN| = `0.0075` meV/atom; max |dE_internal| = `0.0127` meV/atom; runtime span = `14.8` to `32.9` min.
- `72_MgSiO3__360_atoms__and_He, 1MgSiO3_355He`: reference is `KSPACING_0p20`; loosest completed setting within 1 meV/atom in both metrics is `KSPACING_0p50` (0.50); max |dTOTEN| = `0.0008` meV/atom; max |dE_internal| = `0.0012` meV/atom; runtime span = `19.6` to `49.0` min.
- `72_MgSiO3__360_atoms__and_He, 71MgSiO3_5He`: reference is `KSPACING_0p20`; loosest completed setting within 1 meV/atom in both metrics is `KSPACING_0p50` (0.50); max |dTOTEN| = `0.0619` meV/atom; max |dE_internal| = `0.0914` meV/atom; runtime span = `22.1` to `53.9` min.

## ENCUT observations


## Runtime notes

- Mean runtime across all converged runs: `34.1` min.
- Slowest converged run: `KSPACING_0p30` at `53.9` min.
- Fastest converged run: `KSPACING_0p50` at `14.8` min.
