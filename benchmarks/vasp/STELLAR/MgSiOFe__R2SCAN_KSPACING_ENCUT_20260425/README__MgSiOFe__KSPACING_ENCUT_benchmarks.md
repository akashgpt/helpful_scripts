# MgSiOFe KSPACING and ENCUT Benchmarks

- Cases: `1000GPa_13000K__1`, `50GPa_3500K__1`
- Total static single-point calculations: `22`
- ENCUT sweep: `400, 500, 600, 800, 1000, 1200 eV`
- KSPACING sweep: `0.20, 0.25, 0.30, 0.40, 0.50`
- The submit script uses 11 Slurm array tasks because Stellar rejected larger arrays under the per-user submit/QOS limits.
- Each Slurm array task requests one CPU node.
- Account/partition/QOS: `astro` / `pu` / `pu-short-stellar`.
- Array concurrency cap: `8` simultaneous one-node tasks, below the requested 20-node ceiling.
- Some array tasks run multiple manifest rows sequentially on the same allocated node to fit the submit limit.
- KSPACING runs omit `KPOINTS` so VASP uses the `KSPACING` INCAR tag. **IMPORTANT: `KGAMMA` IS INTENTIONALLY OMITTED; VASP DEFAULTS TO `KGAMMA = .TRUE.` FOR KSPACING-GENERATED MESHES.**
- ENCUT runs retain the source `2 2 2` Monkhorst-Pack `KPOINTS` file.
- Static benchmark normalization: `ISTART = 0`, `ICHARG = 2`, `LWAVE = .FALSE.`, `NSW = 0`.

Manifest: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test/benchmark_manifest__MgSiOFe__KSPACING_ENCUT.tsv`

Submit script: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test/submit_MgSiOFe__KSPACING_ENCUT__single_node_array.sh`
