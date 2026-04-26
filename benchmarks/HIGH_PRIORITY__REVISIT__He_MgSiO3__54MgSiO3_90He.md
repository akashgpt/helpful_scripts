# High-Priority Reminder: Revisit He/MgSiO3 Delta Benchmarks

The testing originally stored under:

- `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He`

has now been split into two benchmark packages because it contains **both** `lammps`
and `deepmd` results:

- [LAMMPS results](/projects/bguf/akashgpt/run_scripts/helpful_scripts/benchmarks/lammps/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__lammps_results__ongoing_20260426)
- [DeePMD results](/projects/bguf/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426)

This work is still **ongoing** and should be treated as **high priority to revisit**.

Why this reminder matters:

- the `LAMMPS` side currently says TF inference is clearly better than PT and that
  `PT + KOKKOS` is a regression for this workload
- the `DeePMD` side currently says PT training is slower than TF for `se_e2_a`, but
  the realistic cost of larger `DPA-2` models is still not settled

Before using these results as final workflow guidance, revisit:

- larger `DPA-2` training benchmarks
- updated PT/TF backend performance after software changes
- whether any upstream `pair_style deepmd` / KOKKOS improvements change the LAMMPS result
