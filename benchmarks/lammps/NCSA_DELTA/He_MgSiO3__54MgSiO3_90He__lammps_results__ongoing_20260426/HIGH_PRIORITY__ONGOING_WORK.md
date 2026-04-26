# High-Priority Revisit

This benchmark package should be treated as a **snapshot of ongoing work**.

Please revisit this benchmark family before using it as final performance guidance.

High-priority follow-up items:

- Re-check whether the PT vs TF inference gap remains the same after any DeePMD-kit,
  PyTorch, or LAMMPS updates.
- Revisit the `PT + KOKKOS` conclusion if upstream DeePMD-kit ever gains a true
  KOKKOS-aware `pair_style deepmd` path.
- Reconcile these LAMMPS-side findings with the related `deepmd` training/model-prep
  benchmarks kept in the companion folder under `benchmarks/deepmd/NCSA_DELTA/...`.

Companion folder:

- [He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426](/projects/bguf/akashgpt/run_scripts/helpful_scripts/benchmarks/deepmd/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__deepmd_results__ongoing_20260426)
