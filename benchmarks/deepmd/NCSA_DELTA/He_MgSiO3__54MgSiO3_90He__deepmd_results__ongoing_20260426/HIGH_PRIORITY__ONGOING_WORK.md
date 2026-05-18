# High-Priority Revisit

This benchmark package should be treated as a **snapshot of ongoing work**.

Please revisit this benchmark family before using it as final performance guidance.

High-priority follow-up items:

- Track the 2026-05-17 intermediate `se_e2_a` scaling jobs in
  [intermediate_scaling_20260517](./intermediate_scaling_20260517). When they finish,
  validate them on the same `71MgSiO3_5He` held-out split and update the energy-focused
  ranking. The key scientific question is whether `balanced_2x` or `balanced_5x` can
  recover most of the `balanced_10x` energy improvement at lower training cost.
- Re-run a more realistic `DPA-2` cost benchmark with the larger `big` / `big_v2`
  style architectures rather than relying on the small-batch baseline alone.
- Revisit the source-summary recommendation to test larger batch sizes such as `4` or `8`
  so the benchmark moves out of the strongly overhead-limited regime.
- Compare the eventual `deepmd` conclusion against the companion `lammps` inference
  benchmark before choosing a final production workflow.

Companion folder:

- [He_MgSiO3__54MgSiO3_90He__lammps_results__ongoing_20260426](/projects/bguf/akashgpt/run_scripts/helpful_scripts/benchmarks/lammps/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__lammps_results__ongoing_20260426)
