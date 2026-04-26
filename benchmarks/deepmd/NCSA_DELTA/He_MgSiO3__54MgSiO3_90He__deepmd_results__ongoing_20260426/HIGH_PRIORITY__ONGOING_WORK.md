# High-Priority Revisit

This benchmark package should be treated as a **snapshot of ongoing work**.

Please revisit this benchmark family before using it as final performance guidance.

High-priority follow-up items:

- Re-run a more realistic `DPA-2` cost benchmark with the larger `big` / `big_v2`
  style architectures rather than relying on the small-batch baseline alone.
- Revisit the source-summary recommendation to test larger batch sizes such as `4` or `8`
  so the benchmark moves out of the strongly overhead-limited regime.
- Compare the eventual `deepmd` conclusion against the companion `lammps` inference
  benchmark before choosing a final production workflow.

Companion folder:

- [He_MgSiO3__54MgSiO3_90He__lammps_results__ongoing_20260426](/projects/bguf/akashgpt/run_scripts/helpful_scripts/benchmarks/lammps/NCSA_DELTA/He_MgSiO3__54MgSiO3_90He__lammps_results__ongoing_20260426)
