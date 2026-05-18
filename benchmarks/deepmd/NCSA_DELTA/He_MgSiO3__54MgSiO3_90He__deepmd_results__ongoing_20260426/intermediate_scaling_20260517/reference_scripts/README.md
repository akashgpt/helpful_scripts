# Reference Scripts

Scripts archived here document the 2026-05-17 intermediate scaling run setup.

- `setup_submit_intermediate_se_e2_a.py`
  Generates the shared JSONs, writes per-run Slurm scripts, and submits the five
  intermediate TensorFlow `se_e2_a` jobs. This script plus the compact architecture
  manifest is the reference record; duplicate full generated JSON inputs are not stored
  in this benchmark folder.

- `calc_energy_rmse_sigma.py`
  Recomputes held-out energy RMSE/atom bootstrap intervals from `dp_test.e_peratom.out`
  files in the validation workspace.

- `submission_scripts/`
  Copies of the exact Slurm scripts submitted for the intermediate runs. These scripts
  hardcode `HERE=...` because Slurm executes a copied script from `/var/spool/slurmd`,
  so resolving paths via `BASH_SOURCE[0]` is not reliable in this environment.

Do not archive raw `log.train`, `slurm-*`, checkpoint, frozen-model, compressed-model,
or duplicated generated `input.json` / `input_v2_compat.json` files here. Summarize their
lessons in markdown or TSV form instead.

The active working path at the time these scripts were archived was:

`/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench`
