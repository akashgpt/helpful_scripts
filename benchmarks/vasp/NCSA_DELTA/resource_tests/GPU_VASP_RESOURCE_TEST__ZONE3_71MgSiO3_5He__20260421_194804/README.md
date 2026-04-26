# GPU VASP Resource Test

Sample input copied from `/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/v1_i1/md/ZONE_3/71MgSiO3_5He/pre/recal/1`.

Each subdirectory contains one `sub_vasp_gpu.sh` based on the project template.
Each job records `gpu_memory_trace.csv` with `nvidia-smi` snapshots while VASP runs.
`NELM` is set to 8 so these are resource smoke tests, not production convergence runs.

Submit with:

```bash
bash submit_all.sh
```

Summarize with:

```bash
bash summarize_results.sh
```
