#!/bin/bash
set -euo pipefail

# Submit only the lr_sensitivity cases.
# Review the matrix and queue state before running this.

(cd runs/lr_sensitivity/4gpu_2500_sqrt && sbatch run_srun_train_mem.sbatch)
(cd runs/lr_sensitivity/4gpu_2500_none && sbatch run_srun_train_mem.sbatch)
(cd runs/lr_sensitivity/8gpu_1250_sqrt && sbatch run_srun_train_mem.sbatch)
(cd runs/lr_sensitivity/8gpu_1250_none && sbatch run_srun_train_mem.sbatch)
