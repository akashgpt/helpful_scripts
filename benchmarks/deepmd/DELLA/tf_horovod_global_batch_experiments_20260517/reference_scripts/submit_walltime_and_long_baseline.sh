#!/bin/bash
set -euo pipefail

# Submit only the walltime_and_long_baseline cases.
# Review the matrix and queue state before running this.

(cd runs/walltime_and_long_baseline/1gpu_20k_linear && sbatch run_srun_train_mem.sbatch)
