#!/bin/bash
set -euo pipefail

# Submit only the walltime_matched cases.
# Review the matrix and queue state before running this.

(cd runs/walltime_matched/8gpu_7k_linear && sbatch run_srun_train_mem.sbatch)
(cd runs/walltime_matched/16gpu_5k_linear && sbatch run_srun_train_mem.sbatch)
