#!/bin/bash
set -euo pipefail

# Submit only the sample_matched cases.
# Review the matrix and queue state before running this.

(cd runs/sample_matched/2gpu_5k_linear && sbatch run_srun_train_mem.sbatch)
(cd runs/sample_matched/4gpu_2500_linear && sbatch run_srun_train_mem.sbatch)
(cd runs/sample_matched/8gpu_1250_linear && sbatch run_srun_train_mem.sbatch)
(cd runs/sample_matched/16gpu_625_linear && sbatch run_srun_train_mem.sbatch)
