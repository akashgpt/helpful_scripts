#!/bin/bash
set -euo pipefail

# Submit diagnostic 10x-step cases.
# All jobs use walltime 01:00:00.

(cd runs/long_steps_10x/2gpu_5k_linear_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/4gpu_2500_none_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/8gpu_1250_none_10x && sbatch run_srun_train_mem.sbatch)
