#!/bin/bash
set -euo pipefail

# Submit core 10x-step cases.
# All jobs use walltime 01:00:00.

(cd runs/long_steps_10x/reuse_1gpu_10k_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/reuse_4gpu_10k_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/reuse_8gpu_10k_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/8gpu_7k_linear_10x && sbatch run_srun_train_mem.sbatch)
