#!/bin/bash
set -euo pipefail

# Submit all 10x-step cases.
# All jobs use walltime 01:00:00.

(cd runs/long_steps_10x/reuse_1gpu_10k_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/reuse_2gpu_10k_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/reuse_4gpu_10k_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/reuse_8gpu_10k_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/reuse_16gpu_10k_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/2gpu_5k_linear_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/4gpu_2500_linear_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/4gpu_2500_sqrt_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/4gpu_2500_none_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/8gpu_1250_linear_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/8gpu_1250_sqrt_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/8gpu_1250_none_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/16gpu_625_linear_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/8gpu_7k_linear_10x && sbatch run_srun_train_mem.sbatch)
(cd runs/long_steps_10x/16gpu_5k_linear_10x && sbatch run_srun_train_mem.sbatch)
