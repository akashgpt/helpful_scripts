#!/bin/bash
## VASP 6.6.0 GPU -- NCSA DELTA -- 1 GPU (A100-SXM4 80 GB)
## Status: WORKING (elapsed ~1405 s on benchmark system)
## Build:  HPC-X OpenMPI (nvhpc-hpcx-cuda12/25.3)
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=qmd_gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=00:30:00


##################  Environment  ########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VASP_BIN=/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin/vasp_std__NCSA_DELTA_GPU
module reset
module load nvhpc-hpcx-cuda12/25.3 intel-oneapi-mkl/2024.2.2
ulimit -s unlimited


##############################
###### Log: log.runsim #######
touch running_RUN_VASP
rm -f done_RUN_VASP

if [ -f log.run_sim ]; then
	cat log.run_sim >> old.log.run_sim
fi
echo "Job ID: $SLURM_JOB_ID" > log.run_sim
echo "Started at $(date)" >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "" >> log.run_sim

# ****************************
# ~~~~~~ CALLING VASP ~~~~~~~~
mpirun --bind-to none -np 1 "$VASP_BIN" >> log.run_sim
# ****************************

echo "" >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "Job $SLURM_JOB_ID completed at" `date ` >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "" >> log.run_sim

rm running_RUN_VASP
touch done_RUN_VASP
###### Log: log.runsim #######
##############################
