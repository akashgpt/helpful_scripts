#!/bin/bash
#SBATCH --account=burrows           # *** burrows / jiedeng for DELLA and TIGER; bguf-delta-cpu/gpu for NCSA DELTA; not applicable for STELLAR ***
#SBATCH --job-name=qmd              
##SBATCH --partition=cpu            # *** for NCSA DELTA only ***
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks-per-node=16        # total number of tasks per node
#SBATCH --cpus-per-task=8           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G            # memory per cpu-core (4G or 1G is default)
#SBATCH --time=1:00:00              # total run time limit (HH:MM:SS)
#SBATCH --mail-user=akashgpt@princeton.edu
##SBATCH --constraint=intel         # *** for DELLA only ***


##################  Environment  ########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [[ $(hostname) == *"della"* ]] || [[ $(hostname) == *"stellar"* ]] || [[ $(hostname) == *"tiger"* ]]; then
    VASP_BIN=/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std
    module purge
    module load intel-oneapi/2024.2
    module load intel-mpi/oneapi/2021.13
    module load intel-mkl/2024.2
    module load hdf5/oneapi-2024.2/1.14.4
elif [[ $(hostname) == *"delta"* ]]; then
    VASP_BIN=/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std
    module reset
    module load PrgEnv-gnu cray-hdf5-parallel
    ulimit -s unlimited
fi


##############################
###### Log: log.runsim #######
# record job id in log.run_sim
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
# ****************************
# ~~~~~~ CALLING VASP ~~~~~~~~
if [[ $(hostname) == *"della"* ]] || [[ $(hostname) == *"stellar"* ]] || [[ $(hostname) == *"tiger"* ]]; then
    srun "$VASP_BIN" >> log.run_sim
elif [[ $(hostname) == *"delta"* ]]; then
    srun --cpu-bind=cores --distribution=block:block "$VASP_BIN" >> log.run_sim
fi
# ****************************
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
