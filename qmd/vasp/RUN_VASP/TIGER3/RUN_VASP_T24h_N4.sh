#!/bin/bash
#SBATCH --account=burrows
#SBATCH --job-name=vasp          # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=14    # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ag5805@princeton.edu


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export PATH=$PATH:/scratch/gpfs/BURROWS/akashgpt/softwares/vasp.6.3.2/bin

module purge

## Tiger
#module load intel/19.1/64/19.1.1.217
#module load intel-mkl/2020.1/1/64
#module load intel-mpi/intel/2019.7/64
#module load hdf5/intel-17.0/intel-mpi/1.10.0
#module load cudatoolkit/11.1

## Della
# module load intel/2021.1.2 intel-mpi/intel/2021.1.1

## Stellar
# module load intel/2021.1 intel-mkl/2021.1.1 intel-mpi/intel/2021.1.1
# module load intel/2022.2.0
# module load intel-mpi/intel/2021.7.0
# module load hdf5/intel-2021.1/intel-mpi/1.10.6

## Tiger3
module purge
module load intel-oneapi/2024.2
module load intel-mpi/oneapi/2021.13
module load intel-mkl/2024.2
module load hdf5/oneapi-2024.2/1.14.4


srun vasp_std > log.run_sim
