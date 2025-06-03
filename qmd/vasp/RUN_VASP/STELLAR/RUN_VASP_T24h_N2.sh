#!/bin/bash
#SBATCH --job-name=vasp          # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=12      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ag5805@princeton.edu

# ensure OMP/MPI see the right thread count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK   # some MPI wrappers honour this

export PATH=$PATH:$HOME/softwares/vasp.6.3.2/bin


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
module load intel/2021.1 intel-mkl/2021.1.1 intel-mpi/intel/2021.1.1
# module load intel/2022.2.0
# module load intel-mpi/intel/2021.7.0
module load hdf5/intel-2021.1/intel-mpi/1.10.6

srun vasp_std > log.run_sim
