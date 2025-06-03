#!/bin/bash
#SBATCH --job-name=vasp          # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=8      # total number of tasks per node
#SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PATH:$HOME/softwares/vasp.6.3.2/bin

module purge
#module load intel/2021.1.2 intel-mpi/intel/2021.1.1
module load intel/19.1/64/19.1.1.217
module load intel-mkl/2020.1/1/64
module load intel-mpi/intel/2019.7/64
module load hdf5/intel-17.0/intel-mpi/1.10.0


srun vasp_std > log.run_sim
