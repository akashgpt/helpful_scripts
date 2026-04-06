#!/bin/bash
#SBATCH --account=jiedeng       # not valid for Stellar
#SBATCH --job-name=dp_train      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --gres=gpu:a100:1             # number of gpus per node
#SBATCH --mem=40G                # memory per cpu-core (4G is default)
#SBATCH --time=0:10:00          # total run time limit (HH:MM:SS)
##SBATCH --mail-type=begin       # send email when job begins
##SBATCH --mail-type=end         # send email when job ends
##SBATCH --mail-user=ag5805@princeton.edu

export DP_INFER_BATCH_SIZE=32768

module purge
module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12	
conda activate ALCHEMY_env
export PYTHONNOUSERSITE=1

dp train myinput.json
