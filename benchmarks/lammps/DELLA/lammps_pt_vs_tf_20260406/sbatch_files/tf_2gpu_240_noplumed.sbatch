#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=tf2_240
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32G
#SBATCH --time=0:10:00

module purge
module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate ALCHEMY_env
export PYTHONNOUSERSITE=1

mpirun -np 2 /scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/ALCHEMY_env/bin/lmp -in in.lammps.bench
