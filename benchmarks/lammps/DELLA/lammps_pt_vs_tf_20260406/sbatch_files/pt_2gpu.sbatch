#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=pt_lmp2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32G
#SBATCH --time=0:30:00

# LAMMPS benchmark: ALCHEMY_env__PT (PT backend), 2 GPUs, .pt model
# MPI domain decomposition — 2 MPI ranks, 1 GPU each

module purge
module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate ALCHEMY_env__PT
export PYTHONNOUSERSITE=1

LMP=/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/ALCHEMY_env__PT/bin/lmp
mpirun -np 2 $LMP -in in.lammps.bench
