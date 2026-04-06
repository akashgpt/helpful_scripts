#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=pt_2x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00

# LAMMPS size-scaling benchmark: PT, 1 GPU, 1920 atoms (2x2x2 replicate)

module purge
module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate ALCHEMY_env__PT
export PYTHONNOUSERSITE=1

LMP=/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/ALCHEMY_env__PT/bin/lmp
$LMP -in in.lammps.bench
