#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=tf_lmp1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=0:10:00

# LAMMPS benchmark: ALCHEMY_env (TF backend), 1 GPU, .pb model

module purge
module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate ALCHEMY_env
export PYTHONNOUSERSITE=1

LMP=/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/ALCHEMY_env/bin/lmp
$LMP -in in.lammps.bench
