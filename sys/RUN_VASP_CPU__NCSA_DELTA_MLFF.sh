#!/bin/bash
#SBATCH --job-name=qmd_cpu660_delta
#SBATCH --partition=cpu
#SBATCH --account=bguf-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# Hybrid MPI+OpenMP layout chosen to match the existing 96-core Stellar test:
# 12 MPI ranks x 8 OpenMP threads = 96 total CPU cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

readonly VASP_BIN="/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std__NCSA_DELTA_MLFF"

module reset
module load PrgEnv-gnu cray-hdf5-parallel
ulimit -s unlimited

module list
echo "job is starting on $(hostname)"
echo "using ${VASP_BIN}"

touch running_RUN_VASP
rm -f done_RUN_VASP log.run_sim

{
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "# ========================================="
    echo "Start time: $(date)"
    echo "Working directory: ${SLURM_SUBMIT_DIR:-$PWD}"
    echo "Node list: ${SLURM_NODELIST}"
    echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
    echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
    echo "Binary: ${VASP_BIN}"
    echo "# ========================================="
    echo
} > log.run_sim

srun "${VASP_BIN}" >> log.run_sim 2>&1

{
    echo
    echo "# ========================================="
    echo "Job ${SLURM_JOB_ID} completed at $(date)"
} >> log.run_sim

rm -f running_RUN_VASP
touch done_RUN_VASP
