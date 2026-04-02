#!/bin/bash
set -euo pipefail

# Generic Delta CPU VASP benchmark runner.
#
# Submit this script with sbatch command-line resource options so the same
# script can be reused for multiple core-count benchmarks.

readonly VASP_BIN="${VASP_BIN:-/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std__NCSA_DELTA_MLFF}"

cleanup_job_state() {
    local exit_code

    exit_code=$1
    rm -f running_RUN_VASP
    if [[ ${exit_code} -eq 0 ]]; then
        touch done_RUN_VASP
    else
        touch failed_RUN_VASP
    fi
}

trap 'cleanup_job_state $?' EXIT

module reset
module load PrgEnv-gnu cray-hdf5-parallel
ulimit -s unlimited

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

touch running_RUN_VASP
rm -f done_RUN_VASP log.run_sim

{
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "# ========================================="
    echo "Start time: $(date)"
    echo "Working directory: ${SLURM_SUBMIT_DIR:-$PWD}"
    echo "Node list: ${SLURM_NODELIST}"
    echo "Partition: ${SLURM_JOB_PARTITION}"
    echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
    echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
    echo "Total tasks: ${SLURM_NTASKS}"
    echo "Binary: ${VASP_BIN}"
    echo "# ========================================="
    echo
    module list
    echo
} > log.run_sim 2>&1

SECONDS=0
srun --cpu-bind=cores --distribution=block:block "${VASP_BIN}" >> log.run_sim 2>&1
runtime_seconds="${SECONDS}"

{
    echo
    echo "# ========================================="
    echo "Wall time (s): ${runtime_seconds}"
    echo "Job ${SLURM_JOB_ID} completed at $(date)"
} >> log.run_sim
