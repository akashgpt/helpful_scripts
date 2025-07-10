#!/bin/bash
##SBATCH --account=burrows
#SBATCH --job-name=multi_run      # short name for this multi‐run job
#SBATCH --output=multi_run_%j.out # output file name with job ID
#SBATCH --error=multi_run_%j.err  # error file name with job ID
#SBATCH --nodes=4                  # total nodes (4 jobs × 4 nodes each)
#SBATCH --ntasks-per-node=8       # MPI ranks per node
#SBATCH --cpus-per-task=7          # threads per rank
#SBATCH --partition=normal          # partition to run on
#SBATCH --time=48:00:00
#SBATCH --mail-user=akashgpt@tacc.utexas.edu
#SBATCH --mail-type=END,FAIL            # send email on all events (start, end, fail)

##################  Environment  ########################################

# ensure OMP/MPI see the right thread count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK   # some MPI wrappers honour this

module purge
module load intel/19.1.1
module load impi/19.0.9
module load gnuparallel/git20190729
module load vasp/6.4.1

# directory + date
echo "Running VASP multi‐run script"
echo "Time: $(date)"
echo "Location: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

##############################
###### Log: log.runsim #######
# record job id in log.run_sim
touch running_RUN_VASP

if [ -f log.run_sim ]; then
    rm log.run_sim
fi
echo "Job ID: $SLURM_JOB_ID" > log.run_sim
echo "# =========================================" >> log.run_sim
echo "" >> log.run_sim

##############################
####### CALLING VASP ########
srun vasp_std >> log.run_sim
#############################

echo "" >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "Job $SLURM_JOB_ID completed at" `date ` >> log.run_sim

rm running_RUN_VASP
touch done_RUN_VASP
###### Log: log.runsim #######
##############################
