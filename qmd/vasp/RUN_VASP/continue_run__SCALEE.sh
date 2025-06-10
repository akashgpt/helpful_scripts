#!/bin/bash

#####################################################################################
#####################################################################################
#####################################################################################
# This script is used to continue a VASP run on the cluster. It takes the following
# arguments: Name of the directory where the VASP run is to be continued, number of
# jobs to be submitted, time of simulations, and number of nodes to be used.
# The script copies the RUN_VASP_MASTER_extended.sh script to the specified directory
# and runs it with the specified arguments.
#
# The script is to be run as follows:
# ./continue_run.sh <RUN_DIRNAME> <NUM_JOBS> <RUN_VASP_TIME> <RUN_VASP_NODES> where
# <RUN_DIRNAME> is the base name of the simulation directory such that simulation folders
# are named <RUN_DIRNAME>a, <RUN_DIRNAME>b, ..., where a, b, ... correspond to the number of
# simulations run.
#
# Usage: source $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh <RUN_DIRNAME> <CUMULATIVE_NUM_JOBS_LIMIT> <RUN_VASP_TIME> <RUN_VASP_NODES>
#        e.g., nohup $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh SCALEE_7 > log.continue_run__SCALEE 2>&1 &
#
# Author: Akash Gupta
#####################################################################################
#####################################################################################
#####################################################################################


NUM_JOBS=5 # Number of jobs to be submitted in each call
NUM_RESTART_SHIFTS=5 # Number of restart shifts to be attempted in each call until NUM_JOBS is reached
TOTAL_TIME_STEP_LIMIT=20000 # Total time steps limit for the run
MINIMUM_TIME_STEP_THRESHOLD=200 # Minimum time steps threshold for the run to be considered successful

INITIAL_PERCENTAGE_RESTART_SHIFT=20 # Initial percentage restart shift in percentage
PERCENTAGE_RESTART_SHIFT_INCREMENT=5 # Percentage increment for restart shift after each attempt

home_dir=$(pwd)

RUN_DIRNAME=${1:-0}

CUMULATIVE_NUM_JOBS_LIMIT=${2:-10}  # Cumulative total number of jobs limit = NUM_JOBS_i + NUM_JOBS_(i+1) + ... + NUM_JOBS_n, default of 10; options: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100



module purge
module load anaconda3/2024.6; conda activate ase_env
        



CLUSTER_NAME=$(scontrol show config | grep ClusterName | awk '{print $3}')
if [ "$CLUSTER_NAME" == "tiger3" ]; then
	RUN_VASP_NODES=${4:-2} #number of nodes used, default of 2; options: 1, 2, 4, 8
    RUN_VASP_TIME=${3:-5} #time of simulations, default of 24; options: 0.1, 0.5, 4, 8, 12, 24, 48, 72, 96
elif [ "$CLUSTER_NAME" == "della" ]; then
	RUN_VASP_NODES=${4:-1} #number of nodes used, default of 1; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "stellar" ]; then
	RUN_VASP_NODES=${4:-2} #number of nodes used, default of 2; options: 1, 2, 4, 8
    RUN_VASP_TIME=${3:-24} #time of simulations, default of 24; options: 0.1, 0.5, 4, 8, 12, 24, 48, 72, 96
fi

echo ""
echo "Current time: $(date)"
echo "Cluster name: $CLUSTER_NAME"
echo "Home directory: $home_dir"
echo "Run directory name: $RUN_DIRNAME"
echo "Run VASP time for a single job: $RUN_VASP_TIME"
echo "Run VASP nodes: $RUN_VASP_NODES"
echo "Total number of jobs limit: $CUMULATIVE_NUM_JOBS_LIMIT"
echo "Number of jobs being submitted in each call: $NUM_JOBS"
echo "Number of restart shifts to be attempted: $NUM_RESTART_SHIFTS"
echo "Initial percentage restart shift: $INITIAL_PERCENTAGE_RESTART_SHIFT%"
echo "Percentage restart shift increment: $PERCENTAGE_RESTART_SHIFT_INCREMENT%"
echo "Total time steps limit for the run: $TOTAL_TIME_STEP_LIMIT"
echo "Minimum time steps threshold for a 'successful' run: $MINIMUM_TIME_STEP_THRESHOLD"
echo ""

cumulative_num_jobs_submitted=0 # Initialize cumulative number of jobs submitted so far
current_num_restart_shifts=0 # Initialize current number of restart shifts


if [ "$RUN_DIRNAME" == "0" ]; then
    echo "No RUN_DIRNAME specified"
elif  [ -n "$RUN_DIRNAME" ]; then


    while [ $current_num_restart_shifts -lt $NUM_RESTART_SHIFTS ]; do
        # current_percentage_restart_shift=$INITIAL_PERCENTAGE_RESTART_SHIFT + $PERCENTAGE_RESTART_SHIFT_INCREMENT * current_num_restart_shifts
        current_percentage_restart_shift=$((INITIAL_PERCENTAGE_RESTART_SHIFT + PERCENTAGE_RESTART_SHIFT_INCREMENT * current_num_restart_shifts))

        cd "$home_dir" || exit 1
        # run merge_vasp_runs.py in the RUN_DIRNAME
        python $HELP_SCRIPTS_vasp/merge_vasp_runs.py $RUN_DIRNAME
        echo "Merging VASP runs in directory: $RUN_DIRNAME"

        cp $HELP_SCRIPTS_vasp/RUN_VASP/RUN_VASP_MASTER_extended__SCALEE.sh $RUN_DIRNAME/
        cd $RUN_DIRNAME || exit 1
        run_dir=$(pwd)
        echo "Running VASP in directory: $run_dir"
        total_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
        echo "Total time steps that ${RUN_DIRNAME} has taken so far: $total_time_steps"
        echo ""

        # while total_time_steps is less than $TOTAL_TIME_STEP_LIMIT, continue running RUN_VASP_MASTER_extended__SCALEE.sh
        while [ "$total_time_steps" -lt $TOTAL_TIME_STEP_LIMIT ]; do
            cd "$run_dir" || exit 1
            echo ""
            echo "Total time steps is (still) less than $TOTAL_TIME_STEP_LIMIT, continuing the run..."

            rm -rf done_RUN_VASP_MASTER_extended__SCALEE  # remove the done file if it exists

            source RUN_VASP_MASTER_extended__SCALEE.sh $NUM_JOBS $RUN_VASP_TIME $RUN_VASP_NODES $current_percentage_restart_shift > log.RUN_VASP_MASTER_extended__SCALEE 2>&1 &

            echo "Submitted RUN_VASP_MASTER_extended__SCALEE.sh with $NUM_JOBS jobs, time: $RUN_VASP_TIME hours, and nodes: $RUN_VASP_NODES."
            echo "Waiting for the jobs to finish..."

            # wait until $run_dir/done_RUN_VASP_MASTER_extended__SCALEE file exists
            while [ ! -f "$run_dir/done_RUN_VASP_MASTER_extended__SCALEE" ]; do
                sleep 600 # wait for 10 minutes
            done

            # cumulative_num_jobs_submitted=NUM_JOBS+cumulative_num_jobs_submitted
            cumulative_num_jobs_submitted=$((NUM_JOBS + cumulative_num_jobs_submitted))
            echo "Total number of jobs submitted for so far: $cumulative_num_jobs_submitted"
            

            cd $home_dir || exit 1

            # merge all VASP runs
            python $HELP_SCRIPTS_vasp/merge_vasp_runs.py $RUN_DIRNAME
            echo "Merging VASP runs in directory: $RUN_DIRNAME"

            cd "$run_dir" || exit 1
            total_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
            echo "Total time steps that ${RUN_DIRNAME} has taken so far: $total_time_steps"
            cd "$home_dir" || exit 1

            # find the last run RUN_DIRNAME<letter>
            last_run_dirname=$(ls -d ${RUN_DIRNAME}*/ | sort | tail -n 1)
            echo "Last run directory: $last_run_dirname"
            # change to the last run directory
            cd "$last_run_dirname" || exit 1
            last_run_dir=$(pwd)
            cp $HELP_SCRIPTS_vasp/data_4_analysis.sh .
            source data_4_analysis.sh > log.data_4_analysis 2>&1 &
            time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
            echo "Time steps that last_run_dirname took: $time_steps"
            # if time_steps is less than MINIMUM_TIME_STEP_THRESHOLD, exit and print a message
            if [ "$time_steps" -lt $MINIMUM_TIME_STEP_THRESHOLD ]; then
                echo "Time steps is less than $MINIMUM_TIME_STEP_THRESHOLD for ${last_run_dirname}, exiting..."
                # delete the last run directory
                cd "$home_dir" || exit 1
                echo "Deleting the last run directory: $last_run_dirname"
                rm -rf "$last_run_dirname"
                echo ""
                echo ""
                echo ""
                # increment the current_num_restart_shifts
                echo "Incrementing the current number of restart shifts to $((current_num_restart_shifts + 1))"
                echo "Percentage restart shift for the next run will be: $((INITIAL_PERCENTAGE_RESTART_SHIFT + PERCENTAGE_RESTART_SHIFT_INCREMENT * (current_num_restart_shifts + 1)))%"
                current_num_restart_shifts=$((current_num_restart_shifts + 1))
                echo ""
                # if current_num_restart_shifts is greater than NUM_RESTART_SHIFTS, exit the script
                if [ "$current_num_restart_shifts" -ge "$NUM_RESTART_SHIFTS" ]; then
                    echo "Current number of restart shifts ($current_num_restart_shifts) has exceeded the limit ($NUM_RESTART_SHIFTS), exiting..."
                    exit 0
                fi

                # exit just the loop to change CURRENT_PERCENTAGE_RESTART_SHIFT
                break
            else
                echo "Time steps is greater than or equal to 100 for the last run, continuing..."
            fi

            # if cumulative_num_jobs_submitted > CUMULATIVE_NUM_JOBS_LIMIT, exit the loop
            if [ "$cumulative_num_jobs_submitted" -gt "$CUMULATIVE_NUM_JOBS_LIMIT" ]; then
                echo "Total number of jobs submitted ($cumulative_num_jobs_submitted) has exceeded the limit ($CUMULATIVE_NUM_JOBS_LIMIT), exiting..."
                exit 0
            fi
        done
    done

fi

module purge
echo ""
echo "Current time: $(date)"
echo "All runs completed. Check individual log files in each run directory."