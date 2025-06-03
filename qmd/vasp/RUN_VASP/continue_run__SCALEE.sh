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
# ./continue_run.sh <run_dirname> <num_jobs> <RUN_VASP_TIME> <RUN_VASP_NODES> where
# <run_dirname> is the base name of the simulation directory such that simulation folders
# are named <run_dirname>a, <run_dirname>b, ..., where a, b, ... correspond to the number of
# simulations run.
#
# Usage: source $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh <run_dirname> <total_num_jobs_limit> <RUN_VASP_TIME> <RUN_VASP_NODES>
#        e.g., nohup $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh SCALEE_7 > log.continue_run__SCALEE 2>&1 &
#
# Author: Akash Gupta
#####################################################################################
#####################################################################################
#####################################################################################


home_dir=$(pwd)

run_dirname=${1:-0}

total_num_jobs_limit=${2:-10}  # Default to 10 jobs if not specified

RUN_VASP_TIME=${3:-24} #time of simulations, default of 24; options: 0.1, 0.5, 4, 8, 12, 24, 48, 72, 96

CLUSTER_NAME=$(scontrol show config | grep ClusterName | awk '{print $3}')
if [ "$CLUSTER_NAME" == "tiger3" ]; then
	RUN_VASP_NODES=${4:-2} #number of nodes used, default of 2; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "della" ]; then
	RUN_VASP_NODES=${4:-1} #number of nodes used, default of 1; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "stellar" ]; then
	RUN_VASP_NODES=${4:-2} #number of nodes used, default of 2; options: 1, 2, 4, 8
fi

echo ""
echo "Current time: $(date)"
echo "Home directory: $home_dir"
echo "Run directory name: $run_dirname"
echo "Run VASP time for a single job: $RUN_VASP_TIME"
echo "Run VASP nodes: $RUN_VASP_NODES"
echo "Total number of jobs limit: $total_num_jobs_limit"
echo ""

if [ "$run_dirname" == "0" ]; then
    echo "No run_dirname specified"
elif  [ -n "$run_dirname" ]; then
    # run merge_vasp_runs.py in the run_dirname
    source $HELP_SCRIPTS_vasp/merge_vasp_runs.py $run_dirname # > log.merge_vasp_runs 2>&1 &
    echo "Merging VASP runs in directory: $run_dirname"
    cp $HELP_SCRIPTS_vasp/RUN_VASP/RUN_VASP_MASTER_extended.sh $run_dirname/
    cd $run_dirname || exit 1
    run_dir=$(pwd)
    echo "Running VASP in directory: $run_dir"
    total_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
    echo "Total time steps that ${run_dirname} has taken so far: $total_time_steps"
    echo ""
    # while total_time_steps is less than 20,000, continue running RUN_VASP_MASTER_extended__SCALEE.sh

    while [ "$total_time_steps" -lt 20000 ]; do
        cd "$run_dir" || exit 1
        echo ""
        echo "Total time steps is (still) less than 20,000, continuing the run..."

        source RUN_VASP_MASTER_extended__SCALEE.sh $num_jobs $RUN_VASP_TIME $RUN_VASP_NODES > log.RUN_VASP_MASTER_extended 2>&1

        # total_num_jobs=num_jobs+total_num_jobs
        total_num_jobs=$((num_jobs + total_num_jobs))
        echo "Total number of jobs submitted for so far: $total_num_jobs"
        

        # merge all VASP runs
        python $HELP_SCRIPTS_vasp/merge_vasp_runs.py $run_dir

        cd $home_dir || exit 1

        # run merge_vasp_runs.py in the run_dirname
        source $HELP_SCRIPTS_vasp/merge_vasp_runs.py $run_dirname # > log.merge_vasp_runs 2>&1 &
        echo "Merging VASP runs in directory: $run_dirname"
        cd "$run_dir" || exit 1
        total_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
        echo "Total time steps that ${run_dirname} has taken so far: $total_time_steps"
        cd "$home_dir" || exit 1

        # find the last run run_dirname<letter>
        last_run_dirname=$(ls -d ${run_dirname}*/ | sort | tail -n 1)
        echo "Last run directory: $last_run_dirname"
        # change to the last run directory
        cd "$last_run_dirname" || exit 1
        last_run_dir=$(pwd)
        source data_4_analysis.sh > log.data_4_analysis 2>&1 &
        time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
        echo "Time steps that last_run_dirname took: $time_steps"
        # if time_steps is less than 100, exit and print a message
        if [ "$time_steps" -lt 100 ]; then
            echo "Time steps is less than 100 for ${last_run_dirname}, exiting..."
            echo "Have a closer look."
            exit 0
        else
            echo "Time steps is greater than or equal to 100 for the last run, continuing..."
        fi

        # if total_num_jobs > total_num_jobs_limit, exit the loop
        if [ "$total_num_jobs" -gt "$total_num_jobs_limit" ]; then
            echo "Total number of jobs submitted ($total_num_jobs) has exceeded the limit ($total_num_jobs_limit), exiting..."
            exit 0
        fi
    done


fi

