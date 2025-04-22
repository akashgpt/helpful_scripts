#!/bin/bash

####################################################################################################
# Summary:
# This script launches a VASP simulation run based on user-supplied parameters.
# Usage:
#   source $HELP_SCRIPTS/qmd/vasp/continue_run.sh [run_dir] [num_jobs] [RUN_VASP_TIME] [RUN_VASP_NODES]
#   source $HELP_SCRIPTS_vasp/continue_run.sh [run_dir] [num_jobs] [RUN_VASP_TIME] [RUN_VASP_NODES]
#   - run_dir: Directory to set up and run the simulation (default: 0, meaning no directory specified)
#   - num_jobs: Number of jobs to run (default: 5)
#   - RUN_VASP_TIME: Simulation time in hours (default: 24; valid options include: 0.1, 0.5, 4, 8, 12, 24, 48, 72, 96)
#   - RUN_VASP_NODES: (Optional) Number of nodes to use; the default depends on the cluster:
#                     tiger2 defaults to 4, della defaults to 1, stellar defaults to 2.
#
# The script determines the cluster name from system configuration and sets the appropriate default for nodes.
# If a valid run_dir is provided (i.e. not "0"), the script creates that directory, copies the VASP launcher
# script into it, changes to that directory, and then sources the master script with the provided parameters.
#
# Author: akashgpt
####################################################################################################

run_dir=${1:-0}

num_jobs=${2:-5}  # Default to 5 jobs if not specified

RUN_VASP_TIME=${3:-24}  # time of simulations, default of 24; options: 0.1, 0.5, 1, 4, 8, 12, 24, 48, 72, 96
                        # if 0 -- retains OG RUN_VASP.sh file

CLUSTER_NAME=$(scontrol show config | grep ClusterName | awk '{print $3}')
if [ "$CLUSTER_NAME" == "tiger2" ]; then
	RUN_VASP_NODES=${4:-4} # number of nodes used, default of 4; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "tiger3" ]; then
	RUN_VASP_NODES=${4:-1} # number of nodes used, default of 1; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "della" ]; then
	RUN_VASP_NODES=${4:-1} # number of nodes used, default of 1; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "stellar" ]; then
	RUN_VASP_NODES=${4:-2} # number of nodes used, default of 2; options: 1, 2, 4, 8
fi

if [ "$run_dir" == "0" ]; then
    echo "No run_dir specified"
elif  [ -n "$run_dir" ]; then
    mkdir -p $run_dir
    cp $AG_BURROWS/run_scripts/VASP_scripts/RUN_VASP_MASTER_extended.sh $run_dir/
    cd $run_dir || exit 1
    source RUN_VASP_MASTER_extended.sh $num_jobs $RUN_VASP_TIME $RUN_VASP_NODES
fi