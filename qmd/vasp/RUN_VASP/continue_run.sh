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
# ./continue_run.sh <run_dir> <num_jobs> <RUN_VASP_TIME> <RUN_VASP_NODES> where
# <run_dir> is the base name of the simulation directory such that simulation folders
# are named <run_dir>a, <run_dir>b, ..., where a, b, ... correspond to the number of
# simulations run.
#
# Usage: source continue_run.sh <run_dir> <num_jobs> <RUN_VASP_TIME> <RUN_VASP_NODES>
#
# Author: Akash Gupta
#####################################################################################
#####################################################################################
#####################################################################################




run_dir=${1:-0}

num_jobs=${2:-5}  # Default to 5 jobs if not specified

RUN_VASP_TIME=${3:-24} #time of simulations, default of 24; options: 0.1, 0.5, 4, 8, 12, 24, 48, 72, 96

CLUSTER_NAME=$(scontrol show config | grep ClusterName | awk '{print $3}')
if [ "$CLUSTER_NAME" == "tiger3" ]; then
	RUN_VASP_NODES=${4:-2} #number of nodes used, default of 2; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "della" ]; then
	RUN_VASP_NODES=${4:-1} #number of nodes used, default of 1; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "stellar" ]; then
	RUN_VASP_NODES=${4:-2} #number of nodes used, default of 2; options: 1, 2, 4, 8
fi

if [ "$run_dir" == "0" ]; then
    echo "No run_dir specified"
elif  [ -n "$run_dir" ]; then
    mkdir -p $run_dir
    cp $AG_BURROWS/run_scripts/VASP_scripts/RUN_VASP_MASTER_extended.sh $run_dir/
    cd $run_dir || exit 1
    source RUN_VASP_MASTER_extended.sh $num_jobs $RUN_VASP_TIME $RUN_VASP_NODES > log.RUN_VASP_MASTER_extended 2>&1 &
fi

