#!/bin/bash

#############################################################################################
################################## TRAIN_MLMD_LOCAL.sh ######################################
#############################################################################################
#
# Script to 1 entire iteration training (Training+MD+recal) process 
#
# Usage: source TRAIN_MLMD_LOCAL.sh MgSiOHN 2 > log.TRAIN_MLMD 2>&1 &
#
# Compatible with continue_training.sh + TRAIN_MLMD_MASTER.sh scripts
#
# Directory structure: Each iteration has a prefix with numbers (=#iteration) as the suffix.
# Example: i1, i2, and so on. In each of these there are two directories: train and md.
# The md directory has folders for the different compositions to be simulated (includes the
# respective conf.lmp files)
#
# Usage: nohup ./RECAL_PHASE_MASTER.sh a05017a >> log.RECAL_PHASE_MASTER 2>&1 &
#        nohup ./RECAL_PHASE_MASTER.sh >> log.RECAL_PHASE_MASTER 2>&1 &
#
# Author: akashgpt
#
#############################################################################################
#############################################################################################
#############################################################################################

echo
echo PID of this script is: $$
echo

RECAL_DIR=${1:-"all"} #default to do recalculations in all directories in the current folder


#################
###  TESTING  ###
#################
PARAMETER_FILE="RECAL_PHASE_parameters.txt"

MY_MLMD_SCRIPTS=$(grep -A 1 "MY_MLMD_SCRIPTS" $PARAMETER_FILE | tail -n 1)

# WAIT_TIME_LONG=$(sed -n '19p' ../TRAIN_MLMD_parameters.txt)
# WAIT_TIME_SHORT=$(sed -n '22p' ../TRAIN_MLMD_parameters.txt)
WAIT_TIME_LONG=$(grep -A 1 "WAIT_TIME_LONG" $PARAMETER_FILE | tail -n 1)
WAIT_TIME_SHORT=$(grep -A 1 "WAIT_TIME_SHORT" $PARAMETER_FILE | tail -n 1) 

MOL_SYSTEM=$(grep -A 1 "MOL_SYSTEM" $PARAMETER_FILE | tail -n 1)

JOB_SUBMIT_COMMAND=$(grep -A 1 "JOB_SUBMIT_COMMAND" $PARAMETER_FILE | tail -n 1)

MAX_RECAL_JOBS=$(grep -A 1 "MAX_RECAL_JOBS" $PARAMETER_FILE | tail -n 1)
NUM_RECAL_FRAMES=$(grep -A 1 "NUM_RECAL_FRAMES" $PARAMETER_FILE | tail -n 1)
#################
###  TESTING  ###
#################


# MAX_RECAL_INSTANCES = MAX_RECAL_JOBS/NUM_RECAL_FRAMES
# MAX_RECAL_INSTANCES=$(( $MAX_RECAL_JOBS/$NUM_RECAL_FRAMES ))

# MOL_SYSTEM=${1:-"MgSiOHN"}
# N_ZONES_PTX=${2:-"1"} #if N_ZONES_PTX = 0: pre-defined PTX regime; > 10: then same PTX regime for all such that N_ZONES_PTX=N_ZONES_PTX-10; else random selection of PTX regime}


current_master_dir=$(pwd)

# exec > ${current_master_dir}/log.RECAL_PHASE_MASTER 2>&1


echo "##################################################################"
echo "##################################################################"
echo "##################################################################"
echo
echo
echo
echo "Starting a phase of RECALs."


cp ${MY_MLMD_SCRIPTS}/RECAL_PHASE_LOCAL.sh .
chmod +x RECAL_PHASE_LOCAL.sh

rm -f recal_* #remove the file if it exists
echo "# Keeping track of completed recal runs" > recal_counter #create new counter files
echo "# Keeping track of completed recal runs w errors" > recal_counter_error


# Function to check if all sub-directories have md_done files
check_recal_done() {
    for dir in */; do
        if [[ ! -f "${dir}recal_done" ]]; then
            return 1
        fi
    done
    return 0
}

# NUM_RECAL_INSTANCES > MAX_RUNS, then wait until NUM_RECAL_INSTANCES < MAX_RUNS
wait_if_too_many_recal() {
    NUM_RECAL_INSTANCES=$(find . -name "recal_running" | wc -l)
    while [ ${NUM_RECAL_INSTANCES} -gt ${MAX_RECAL_INSTANCES} ]; do
        # echo "Waiting for RECALs to finish. Currently ${NUM_RECAL_INSTANCES} RECALs are running."
        sleep ${WAIT_TIME_LONG}
        NUM_RECAL_INSTANCES=$(find . -name "recal_running" | wc -l)
    done
}

wait_if_too_many_submitted_jobs() {
    jobs_running=$(squeue -u $USER -h -t R -o "%.18i %.9P %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | wc -l)
    jobs_pending=$(squeue -u $USER -h -t PD -o "%.18i %.9P %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | wc -l)
    total_jobs=$(($jobs_running + $jobs_pending))

    max_jobs_allowed=$(($MAX_RECAL_JOBS - $NUM_RECAL_FRAMES)) # i.e., before submitting a new one!

    while [ ${total_jobs} -gt ${max_jobs_allowed} ]; do
        sleep ${WAIT_TIME_LONG}
        jobs_running=$(squeue -u $USER -h -t R -o "%.18i %.9P %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | wc -l)
        jobs_pending=$(squeue -u $USER -h -t PD -o "%.18i %.9P %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | wc -l)
        total_jobs=$(($jobs_running + $jobs_pending))
    done
}


# if RECAL_DIR is not equal to "all", that RECAL_PHASE_LOCAL.sh is run only in the specified directory
if [ $RECAL_DIR != "all" ]; then
    echo
    echo "Running RECAL_PHASE_LOCAL.sh for ${RECAL_DIR}"
    echo
    nohup ./RECAL_PHASE_LOCAL.sh ${RECAL_DIR} &
    echo PID of the script running in $RECAL_DIR: $!
    echo
    sleep ${WAIT_TIME_SHORT}
else

    # Loop over only MAX_DIR directories to run RECAL_PHASE_LOCAL.sh, and then wait. As soon as one finishes, start another one. recal_done file is created in each directory when the RECAL_PHASE_LOCAL.sh script is done.
    for dir in */; do
        # if directory includes the word "input_files" in it, skip this directory
        if [[ $dir == *"input_files"* ]]; then
            continue
        fi
        echo "##############################################"
        echo
        echo
        echo
        echo "Running RECAL_PHASE_LOCAL.sh for ${dir}"
        echo
        nohup ./RECAL_PHASE_LOCAL.sh ${dir} &
        echo PID of the script running in $dir: $!
        echo
        echo
        echo
        echo "##############################################"
        sleep ${WAIT_TIME_SHORT}
        # wait if NUM_RECAL_INSTANCES > MAX_RECAL_INSTANCES
        wait_if_too_many_submitted_jobs

        # if NUM_RECAL_INSTANCES > MAX_RUNS, then wait until NUM_RECAL_INSTANCES < MAX_RUNS
    done
fi


MAX_RECAL_INSTANCES=0 #wait while even 1 RECAL_PHASE_LOCAL.sh script is running
wait_if_too_many_recal


################################################################################################
# Wait until all continue_training.sh scripts are done
file_to_check1="recal_counter"
file_to_check2="recal_counter_error"

lines_file_to_check1=$(wc -l < "$file_to_check1")
lines_file_to_check2=$(wc -l < "$file_to_check2") #if > 1, then there are errors in the MD runs
################################################################################################







sleep ${WAIT_TIME_LONG}

# echo
# echo
# echo
# echo "##################################################################"
# echo "##################################################################"
# echo "##################################################################"
# echo
# echo
# echo
# echo "All continue_training.sh scripts have been submitted. Now waiting for them to finish."
# echo "This will take a while ... grab a coffee, go for a walk, workout, or work on the next exciting thing!"
# echo "You could also check the status by going to the respective configuration folders and checking the log.train files."
# echo



echo
echo
echo "##################################################################"
echo "##################################################################"
echo "##################################################################"
echo
echo
echo
if [ $lines_file_to_check2 -gt 1 ]; then
    echo "Recalculations are now done but there are some errors. Check the recal_counter_error file for more details."
else
    echo "Recalculations are now done! Check the logs for more details."
fi
echo
echo
echo
echo "##################################################################"
echo "##################################################################"
echo "##################################################################"


exit 0