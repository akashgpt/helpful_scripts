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
# Usage: source $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh <RUN_DIRNAME> <RESTART_MODE> <RUN_VASP_TIME> <RUN_VASP_NODES> <CUMULATIVE_NUM_JOBS_SUBMITTED_LIMIT>
#        e.g., nohup bash $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh SCALEE_7 > log.continue_run__SCALEE_7 2>&1 &
#        e.g., nohup bash $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh SCALEE_7 5 > log.continue_run__SCALEE_7 2>&1 &
#        e.g., nohup bash $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh SCALEE_7 5 24 > log.continue_run__SCALEE_7 2>&1 &
#        e.g., nohup bash $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh SCALEE_7 0 24 > log.continue_run__SCALEE_7 2>&1 &
#
# Author: Akash Gupta
#####################################################################################
#####################################################################################
#####################################################################################

CURRENT_PROCESS_ID=$$
echo "Current process ID: $CURRENT_PROCESS_ID"


NUM_JOBS=5 # Number of jobs to be submitted in each call of RUN_VASP_MASTER_extended__SCALEE.sh
NUM_RESTART_SHIFTS=50 # Number of restart shifts to be attempted in each call until NUM_JOBS is reached
TOTAL_TIME_STEP_LIMIT=20000 # Total time steps limit for the run
MINIMUM_TIME_STEP_THRESHOLD=160 # Minimum time steps threshold for the run to be considered successful
ALGO_SWITCH=1 # Switch to use ALGO = All in INCAR file, default of 1 (on); options: 0 (off), 1 (on)

INITIAL_PERCENTAGE_RESTART_SHIFT=20 # Initial percentage restart shift in percentage
PERCENTAGE_RESTART_SHIFT_INCREMENT=1 # Percentage increment for restart shift after each attempt

home_dir=$(pwd)

RUN_DIRNAME=${1:-0}

RESTART_MODE=${2:0} # Run mode, default of 0: initial condition based on RUN_DIRNAME{last_run} (just as usual); 
                    # 5: initial condition based on shifting SCALEE_5
# if [ "$RESTART_MODE" -gt 0 ]; then
#     echo ""
#     echo "RESTART_MODE: $RESTART_MODE. Changing NUM_JOBS to 1 (if different)."
#     NUM_JOBS=1
#     echo ""
# fi

CLUSTER_NAME=$(scontrol show config | grep ClusterName | awk '{print $3}')
if [ "$CLUSTER_NAME" == "tiger3" ]; then
    RUN_VASP_TIME=${3:-5} #time of simulations, default of 5; options: 0.1, 0.5, 4, 8, 12, 24, 48, 72, 96
	RUN_VASP_NODES=${4:-2} #number of nodes used, default of 2; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "della" ]; then
    RUN_VASP_TIME=${3:-24} #time of simulations, default of 24; options: 0.1, 0.5, 4, 8, 12, 24, 48, 72, 96
	RUN_VASP_NODES=${4:-1} #number of nodes used, default of 1; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "stellar" ]; then
    RUN_VASP_TIME=${3:-24} #time of simulations, default of 24; options: 0.1, 0.5, 4, 8, 12, 24, 48, 72, 96
	RUN_VASP_NODES=${4:-2} #number of nodes used, default of 2; options: 1, 2, 4, 8
fi

CUMULATIVE_NUM_JOBS_SUBMITTED_LIMIT=${5:-200}  # Cumulative total number of jobs limit SUBMITTED = NUM_JOBS_i + NUM_JOBS_(i+1) + ... + NUM_JOBS_n, default of 30; options: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
CUMULATIVE_NUM_JOBS_COMPLETED_LIMIT=${6:-40}  # Cumulative total number of jobs limit COMPLETED = NUM_JOBS_i + NUM_JOBS_(i+1) + ... + NUM_JOBS_n, default of 30; options: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100



# sed -i 's/ALGO   = F/ALGO   = All/' "SCALEE_6b/INCAR" && sed -i 's/ALGO   = F/ALGO   = All/' "SCALEE_7b/INCAR"


module purge
module load anaconda3/2024.6; conda activate ase_env
        





echo ""
echo "Current time: $(date)"
echo "Cluster name: $CLUSTER_NAME"
echo "Home directory: $home_dir"
echo "Run directory name: $RUN_DIRNAME"
echo "Restart mode: $RESTART_MODE"
echo "Run VASP time for a single job: $RUN_VASP_TIME"
echo "Run VASP nodes: $RUN_VASP_NODES"
echo "Total number of jobs limit SUBMITTED: $CUMULATIVE_NUM_JOBS_SUBMITTED_LIMIT"
echo "Total number of jobs limit COMPLETED: $CUMULATIVE_NUM_JOBS_COMPLETED_LIMIT"
echo "Number of jobs being submitted in each call: $NUM_JOBS"
echo "Number of restart shifts to be attempted: $NUM_RESTART_SHIFTS"
echo "Initial percentage restart shift: $INITIAL_PERCENTAGE_RESTART_SHIFT%"
echo "Percentage restart shift increment: $PERCENTAGE_RESTART_SHIFT_INCREMENT%"
echo "Total time steps limit for the run: $TOTAL_TIME_STEP_LIMIT"
echo "Minimum time steps threshold for a 'successful' run: $MINIMUM_TIME_STEP_THRESHOLD"
echo "ALGO switch: $ALGO_SWITCH (1: ALL; 0: default)"
echo ""



# define exit function to give end time and other counters
function exit_script {
    echo ""
    echo "=========================================================="
    echo "Exiting script at $(date)"
    echo "Current process ID: $CURRENT_PROCESS_ID"
    echo "Total number of Python calls made: $counter_python_calls"
    echo "Cumulative time spent in Python calls: $cumulative_time_in_python_calls seconds"
    echo "Cumulative number of jobs submitted so far: $cumulative_num_jobs_submitted"
    echo "Cumulative number of jobs completed so far: $cumulative_num_jobs_completed"
    echo "Total number of restart shifts: $current_num_restart_shifts"
    echo "=========================================================="
    echo ""
    exit 0
}


cumulative_num_jobs_submitted=0 # Initialize cumulative number of jobs submitted so far
cumulative_num_jobs_completed=0 # Initialize cumulative number of jobs completed so far
current_num_restart_shifts=0 # Initialize current number of restart shifts

counter_python_calls=0 # Initialize counter for Python calls
cumulative_time_in_python_calls=0 # Initialize cumulative time in Python calls

if [ "$RUN_DIRNAME" == "0" ]; then
    echo "No RUN_DIRNAME specified"

elif  [ -n "$RUN_DIRNAME" ]; then

    # check if there is $RUN_DIRNAMEa -- if not, cp RUN_DIRNAME to $RUN_DIRNAMEa
    if [ ! -d "${RUN_DIRNAME}a" ]; then
        # check how many ${RUN_DIRNAME}a, ... exist -- if just 1, copy RUN_DIRNAME to ${RUN_DIRNAME}a, else throw error
        existing_dirs=$(ls -d ${RUN_DIRNAME}*/ | wc -l)
        if [ "$existing_dirs" -gt 1 ]; then
            echo "Error: More than one run directory exists for $RUN_DIRNAME. But ${RUN_DIRNAME}a does not! Please check the directories."
            exit 1
        fi
        cp -r "$RUN_DIRNAME" "${RUN_DIRNAME}a"
        echo "${RUN_DIRNAME}a didn't exist. Copied $RUN_DIRNAME to ${RUN_DIRNAME}a"
    fi

    # check the last run directory, which should be ${RUN_DIRNAME}a, ${RUN_DIRNAME}b, ..., ${RUN_DIRNAME}z, ${RUN_DIRNAME}aa, ${RUN_DIRNAME}ab, ..., ${RUN_DIRNAME}az, ${RUN_DIRNAME}ba, ${RUN_DIRNAME}bb, ... ${RUN_DIRNAME}bz, ... ${RUN_DIRNAME}zz and run: ALGO = All in INCAR file
    # last_run_dirname=$(ls -d ${RUN_DIRNAME}*/ | sort | tail -n 1)
    # 1) ensure non-matching globs disappear instead of staying literal
    shopt -s nullglob

    # 2) init
    best_idx=0
    best_dir=""

    # 3) loop over all matching run dirs
    for dir in "${RUN_DIRNAME}"*/; do
    # strip prefix & trailing slash to get just the suffix, e.g. "a", "z", "aa", …
    suffix="${dir#${RUN_DIRNAME}}"
    suffix="${suffix%/}"

    # compute base-26 value
    idx=0
    for ((i=0; i<${#suffix}; i++)); do
        ch="${suffix:i:1}"
        # ascii('a')=97 → we want a→1, b→2, … z→26
        ascii=$(printf '%d' "'$ch")
        val=$(( ascii - 96 ))
        idx=$(( idx*26 + val ))
    done

    # keep the max
    if (( idx > best_idx )); then
        best_idx=$idx
        best_dir="$dir"
    fi
    done

    # 4) error if nothing found
    if [[ -z $best_dir ]]; then
    echo "Error: no directories found matching ${RUN_DIRNAME}*" >&2
    exit 1
    fi

    # 5) strip trailing slash and print
    last_run_dirname="${best_dir%/}"
    echo "Last run directory: $last_run_dirname"
    # echo "Last run directory: $last_run_dirname"


    # if ALGO_SWITCH is on, modify the INCAR file in ${RUN_DIRNAME}a
    if [ "$ALGO_SWITCH" -eq 1 ]; then
        # run this shell command in the new RUN_DIRNAMEa: sed -i 's/ALGO   = F/ALGO   = All/' "SCALEE_6b/INCAR" && sed -i 's/ALGO   = F/ALGO   = All/' "SCALEE_7b/INCAR"
        sed -i 's/ALGO   = F/ALGO   = All/' "$last_run_dirname/INCAR" && sed -i 's/ALGO   = F/ALGO   = All/' "$last_run_dirname/INCAR"
        echo "Modified INCAR file in $last_run_dirname for future runs to use ALGO = All if different previously."
    fi
    echo ""

    while [ $current_num_restart_shifts -lt $NUM_RESTART_SHIFTS ]; do
        echo "=========================================================="
        echo "current_num_restart_shifts: $current_num_restart_shifts"
        echo "=========================================================="
        # current_percentage_restart_shift=$INITIAL_PERCENTAGE_RESTART_SHIFT + $PERCENTAGE_RESTART_SHIFT_INCREMENT * current_num_restart_shifts
        current_percentage_restart_shift=$((INITIAL_PERCENTAGE_RESTART_SHIFT + PERCENTAGE_RESTART_SHIFT_INCREMENT * current_num_restart_shifts))

        echo ""
        cd "$home_dir" || exit 1

        counter_python_calls=$((counter_python_calls + 1))
        echo "Counter for Python calls: $counter_python_calls"
        time_start=$(date +%s)  # Start time for the Python call
        # run merge_vasp_runs.py in the RUN_DIRNAME
        python $HELP_SCRIPTS_vasp/merge_vasp_runs.py $RUN_DIRNAME
        time_end=$(date +%s)  # End time for the Python call
        time_diff=$((time_end - time_start))  # Calculate the time difference in seconds
        cumulative_time_in_python_calls=$((cumulative_time_in_python_calls + time_diff))
        echo "Time taken for the Python call: $time_diff seconds"
        echo "Cumulative time in Python calls: $cumulative_time_in_python_calls seconds"
        echo ""

        cp $HELP_SCRIPTS_vasp/RUN_VASP/RUN_VASP_MASTER_extended__SCALEE.sh $RUN_DIRNAME/
        cd $RUN_DIRNAME || exit 1
        run_dir=$(pwd)
        # if current_num_restart_shifts is 0, then create new log.RUN_VASP_MASTER_extended__SCALEE
        if [ "$current_num_restart_shifts" -eq 0 ]; then
            touch log.RUN_VASP_MASTER_extended__SCALEE
            echo "Created new log file: log.RUN_VASP_MASTER_extended__SCALEE"
        else # add 10 new empty lines
            for i in {1..10}; do echo "" >> log.RUN_VASP_MASTER_extended__SCALEE; done
        fi
        echo "Merged VASP runs in directory: $RUN_DIRNAME"
        echo "Running VASP in directory: $run_dir"
        echo "Current percentage restart shift: $current_percentage_restart_shift%"
        total_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
        echo "Total time steps that ${RUN_DIRNAME} has taken so far: $total_time_steps"
        echo ""

        # while total_time_steps is less than $TOTAL_TIME_STEP_LIMIT, continue running RUN_VASP_MASTER_extended__SCALEE.sh
        while [ "$total_time_steps" -lt $TOTAL_TIME_STEP_LIMIT ]; do
            cd "$run_dir" || exit 1
            echo ""
            echo "Total time steps is (still) less than $TOTAL_TIME_STEP_LIMIT, continuing the run..."

            rm -rf done_RUN_VASP_MASTER_extended__SCALEE  # remove the done file if it exists

            echo "Submitting RUN_VASP_MASTER_extended__SCALEE.sh with $NUM_JOBS jobs, time: $RUN_VASP_TIME hours, and nodes: $RUN_VASP_NODES; @ $(date)."
            if [ "$RESTART_MODE" -eq 0 ]; then
                echo "RESTART_MODE is 0: using the last run directory with a restart shift of $current_percentage_restart_shift% as the initial condition."
                source RUN_VASP_MASTER_extended__SCALEE.sh $NUM_JOBS $RUN_VASP_TIME $RUN_VASP_NODES $current_percentage_restart_shift $MINIMUM_TIME_STEP_THRESHOLD $RESTART_MODE >> log.RUN_VASP_MASTER_extended__SCALEE 2>&1
            elif [ "$RESTART_MODE" -gt 0 ]; then
                echo "RESTART_MODE is $RESTART_MODE: using SCALEE_${RESTART_MODE} with a restart shift of $current_percentage_restart_shift% as the initial condition."
                source RUN_VASP_MASTER_extended__SCALEE.sh $NUM_JOBS $RUN_VASP_TIME $RUN_VASP_NODES $current_percentage_restart_shift $MINIMUM_TIME_STEP_THRESHOLD $RESTART_MODE >> log.RUN_VASP_MASTER_extended__SCALEE 2>&1
            fi
            # PREVIOUS_JOB_ID=$!  # get the job ID of the last background process
            echo "JOB_ID: $PREVIOUS_JOB_ID"
            echo "Waiting for the jobs to finish..."
            echo ""
            echo ""
            echo ""

            # wait until $run_dir/done_RUN_VASP_MASTER_extended__SCALEE file exists
            while [ ! -f "$run_dir/done_RUN_VASP_MASTER_extended__SCALEE" ]; do
                sleep 60 # wait for 1 minute
            done

            # cumulative_num_jobs_submitted=NUM_JOBS+cumulative_num_jobs_submitted
            cumulative_num_jobs_submitted=$((NUM_JOBS + cumulative_num_jobs_submitted))
            echo "Total number of jobs submitted for so far: $cumulative_num_jobs_submitted"

            cumulative_num_jobs_completed=$(ls -d ${home_dir}/${RUN_DIRNAME}*/ | wc -l)
            echo "Total number of jobs completed so far: $cumulative_num_jobs_completed"

            cd $home_dir || exit 1

            # run the block below when current_num_restart_shifts is a factor of 5 or $NUM_RESTART_SHIFTS
            # if (( current_num_restart_shifts % 5 == 0 )) || (( current_num_restart_shifts == NUM_RESTART_SHIFTS )); then
            # merge all VASP runs
            echo "Merging VASP runs in directory: $RUN_DIRNAME"
            counter_python_calls=$((counter_python_calls + 1))
            time_start=$(date +%s)  # Start time for the Python call
            python $HELP_SCRIPTS_vasp/merge_vasp_runs.py $RUN_DIRNAME
            time_end=$(date +%s)  # End time for the Python call
            time_diff=$((time_end - time_start))  # Calculate the time difference in seconds
            cumulative_time_in_python_calls=$((cumulative_time_in_python_calls + time_diff))
            echo "Time taken for the Python call: $time_diff seconds"
            echo "Cumulative time in Python calls: $cumulative_time_in_python_calls seconds"
            echo ""
            # fi

            cd "$run_dir" || exit 1
            total_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
            echo "Total time steps that ${RUN_DIRNAME} has taken so far: $total_time_steps"
            cd "$home_dir" || exit 1

            # find the last run RUN_DIRNAME<letter>
            # last_run_dirname=$(ls -d ${RUN_DIRNAME}*/ | sort | tail -n 1)
            # echo "Last run directory: $last_run_dirname"
            # 1) ensure non-matching globs disappear instead of staying literal
            shopt -s nullglob

            # 2) init
            best_idx=0
            best_dir=""

            # 3) loop over all matching run dirs
            for dir in "${RUN_DIRNAME}"*/; do
            # strip prefix & trailing slash to get just the suffix, e.g. "a", "z", "aa", …
            suffix="${dir#${RUN_DIRNAME}}"
            suffix="${suffix%/}"

            # compute base-26 value
            idx=0
            for ((i=0; i<${#suffix}; i++)); do
                ch="${suffix:i:1}"
                # ascii('a')=97 → we want a→1, b→2, … z→26
                ascii=$(printf '%d' "'$ch")
                val=$(( ascii - 96 ))
                idx=$(( idx*26 + val ))
            done

            # keep the max
            if (( idx > best_idx )); then
                best_idx=$idx
                best_dir="$dir"
            fi
            done

            # 4) error if nothing found
            if [[ -z $best_dir ]]; then
            echo "Error: no directories found matching ${RUN_DIRNAME}*" >&2
            exit 1
            fi

            # 5) strip trailing slash and print
            last_run_dirname="${best_dir%/}"
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
                echo "Time steps less than $MINIMUM_TIME_STEP_THRESHOLD for ${last_run_dirname}, exiting..."
                # delete the last run directory
                cd "$home_dir" || exit 1
                echo "Deleting the last run directory: $last_run_dirname"
                # count number of $RUN_DIRNAME<letter> directories
                num_run_dirs=$(ls -d ${RUN_DIRNAME}*/ | wc -l)
                # if num_run_dirs is 1, do not delete the last run directory, else do delete it
                if [ "$num_run_dirs" -lt 3 ]; then
                    echo "Fewer than 3 run directories exist for $RUN_DIRNAME, not deleting the last one."
                else
                    echo "Deleting the last run directory: $last_run_dirname"
                    rm -rf "$last_run_dirname"
                fi
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
                    exit_script
                fi

                # exit just the loop to change CURRENT_PERCENTAGE_RESTART_SHIFT
                break
            else
                echo "Time steps is greater than or equal to $MINIMUM_TIME_STEP_THRESHOLD for the last run, continuing..."
            fi

            # if cumulative_num_jobs_submitted > CUMULATIVE_NUM_JOBS_SUBMITTED_LIMIT, exit the loop
            if [ "$cumulative_num_jobs_submitted" -gt "$CUMULATIVE_NUM_JOBS_SUBMITTED_LIMIT" ]; then
                echo "Total number of jobs submitted ($cumulative_num_jobs_submitted) has exceeded the limit ($CUMULATIVE_NUM_JOBS_SUBMITTED_LIMIT), exiting..."
                exit_script
            fi
            if [ "$cumulative_num_jobs_completed" -gt "$CUMULATIVE_NUM_JOBS_COMPLETED_LIMIT" ]; then
                echo "Total number of jobs completed ($cumulative_num_jobs_completed) has exceeded the limit ($CUMULATIVE_NUM_JOBS_COMPLETED_LIMIT), exiting..."
                exit_script
            fi
        done
    done

fi

module purge
echo ""
echo "All runs completed. Check individual log files in each run directory."
exit_script