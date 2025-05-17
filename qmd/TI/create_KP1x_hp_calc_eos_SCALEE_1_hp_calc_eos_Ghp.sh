#!/bin/bash

# Usage:  source $HELP_SCRIPTS_TI/create_KP1x_hp_calc_eos_SCALEE_1_hp_calc_eos_Ghp.sh > log.create_KP1x_hp_calc_eos_SCALEE_1_hp_calc_eos_Ghp 2>&1 &
# This script will run the GhP_analysis.py script in all directories that contain a SCALEE_1 directory.
# It will run the script in the background and log the output to log.GhP_analysis.
# It will also check for errors in the log files and print a message if any errors are found.

current_dir=$(pwd)
parent_dir=$(dirname "$current_dir")

echo "Parent directory: $current_dir"
echo ""
echo ""

module purge
module load anaconda3/2024.6; conda activate ase_env
# for all directories in the current directory, go in and run "nohup python $HELP_SCRIPTS_TI/GhP_analysis.py > log.GhP_analysis 2>&1 &"
for dir in */; do
    # if directory contains SCALEE_1 directory
    if [ -d "$dir/SCALEE_1" ]; then
        echo "Evaluating $dir"
        cd "$dir" || exit
        # run the command
        nohup python $HELP_SCRIPTS_TI/GhP_analysis.py > log.GhP_analysis 2>&1 &
        cd "$current_dir" || exit
    fi
done
module purge 

echo ""
echo ""

# wait for all background processes to finish
wait
# check if there are any errors in the log files
for dir in */; do
    if [ -d "$dir/SCALEE_1" ]; then
        echo "Checking for errors in $dir"
        if grep -q "ERROR" "$dir/log.GhP_analysis"; then
            echo "*** 'ERROR' found ***"
        else
            echo "No 'ERROR' found"
        fi
        # check "Traceback"
        if grep -q "Traceback" "$dir/log.GhP_analysis"; then
            echo "*** 'Traceback' error found ***"
        else
            echo "No 'Traceback' error found"
        fi
    fi
    echo ""
done

echo ""
echo ""
echo "All directories have been evaluated."