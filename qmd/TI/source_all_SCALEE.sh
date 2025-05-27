#!/bin/bash

# Find all folders and sub-folders starting with "SCALEE" and run "source data_analysis.sh" in each

# Usage: nohup $HELP_SCRIPTS_TI/source_all_SCALEE.sh > log.source_all_SCALEE 2>&1 &

# Check if the script is being run from the correct directory
current_dir=$(pwd)
parent_dir=$current_dir

echo "======================================================"
echo "Running data_analysis.sh in all SCALEE folders in $parent_dir at $(date)"
echo "======================================================"

pids=()



for dir in $(find . -type d -name "SCALEE*"); do
    echo ""
    if [ -d "$dir" ]; then
        # check if OUTCAR file exists
        if [ ! -f "$dir/OUTCAR" ]; then
            echo "No OUTCAR file found in $dir, skipping..."
            continue
        fi
        echo "Running data_analysis.sh in $dir"
        cd "$dir" 
        cp $LOCAL_HELP_SCRIPTS/qmd/vasp/data_4_analysis.sh . # from the local "rsync" copy
        # cp $parent_dir/data_4_analysis.sh .
        # nohup bash data_4_analysis.sh > log.data_4_analysis 2>&1 &
        source data_4_analysis.sh > log.data_4_analysis 2>&1 &
        pids+=($!)          # $! is the PID of the backgrounded command
        cd $parent_dir
    else
        echo "$dir is not a directory."
    fi
    echo ""
done

echo "======================================================"
echo "Waiting for all data_analysis.sh scripts to finish..."
echo "======================================================"
# Now wait only for those jobs
wait "${pids[@]}"
echo ""
echo "======================================================"
echo "All data_analysis.sh scripts have been run in all SCALEE folders in $parent_dir at $(date)"
echo "======================================================"
echo ""