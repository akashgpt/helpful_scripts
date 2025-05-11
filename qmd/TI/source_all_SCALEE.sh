#!/bin/bash

# Find all folders and sub-folders starting with "SCALEE" and run "source data_analysis.sh" in each

# Check if the script is being run from the correct directory
current_dir=$(pwd)
parent_dir=$current_dir

echo "======================================================"
echo "Running data_analysis.sh in all SCALEE folders in $parent_dir."
echo "======================================================"

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
        cp /projects/BURROWS/akashgpt/run_scripts/helpful_scripts/qmd/vasp/data_4_analysis.sh .
        source data_4_analysis.sh > /dev/null 2>&1 &
        cd $parent_dir
    else
        echo "$dir is not a directory."
    fi
    echo ""
done