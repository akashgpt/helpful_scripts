#!/bin/bash

# run "nohup $HELP_SCRIPTS_TI/tar_KP1.sh > log.tar_KP1 2>&1 &" in all folders V_est whose parent's parent folder is not isobar_calc
# Usage: nohup $HELP_SCRIPTS_TI/tar_KP1_LEVEL0.sh > log.tar_KP1_LEVEL0 2>&1 &


home_dir=$(pwd)
counter=0
echo "Current time: $(date)"
echo "Home directory: $home_dir"
module purge
module load anaconda3/2024.6; conda activate hpc-tools
echo "======================================================="
echo ""
echo ""

for dir in $(find . -type d -name "V_est" -not -path "*/isobar_calc/*"); do
    
    # if PX_TY on path, skip this directory
    if [[ "$dir" == *"PX_TY"* ]]; then
        # echo "Skipping directory: $dir (contains PX_TY)"
        continue
    fi
    if [ -d "$dir" ]; then
        cd "$dir"
        echo "Running in directory: $(pwd)"
        counter=$((counter + 1))
        # echo "Counter: $counter"
        nohup $HELP_SCRIPTS_TI/tar_KP1.sh > log.tar_KP1 2>&1 &
        cd "$home_dir"

    else
        echo "Directory $dir does not exist."
    fi
    echo ""
done


echo ""
echo "======================================================"
module purge
echo "All tasks started in V_est directories (where isobar_calc is not in the path)."
echo "Total directories processed: $counter"
echo "Current time: $(date)"