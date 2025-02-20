#!/bin/bash

# ========================================================================================
# Script: Track Progress in Directories Matching PREFIX (Sorted Numerically)
# Author: [Your Name] (Optional)
#
# Summary:
# - Iterates through directories starting with a given prefix (default: "v8").
# - Processes only those directories that contain "done_iteration" file.
# - Runs $HELP_SCRIPTS/qmd/count.sh in each directory.
# - Logs output to `log.count` and extracts relevant details into `log.track_progress`.
# - Ensures directories are sorted in **true numerical order**.
#
# Usage:
# - Run this script from the parent directory.
# - Example: `./track_progress.sh v8`
# ========================================================================================

PREFIX=${1:-v8}

# Get the current directory
parent_dir=$(pwd)
echo "Starting track_progress.sh in $parent_dir"

# Initialize log file
echo "#################" > log.track_progress
echo "Tracking progress" >> log.track_progress
echo "#################" >> log.track_progress
echo "" >> log.track_progress

# Get directories matching $PREFIX* and sort them numerically
dirs=($(ls -d ${PREFIX}* 2>/dev/null | sort -V))

# Process each directory in sorted order
for dir in "${dirs[@]}"; do
    if [ -f "$dir/done_iteration" ]; then
        echo "Processing $dir"
        cd "$dir" || { echo "Failed to enter $dir"; continue; }

        # Initialize log.count
        echo "====================" > log.count
        echo "$dir" >> log.count
        echo "====================" >> log.count

        echo "====================" >> "$parent_dir/log.track_progress"
        echo "$dir" >> "$parent_dir/log.track_progress"
        echo "====================" >> "$parent_dir/log.track_progress"

        # Execute count.sh and append output to log.count
        source "$HELP_SCRIPTS/qmd/count.sh" >> log.count

        # Extract "Fraction of frames selected" from log.count and append to log.track_progress
        for i in {1..4}; do
            ffsel=$(grep "Fraction of frames selected" log.count | sed -n "${i}p" | awk '{for(j=5;j<=NF;j++) printf "%s ", $j; print ""}')
            case $i in
                1) label="dp_test_id_e_and_f:" ;;
                2) label="dp_test_id_e_or_f:" ;;
                3) label="dp_test_id_e_and_f_optimum_range:" ;;
                4) label="dp_test_id_e_or_f_optimum_range:" ;;
            esac
            echo "$label $ffsel" >> "$parent_dir/log.track_progress"
        done

        echo "" >> "$parent_dir/log.track_progress"

        cd "$parent_dir" || exit
    fi
done

echo "Done!"