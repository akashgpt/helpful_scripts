#!/bin/bash

#############################################################################################################
# Summary:
#   This script is used to update the "recal" directories in the current directory.
#   It navigates to each "recal" directory and performs the following steps:
#   1. If the "done_recal_update" file exists, skip the current directory.
#   2. Remove the "deepmd" directory if it exists.
#   3. If the searched directory is "old_recal", remove the "recal" directory and rename "old_recal" to "recal".
#   4. Check if the path contains "v5_iX" where X is a number larger than 20.
#       If true, run the additional analysis scripts and print the number of lines in DP_TEST_INPUT_FILE (excluding the first line).
#   5. Run the extract_deepmd.py script.
#   6. Mark the directory as processed by creating the "done_recal_update" file.
#
# Usage: 
#   source recal_update.sh
#
# Author: akashgpt
#############################################################################################################



# Define paths to scripts and files
MLDP_SCRIPTS="/projects/BURROWS/akashgpt/misc_libraries/scripts_Jie/mldp"
DP_TEST_INPUT_FILE="dp_test_id_e_or_f"
dir_to_be_searched="old_recal"

parent_dir=$(pwd)

# Find all directories named "recal" in the current directory
find . -type d -name "$dir_to_be_searched" | while read -r recal_dir; do
    # Navigate to each "recal" directory or skip if unable
    cd "$recal_dir" || { echo "Error navigating to $recal_dir"; continue; }
    echo "Navigated to $recal_dir"

    # If the "done_recal_update" file exists, skip the current directory
    if [ -f done_recal_update ]; then
        echo "done_recal_update exists in $recal_dir"
        cd "$parent_dir" || exit 1
        continue
    fi

    # Remove the "deepmd" directory if it exists
    if [ -d "deepmd" ]; then
        rm -r deepmd
        echo "Removed deepmd"
    fi

    # If the searched directory is "old_recal", perform additional steps
    if [ "$dir_to_be_searched" == "old_recal" ]; then
        echo "$dir_to_be_searched is old_recal"
        cd .. || continue
        rm -r recal
        echo "Removed recal in $(pwd)"
        mv old_recal recal
        cd recal || continue
    fi

    # Check if the path contains "v5_iX" where X is a number larger than 30
    folder_name=$(pwd)
    if [[ $folder_name =~ v5_i([0-9]+) ]]; then
        X=${BASH_REMATCH[1]}
        if (( X > 20 )); then
            # Run the additional analysis scripts if X > 30
            python "${MLDP_SCRIPTS}/model_dev/analysis.py" -tf . -mp dp_test -rf . -euc 10 -fuc 100 -flc 0.15 -elc 0.005
            python "${MLDP_SCRIPTS}/plot_MLMD_vs_DFT_test.py"
            echo "Ran analysis.py and plot_MLMD_vs_DFT_test.py in $recal_dir with X = $X"

            # Count lines in DP_TEST_INPUT_FILE, excluding the first line
            line_count=$(tail -n +2 "$DP_TEST_INPUT_FILE" | wc -l)
            echo "Number of lines in $DP_TEST_INPUT_FILE (excluding first line): $line_count"

            # Count total folders in the current directory
            total_folders=$(find . -maxdepth 1 -type d | wc -l)
            echo "Total folders in $recal_dir: $((total_folders - 1))" # Exclude '.' directory
        fi
    fi

    # Run the extract_deepmd.py script
    python "${MLDP_SCRIPTS}/extract_deepmd.py" -f OUTCAR -id "$DP_TEST_INPUT_FILE"
    echo "Ran extract_deepmd.py"

    echo ""
    echo ""

    # Mark the directory as processed
    touch done_recal_update

    # Return to the starting directory
    cd "$parent_dir" || { echo "Error navigating to $parent_dir"; exit 1; }
done
