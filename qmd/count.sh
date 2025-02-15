#!/bin/bash

####################################################################################################
# Summary:
# This script processes files matching specified patterns and computes summary statistics for each set.
# For each file matching the given pattern, it:
#   - Reads the number of lines (subtracting one from the count),
#   - Accumulates the total lines and the square of the line counts,
#   - Determines the file with the maximum and minimum selected frames,
#   - Computes the total frame count by counting directories (in the file's home directory)
#     whose names start with a digit.
#
# It then calculates:
#   - The average_frame_selected frames per simulation,
#   - The standard deviation of selected frames across simulations,
#   - The average frame count per simulation.
#
# The script defines a function 'process_files' to perform these computations for a given file name pattern.
# Finally, it calls this function on four different patterns:
#   - "dp_test_id_e_and_f"
#   - "dp_test_id_e_or_f"
#   - "dp_test_id_e_and_f_optimum_range"
#   - "dp_test_id_e_or_f_optimum_range"
#
# Usage:
#   Simply run the script: ./this_script.sh
#
# Author: akashgpt
####################################################################################################

process_files() {
    local pattern=$1
    local total_lines=0
    local total_lines_squared=0
    local total_frame_count=0
    local file_count=0
    local max_lines=0
    local max_lines_file=""
    local min_lines=-1
    local min_lines_file=""

    # Process each file found by the given pattern
    while IFS= read -r -d '' file; do
        lines=$(wc -l < "$file")
        ((lines--))  # Subtract 1
        if [ "$lines" -ge 0 ]; then
            total_lines=$((total_lines + lines))
            total_lines_squared=$((total_lines_squared + lines * lines))
            ((file_count++))
            # Update maximum
            if [ "$lines" -gt "$max_lines" ]; then
                max_lines=$lines
                max_lines_file=$file
            fi
            # Update minimum (initialize if min_lines == -1)
            if [ "$min_lines" -eq -1 ] || [ "$lines" -lt "$min_lines" ]; then
                min_lines=$lines
                min_lines_file=$file
            fi
        fi
        # total number of folders that start with a digit, in the directory with the pattern
        # home directory of "file"
        home_dir=$(dirname "$file")
        frame_count=$(find "$home_dir" -maxdepth 1 -type d -name '[0-9]*' | wc -l)
        total_frame_count=$((total_frame_count + frame_count))
        
    done < <(find . -type f -name "$pattern" -print0)

    # Calculate average using bc for floating-point division
    if [ "$file_count" -gt 0 ]; then
        average_frame_selected=$(echo "scale=2; $total_lines / $file_count" | bc)

        # Calculate standard deviation
        variance=$(echo "scale=2; ($total_lines_squared - $total_lines * $average_frame_selected) / $file_count" | bc)
        standard_deviation=$(echo "scale=2; sqrt($variance)" | bc)

        average_frame_count=$(echo "scale=2; $total_frame_count / $file_count" | bc)

        fraction_frame_selected=$(echo "scale=2; $total_lines / $total_frame_count" | bc)
        std_fraction_frame_selected=$(echo "scale=2; $standard_deviation*$file_count / $total_frame_count" | bc)

    else
        average_frame_selected=0
        standard_deviation=0
        average_frame_count=0
        fraction_frame_selected=0
        std_fraction_frame_selected=0
    fi

    echo ""
    echo "### \"$pattern\" ###"
    echo "Total sims: $file_count"
    echo "Total frame count: $total_frame_count"
    echo "Total selected frames: $total_lines"
    echo "Average frame count per sim: $average_frame_count"
    echo "Average frames per sim: $average_frame_selected +/- $standard_deviation"
    echo "Fraction of frames selected: $fraction_frame_selected +/- $std_fraction_frame_selected"
    echo "Maximum number of selected frames: $max_lines"
    echo "Minimum number of selected frames: $min_lines"
    echo "Sim with maximum selected frames: $max_lines_file"
    echo "Sim with minimum selected frames: $min_lines_file"
    echo ""
}

# Process both file patterns
process_files "dp_test_id_e_and_f"
process_files "dp_test_id_e_or_f"
process_files "dp_test_id_e_and_f_optimum_range"
process_files "dp_test_id_e_or_f_optimum_range"