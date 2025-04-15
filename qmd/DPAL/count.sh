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

process_files_v2() {
    local pattern=$1
    local frame_count
    local energy_rmse_per_atom
    local force_rmse
    local virial_rmse_per_atom
    local total_frame_count=0
    local change_search_mode=0

    # Variables to accumulate weighted sums.
    local sum_energy=0
    local sum_force=0
    local sum_virial=0

    # First pass: compute weighted sums and total frame count.
    while IFS= read -r -d '' file; do
        # Get the frame count from the line containing "number of test data" (3rd field)
        frame_count=$(grep -m 1 "number of test data" "$file" | awk '{print $9}')

        # Check if the frame count is a number or not, and change search mode if not
        change_search_mode=0
        if [ $change_search_mode -eq 0 ]; then
            if ! [[ "$frame_count" =~ ^[0-9]+$ ]]; then
                change_search_mode=1
                # echo "Error: frame_count is not a number in file $file. Changing the search string #"
            fi
        fi

        if [ $change_search_mode -eq 0 ]; then
            frame_count=$(grep -m 1 "number of test data" "$file" | awk '{print $9}')
            energy_rmse_per_atom=$(grep -m 1 "Energy RMSE/Natoms" "$file" | awk '{print $6}') # eV
            force_rmse=$(grep -m 1 "Force  RMSE" "$file" | awk '{print $6}') # eV/Angstrom
            virial_rmse_per_atom=$(grep -m 1 "Virial RMSE/Natoms" "$file" | awk '{print $6}') # eV
        else
            frame_count=$(grep -m 1 "number of test data" "$file" | awk '{print $11}')
            energy_rmse_per_atom=$(grep -m 1 "Energy RMSE/Natoms" "$file" | awk '{print $8}') # eV
            force_rmse=$(grep -m 1 "Force  RMSE" "$file" | awk '{print $8}') # eV/Angstrom
            virial_rmse_per_atom=$(grep -m 1 "Virial RMSE/Natoms" "$file" | awk '{print $8}') # eV
            # continue
        fi


        # echo "file: $file"
        # echo "$(grep -m 1 "number of test data" "$file"  | awk '{print $9}')"
        # echo "$(grep -m 1 "number of test data" "$file"  | awk '{print $11}')"
        # echo "$(grep -m 1 "Energy RMSE/Natoms" "$file")"
        # echo "$(grep -m 1 "Force  RMSE" "$file")"
        # echo "$(grep -m 1 "Virial RMSE/Natoms" "$file")"
        # echo "frame_count: $frame_count"
        # echo "energy_rmse_per_atom: $energy_rmse_per_atom"
        # echo "force_rmse: $force_rmse"
        # echo "virial_rmse_per_atom: $virial_rmse_per_atom"
        # echo ""
        # continue

        energy_rmse_per_atom=$(printf "%.10f" "$energy_rmse_per_atom")
        force_rmse=$(printf "%.10f" "$force_rmse")
        virial_rmse_per_atom=$(printf "%.10f" "$virial_rmse_per_atom")

        # Accumulate weighted sums.
        total_frame_count=$(( total_frame_count + frame_count ))
        sum_energy=$(echo "scale=8; $sum_energy + ($frame_count * $energy_rmse_per_atom)" | bc)
        sum_force=$(echo "scale=8; $sum_force + ($frame_count * $force_rmse)" | bc)
        sum_virial=$(echo "scale=8; $sum_virial + ($frame_count * $virial_rmse_per_atom)" | bc)
    done < <(find . -type f -name "$pattern" -print0)

    # echo ""
    # echo "### \"$pattern\" ###"
    # echo "frame_count: $frame_count"
    # echo "energy_rmse_per_atom: $energy_rmse_per_atom"
    # echo "force_rmse: $force_rmse"
    # echo "virial_rmse_per_atom: $virial_rmse_per_atom"
    # echo "total_frame_count: $total_frame_count"
    # echo "sum_energy: $sum_energy"
    # echo "sum_force: $sum_force"
    # echo "sum_virial: $sum_virial"

    # Calculate weighted averages.
    local energy_rmse_per_atom_avg
    local force_rmse_avg
    local virial_rmse_per_atom_avg
    energy_rmse_per_atom_avg=$(echo "scale=8; $sum_energy / $total_frame_count" | bc -l)
    force_rmse_avg=$(echo "scale=8; $sum_force / $total_frame_count" | bc -l)
    virial_rmse_per_atom_avg=$(echo "scale=8; $sum_virial / $total_frame_count" | bc -l)

    # echo "energy_rmse_per_atom_avg: $energy_rmse_per_atom_avg"
    # echo "force_rmse_avg: $force_rmse_avg"
    # echo "virial_rmse_per_atom_avg: $virial_rmse_per_atom_avg"

    # Initialize variables for variance accumulation.
    local energy_variance_sum=0
    local force_variance_sum=0
    local virial_variance_sum=0

    # echo "##############################################################"
    # echo "Second pass:"
    # echo "##############################################################"
    # Second pass: compute weighted squared differences.
    while IFS= read -r -d '' file; do
        # Get the frame count from the line containing "number of test data" (3rd field)
        frame_count=$(grep -m 1 "number of test data" "$file" | awk '{print $9}')
        # Check if the frame count is a number or not, and change search mode if not
        change_search_mode=0
        if [ $change_search_mode -eq 0 ]; then
            if ! [[ "$frame_count" =~ ^[0-9]+$ ]]; then
                change_search_mode=1
                # echo "Error: frame_count is not a number in file $file. Changing the search string #"
            fi
        fi

        # if change_search_mode=0, else
        if [ $change_search_mode -eq 0 ]; then
            frame_count=$(grep -m 1 "number of test data" "$file" | awk '{print $9}')
            energy_rmse_per_atom=$(grep -m 1 "Energy RMSE/Natoms" "$file" | awk '{print $6}')
            force_rmse=$(grep -m 1 "Force  RMSE" "$file" | awk '{print $6}')
            virial_rmse_per_atom=$(grep -m 1 "Virial RMSE/Natoms" "$file" | awk '{print $6}')
        else # change_search_mode=1
            frame_count=$(grep -m 1 "number of test data" "$file" | awk '{print $11}')
            energy_rmse_per_atom=$(grep -m 1 "Energy RMSE/Natoms" "$file" | awk '{print $8}')
            force_rmse=$(grep -m 1 "Force  RMSE" "$file" | awk '{print $8}')
            virial_rmse_per_atom=$(grep -m 1 "Virial RMSE/Natoms" "$file" | awk '{print $8}')
        fi

        # echo "file: $file"
        # echo "frame_count: $frame_count"
        # echo "energy_rmse_per_atom: $energy_rmse_per_atom"
        # echo "force_rmse: $force_rmse"
        # echo "virial_rmse_per_atom: $virial_rmse_per_atom"
        # echo ""

        energy_rmse_per_atom=$(printf "%.10f" "$energy_rmse_per_atom")
        force_rmse=$(printf "%.10f" "$force_rmse")
        virial_rmse_per_atom=$(printf "%.10f" "$virial_rmse_per_atom")

        # Compute differences
        local diff_energy diff_force diff_virial
        diff_energy=$(echo "scale=8; $energy_rmse_per_atom - $energy_rmse_per_atom_avg" | bc -l)
        diff_force=$(echo "scale=8; $force_rmse - $force_rmse_avg" | bc -l)
        diff_virial=$(echo "scale=8; $virial_rmse_per_atom - $virial_rmse_per_atom_avg" | bc -l)

        # Accumulate weighted squared differences.
        energy_variance_sum=$(echo "scale=8; $energy_variance_sum + $frame_count * ($diff_energy * $diff_energy)" | bc -l)
        force_variance_sum=$(echo "scale=8; $force_variance_sum + $frame_count * ($diff_force * $diff_force)" | bc -l)
        virial_variance_sum=$(echo "scale=8; $virial_variance_sum + $frame_count * ($diff_virial * $diff_virial)" | bc -l)
    done < <(find . -type f -name "$pattern" -print0)

    # echo "energy_variance_sum: $energy_variance_sum"
    # echo "force_variance_sum: $force_variance_sum"
    # echo "virial_variance_sum: $virial_variance_sum"

    # Compute weighted standard deviations.
    local energy_rmse_per_atom_std
    local force_rmse_std
    local virial_rmse_per_atom_std
    energy_rmse_per_atom_std=$(echo "scale=8; sqrt($energy_variance_sum / $total_frame_count)" | bc -l)
    force_rmse_std=$(echo "scale=8; sqrt($force_variance_sum / $total_frame_count)" | bc -l)
    virial_rmse_per_atom_std=$(echo "scale=8; sqrt($virial_variance_sum / $total_frame_count)" | bc -l)

    # multiple by 1000 to convert to meV
    energy_rmse_per_atom_avg=$(echo "scale=8; $energy_rmse_per_atom_avg * 1000" | bc -l)
    energy_rmse_per_atom_std=$(echo "scale=8; $energy_rmse_per_atom_std * 1000" | bc -l)

    # Output results.
    echo ""
    echo "### \"$pattern\" ###"
    echo "Total number of test data frames: $total_frame_count"
    # echo "Average Energy RMSE/Natoms: $energy_rmse_per_atom_avg +/- $energy_rmse_per_atom_std eV"
    # echo "Average Force RMSE: $force_rmse_avg +/- $force_rmse_std eV/A"
    # echo "Average Virial RMSE/Natoms: $virial_rmse_per_atom_avg +/- $virial_rmse_per_atom_std eV"
    # # print only 2 decimal places
    echo "Average Energy RMSE/Natoms: $(printf "%.4f" $energy_rmse_per_atom_avg) +/- $(printf "%.4f" $energy_rmse_per_atom_std) meV"
    echo "Average Force RMSE: $(printf "%.4f" $force_rmse_avg) +/- $(printf "%.4f" $force_rmse_std) eV/A"
    echo "Average Virial RMSE/Natoms: $(printf "%.4f" $virial_rmse_per_atom_avg) +/- $(printf "%.4f" $virial_rmse_per_atom_std) eV"
    echo ""
}


# Process both file patterns
process_files "dp_test_id_e_and_f"
process_files "dp_test_id_e_or_f"
process_files "dp_test_id_e_and_f_optimum_range"
process_files "dp_test_id_e_or_f_optimum_range"

process_files_v2 "log.dp_test"