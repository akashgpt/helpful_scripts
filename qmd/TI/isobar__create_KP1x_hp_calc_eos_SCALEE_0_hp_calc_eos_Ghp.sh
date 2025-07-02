#!/bin/bash

# Usage:  source $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc_eos_Ghp.sh > log.isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc_eos_Ghp 2>&1 &
#         nohup bash $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc_eos_Ghp.sh > log.isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc_eos_Ghp 2>&1 &
# This script will run the Ghp_analysis.py script in all directories that contain a SCALEE_1 directory.
# It will run the script in the background and log the output to log.Ghp_analysis.
# It will also check for errors in the log files and print a message if any errors are found.

# Save the current working directory for later
ISOBAR_CALC_dir=$(pwd)
CONFIG_dir=$(dirname "$ISOBAR_CALC_dir") # CONFIG_dir is the parent directory of isobar_calc
PT_dir=$(dirname "$CONFIG_dir") # PT_dir is the parent directory of CONFIG_dir
PT_dir_name=$(basename "$PT_dir")

COMPOSITION_dir=$(dirname "$PT_dir")
COMPOSITION_dir_name=$(basename "$COMPOSITION_dir")

echo "Current time: $(date)"
echo "PT directory: $PT_dir"
echo "PT directory name: $PT_dir_name"
echo "COMPOSITION directory: $COMPOSITION_dir"
echo "COMPOSITION directory name: $COMPOSITION_dir_name"
echo "CONFIG directory: $CONFIG_dir"
echo "ISOBAR_CALC_dir: $ISOBAR_CALC_dir"
echo ""
echo ""
echo ""

module purge
module load anaconda3/2024.6; conda activate ase_env


# check if there exist 4 peavg.out files and if (grep time */SCALEE_0/analysis/peavg.out | awk '{print $5}') results in 4 numbers, if not skip and give error message
time_array=()
for dir in */SCALEE_0/analysis/; do
    if [ -f "$dir/peavg.out" ]; then
        time_value=$(grep "time" "$dir/peavg.out" | awk '{print $5}')
        if [[ -n "$time_value" ]]; then
            time_array+=("$time_value")
        fi
    fi
done
if [ ${#time_array[@]} -ne 4 ]; then
    echo "Error: Not all directories contain a valid peavg.out file or the time values are not found."
    echo "Found ${#time_array[@]} valid time values instead of 4. Exiting the script."
    exit 1
else
    echo "Found 4 valid time values: ${time_array[*]}. Continuing with the script."
fi

# for all directories in the current directory, go in and run "nohup python $HELP_SCRIPTS_TI/Ghp_analysis.py > log.Ghp_analysis 2>&1 &"
# check if $HELP_SCRIPTS_TI/isobar_Ghp_analysis.py exists
if [ -f "$HELP_SCRIPTS_TI/isobar_Ghp_analysis.py" ]; then
    nohup python $HELP_SCRIPTS_TI/isobar_Ghp_analysis.py > log.isobar_Ghp_analysis 2>&1 &
else # use LOCAL_...
    nohup python $LOCAL_HELP_SCRIPTS_TI/isobar_Ghp_analysis.py > log.isobar_Ghp_analysis 2>&1 &
fi

echo "Running isobar_Ghp_analysis.py in $ISOBAR_CALC_dir"

module purge 

echo ""
echo ""

# wait for all background processes to finish
wait
# # check if there are any errors in the log files
# for dir in */; do
#     if [ -d "$dir/SCALEE_1" ]; then
#         echo "Checking for errors in $dir"
#         if grep -q "ERROR" "$dir/log.Ghp_analysis"; then
#             echo "*** 'ERROR' found ***"
#         else
#             echo "No 'ERROR' found"
#         fi
#         # check "Traceback"
#         if grep -q "Traceback" "$dir/log.Ghp_analysis"; then
#             echo "*** 'Traceback' error found ***"
#         else
#             echo "No 'Traceback' error found"
#         fi
#     fi
#     echo ""
# done

echo ""
echo ""
echo "All directories have been evaluated."