#!/bin/bash

################################################################################################
# Summary:
# This script runs the "refine_deepmd.sh" process on all subdirectories named "recal" found in the current directory.
# It sets recalculation cutoff thresholds, executes a series of commands (using deepMD and VASP-related tools)
# to merge, extract, and analyze data, and logs the process to "log.refine".
#
# Usage:
#   ./this_script.sh [RECAL_CUTOFF_e_low] [RECAL_CUTOFF_f_low] [DP_TEST_INPUT_FILE]
#
#   - RECAL_CUTOFF_e_low: Lower cutoff for energy (default 0.005)
#   - RECAL_CUTOFF_f_low: Lower cutoff for force (default 0.3)
#   - DP_TEST_INPUT_FILE: The input file name to use for deepMD testing (default "dp_test_id_e_and_f_sweet_range")
#
# After processing all directories, the script waits for all background jobs to finish and appends a summary
# to the log file.
#
# Author: akashgpt
################################################################################################

# Set default recalculation cutoff thresholds and deepMD test input file
RECAL_CUTOFF_e_low=${1:-0.005}
RECAL_CUTOFF_f_low=${2:-0.15}
DP_TEST_INPUT_FILE=${3:-"dp_test_id_e_and_f_sweet_range"} # dp_test_id_e_and_f or dp_test_id_e_or_f
RECAL_CUTOFF_e_high=10
RECAL_CUTOFF_f_high=100

# Get the current working directory
pwd=$(pwd)

# Write header information to log.refine
echo "Running refine_deepmd.sh for ${pwd}" > log.refine
echo "" >> log.refine
echo "DP_TEST_INPUT_FILE = ${DP_TEST_INPUT_FILE}" >> log.refine
echo "RECAL_CUTOFF_e_low = ${RECAL_CUTOFF_e_low}" >> log.refine
echo "RECAL_CUTOFF_f_low = ${RECAL_CUTOFF_f_low}" >> log.refine
echo "RECAL_CUTOFF_e_high = ${RECAL_CUTOFF_e_high}" >> log.refine
echo "RECAL_CUTOFF_f_high = ${RECAL_CUTOFF_f_high}" >> log.refine
# echo "" >> log.refine

# Initialize an array to store job IDs for background processes
job_ids=()

# Save the current directory as the master directory
current_master_dir=$(pwd)   
# Find every directory named "recal" and process it
for dir in $(find . -type d -name 'recal'); do
    cd $dir
    echo "$pwd" >> log.refine
    # Run the deepMD commands:
    # - Clear previous outputs (OUTCAR and deepmd directory)
    # - Merge outputs using merge_out.py and extract data using extract_deepmd.py
    # - Execute deepMD test via apptainer and analyze using analysis_v2.py
    # - Finally, clean up the deepmd directory and extract deepmd data again with the specified input file
    l_deepmd && rm -rf OUTCAR deepmd && python $mldp/merge_out.py -o OUTCAR && python $mldp/extract_deepmd.py -d deepmd -ttr 10000 && apptainer exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m /scratch/gpfs/ag5805/qmd_data/NH3_MgSiO3/sim_data_ML_v2/setup_MLMD/latest_trained_potential/pv_comp.pb -d dp_test -n 0 > log.dp_test 2>&1 && python $mldp/model_dev/analysis_v2.py -tf . -mp dp_test -rf . -euc ${RECAL_CUTOFF_e_high} -fuc ${RECAL_CUTOFF_f_high} -flc ${RECAL_CUTOFF_f_low} -elc ${RECAL_CUTOFF_e_low} && rm -rf deepmd && python $mldp/extract_deepmd.py -f OUTCAR -d deepmd -id ${DP_TEST_INPUT_FILE} &
    
    # Grab the job ID of the last background command
    job_id=$!

    # Store the job ID in the job_ids array
    job_ids+=($job_id)

    # Return to the master directory
    cd $current_master_dir
#   l_deepmd && rm -rf OUTCAR deepmd && python $mldp/merge_out.py -o OUTCAR && python $mldp/extract_deepmd.py -d deepmd -ttr 10000 && apptainer exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m /scratch/gpfs/ag5805/qmd_data/NH3_MgSiO3/sim_data_ML_v2/setup_MLMD/latest_trained_potential/pv_comp.pb -d dp_test -n 0 > log.dp_test 2>&1 && python $mldp/model_dev/analysis.py -tf . -mp dp_test -rf . -euc 10 -fuc 100 -flc 0.8 -elc 0.02 && code vasp* && rm -rf deepmd && python $mldp/extract_deepmd.py -f OUTCAR -d deepmd -id dp_test_id_e_and_f &
done

# Wait for all background jobs to complete
for job_id in ${job_ids[@]}; do
    wait $job_id
done

# Append summary information to the log file
echo "" >> log.refine
echo "Finished refine_deepmd.sh for ${pwd} with elc and flc = ${RECAL_CUTOFF_e_low} and ${RECAL_CUTOFF_f_low}" >> log.refine
echo "" >> log.refine
echo "##################################################################################" >> log.refine
echo "" >> log.refine

##################

# Optionally, source ../count.sh to add further summary information to the log
# source ../count.sh >> log.refine