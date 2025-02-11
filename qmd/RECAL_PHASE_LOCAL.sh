#!/bin/bash

################################################################################################
# Summary:
# This script (RECAL_PHASE_LOCAL.sh) performs the recalculations phase of the MLMD workflow.
# It accepts up to three arguments:
#   1. RECAL_DIR: The directory for recalculations (default "0" indicates an invalid directory).
#   2. RECAL_MODE: The mode for recalculations (default is "VASP"; alternative is "LAMMPS").
#   3. RECAL_TEMP: The temperature for recalculations (default "0" means auto-mode for VASP).
#
# The script changes to the specified RECAL_DIR, sets up logging, and checks whether recalculations
# have already been completed. It then loads various parameters from a parameter file and, if operating
# in VASP mode, performs the following:
#   - Reads temperature information from the INCAR file if no temperature is specified.
#   - Activates the ASAP environment to run frame selection.
#   - Activates the DeepMD environment to extract and process data.
#   - Submits and monitors VASP recalculation jobs, with logic to re-run jobs if issues occur.
#
# Upon successful completion, the script logs details, cleans up temporary files, and marks the
# recalculations as done.
#
# Usage:
#   ./RECAL_PHASE_LOCAL.sh [RECAL_DIR] [RECAL_MODE] [RECAL_TEMP]
# Example:
#   ./RECAL_PHASE_LOCAL.sh my_recal_dir VASP 0
#
# Author: akashgpt
################################################################################################

RECAL_DIR=${1:-"0"} #default to invalid directory
RECAL_MODE=${2:-"VASP"} #default mode is VASP; other option is LAMMPS
RECAL_TEMP=${3:-"0"} #default temperature is "0", i.e. auto-mode (not applicable for LAMMPS)


# Check if the RECAL_DIR is ne 0
# if [ "$RECAL_DIR" -eq "0" ]; then
#     echo "No directory provided for recalculations. Exiting."
#     exit 1
# fi
cd $RECAL_DIR
current_master_dir=$(pwd)
exec > ${current_master_dir}/log.RECAL_PHASE_LOCAL 2>&1


#if recal_done exists, then exit
if [ -f recal_done ]; then
    echo "recal_done exists. Exiting."
    exit 1
fi
rm -r *index *ASAP*

# as a placeholder to indicate that the recalculations are running
touch recal_running


PARAMETER_FILE=../RECAL_PHASE_parameters.txt

MY_MLMD_SCRIPTS=$(grep -A 1 "MY_MLMD_SCRIPTS" $PARAMETER_FILE | tail -n 1)
RECAL_MODE=$(grep -A 1 "RECAL_MODE" $PARAMETER_FILE | tail -n 1)
MOL_SYSTEM=$(grep -A 1 "MOL_SYSTEM" $PARAMETER_FILE | tail -n 1)
DEEPMD_ENV=$(grep -A 1 "DEEPMD_ENV" $PARAMETER_FILE | tail -n 1)
ASAP_ENV=$(grep -A 1 "ASAP_ENV" $PARAMETER_FILE | tail -n 1)
JOB_SUBMIT_COMMAND=$(grep -A 1 "JOB_SUBMIT_COMMAND" $PARAMETER_FILE | tail -n 1)
WAIT_TIME_LONG=$(grep -A 1 "WAIT_TIME_LONG" $PARAMETER_FILE | tail -n 1)
WAIT_TIME_SHORT=$(grep -A 1 "WAIT_TIME_SHORT" $PARAMETER_FILE | tail -n 1)
NUM_RECAL_FRAMES=$(grep -A 1 "NUM_RECAL_FRAMES" $PARAMETER_FILE | tail -n 1)
MAX_RECAL_JOBS=$(grep -A 1 "MAX_RECAL_JOBS" $PARAMETER_FILE | tail -n 1)

TRAINED_POTENTIAL_DIR_temp=$(grep -A 1 "TRAINED_POTENTIAL_DIR" $PARAMETER_FILE | tail -n 1)
TRAINED_POTENTIAL_DIR=$MY_MLMD_SCRIPTS/mol_systems/$TRAINED_POTENTIAL_DIR_temp

INPUT_FILES_DIR_temp=$(grep -A 1 "INPUT_FILES_DIR" $PARAMETER_FILE | tail -n 1)
INPUT_FILES_DIR=${current_master_dir}/../input_files/${MOL_SYSTEM}/${INPUT_FILES_DIR_temp}



recal_counter_input=$current_master_dir/pre/recal/deepmd

# Function to call when exiting the script to let parent scripts know what happened
bad_exit_update() {
    echo ${current_master_dir} >> ${current_master_dir}/../recal_counter_error
    echo ${recal_counter_input} >> ${current_master_dir}/../recal_counter
    echo $current_master_dir > ${current_master_dir}/recal_done
    rm ${current_master_dir}/recal_running
}


if RECAL_MODE="VASP"; then
    if [ "$RECAL_TEMP" -eq "0" ]; then
        # Read INCAR file for TEBEG and grab the that is the third item in the line after "TEBEG"
        RECAL_TEMP_BEG=$(grep "TEBEG" INCAR | awk '{print $3}')
        RECAL_TEMP_END=$(grep "TEEND" INCAR | awk '{print $3}')

        # if TEBEG and TEEND are not the same, then exit
        if [ "$RECAL_TEMP_BEG" -ne "$RECAL_TEMP_END" ]; then
            echo "TEBEG and TEEND are not the same, and neither was any RECAL_TEMP provided. Exiting."
            exit 1
        fi

        # else
        RECAL_TEMP=${RECAL_TEMP_BEG}
    fi


    echo "Beginning the recalculation phase. Temperature for recal = ${RECAL_TEMP} K."


    ################################################################################################
    #activating modules for ASAP
    module purge
    echo $ASAP_ENV > setting_env.sh
    source setting_env.sh
    rm setting_env.sh
    # module load anaconda3/2021.5
    # conda activate deepmd_gpu
    ################################################################################################


    # running ASAP analysis to choose frames for VASP recalculation
    asap gen_desc -s 100 --fxyz OUTCAR soap -e -c 6 -n 6 -l 6 -g 0.44
    python $mldp/asap/select_frames.py -i ASAP-desc.xyz -n ${NUM_RECAL_FRAMES}

    rm -rf pre
    mkdir -p pre
    cd pre

    ################################################################################################
    #activating modules for DeepMD
    module purge
    echo $DEEPMD_ENV > setting_env.sh
    source setting_env.sh
    rm setting_env.sh
    # module load anaconda3/2021.5
    # conda activate deepmd_gpu
    ################################################################################################

    python $mldp/extract_deepmd.py -f ../OUTCAR -id ../*.index -st
    python $mldp/recal_dpdata.py -d deepmd/ -if $INPUT_FILES_DIR/${RECAL_TEMP}K/ -sc ${JOB_SUBMIT_COMMAND}
    # python $mldp/extract_deepmd.py -f ../npt.dump -fmt dump -id ../test-frame-select-fps-n-${NUM_RECAL_FRAMES}.index -st -t $RECAL_TEMP

    #not needed for VASP
    #asap>deepmd based on npt.dump seems to be leading to wrong type_map.raw and so correcting it
    # cp $INPUT_FILES_DIR/sample_type_map.raw deepmd/type_map.raw 
    
    # Submitting selected frames for VASP recalculation
    # python $mldp/recal_dpdata.py -d deepmd/ -if $INPUT_FILES_DIR/${RECAL_TEMP}K/ -sc ${JOB_SUBMIT_COMMAND}

    job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}')
    echo "VASP-recal jobs for ${RECAL_DIR} submitted at" `date `
    echo "VASP-recal jobs for ${RECAL_DIR} submitted at" `date ` >> ${current_master_dir}/../log.RECAL_PHASE_MASTER 2>&1
    # after submission, now wait as the last of the recal jobs gets finished
    while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id; do
        sleep ${WAIT_TIME_SHORT}
    done
    sleep ${WAIT_TIME_SHORT}


    # # backup check
    # # for all sub-directories, check if OUTCAR has "Total CPU time used" -- wait if not
    # # Function to check if "Total CPU time used" is in OUTCAR
    # check_outcar() {
    #     for dir in */; do
    #         if [ -f "$dir/OUTCAR" ]; then
    #             if grep -q "Total CPU time used" "$dir/OUTCAR"; then
    #                 # echo "Found 'Total CPU time used' in $dir/OUTCAR"
    #             else
    #                 # echo "'Total CPU time used' not found in $dir/OUTCAR. Waiting..."
    #                 return 1  # Return 1 to indicate that not all directories are ready
    #             fi
    #         else
    #             echo "No OUTCAR file found in $dir"
    #         fi
    #     done
    #     return 0  # Return 0 to indicate that all directories are ready
    # }

    # # Main loop
    # while true; do
    #     if check_outcar; then
    #         echo "All OUTCAR files have 'Total CPU time used'."
    #         break
    #     fi
    #     # echo "Checking again in 30 seconds..."
    #     sleep ${WAIT_TIME_LONG}
    # done



    
    echo "VASP-recal jobs for ${RECAL_DIR} done at" `date `
    # echo "VASP-recal jobs for ${RECAL_DIR} done at" `date ` >> ${current_master_dir}/../log.RECAL_PHASE_MASTER 2>&1
    echo
    echo
    echo
    echo "##################################################################"
    echo "##################################################################"
    echo "##################################################################"
    echo
    echo
    echo

    cd recal || { echo "Failed to enter recal directory. Exiting."; bad_exit_update; exit 1; }
    echo "Starting the post-recal phase of checking if it went correctly ..."

    #inside recal to rerun sims that didn't start or some issue with OUTCAR (didn't finish, converge ...)
    python ${mldp}/post_recal_rerun.py -ip all -v -ss $INPUT_FILES_DIR/sub_vasp_xtra.sh > log.recal_test 2>&1
    file="log.recal_test"
    line_count=$(wc -l < "$file")    # Count the number of lines in the file

    if [ "$line_count" -gt 14 ]; then

        echo "Problem with recal phase ($((${line_count}-14))) at $(date). Or so it seems but sleeping to see if it gets resolved."
        
        #10 times check if line_count is still > 14 and sleep for 600 seconds each time if not
        for i in {1..60}; do
            sleep ${WAIT_TIME_LONG}
            python ${mldp}/post_recal_rerun.py -ip all -v -ss $INPUT_FILES_DIR/sub_vasp_xtra.sh > log.recal_test 2>&1
            line_count=$(wc -l < "$file")    # Count the number of lines in the file
            if [ "$line_count" -le 14 ]; then
                break
            fi
        done

        if [ "$line_count" -gt 14 ]; then
            echo "Problem with recal phase persists ($((${line_count}-14))) -- rerunning now with higher NBANDS at $(date)."

            cp $INPUT_FILES_DIR/${RECAL_TEMP}K/INCAR_xtra INCAR # seeing if perhaps the issue was number of bands -- increasing it
            find . -type d | xargs -I {} cp INCAR {} # copying the new INCAR to all sub-directories
            
            echo "Doing a rerun of VASP-recal jobs for ${RECAL_DIR} submitted at" `date `
            echo "Doing a rerun of VASP-recal jobs for ${RECAL_DIR} submitted at" `date ` >> ${current_master_dir}/../log.RECAL_PHASE_MASTER 2>&1

            source rerun
            sleep ${WAIT_TIME_SHORT}
            job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}')

            # after submission, now wait as the last of the recal jobs gets finished
            while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id; do
                sleep ${WAIT_TIME_LONG}
            done

            echo "A rerun of VASP-recal jobs for ${RECAL_DIR} done at" `date `
            echo "A rerun of VASP-recal jobs for ${RECAL_DIR} done at" `date ` >> ${current_master_dir}/../log.RECAL_PHASE_MASTER 2>&1

            #10 times check if line_count is still > 14 and sleep for 600 seconds each time if not
            for i in {1..60}; do
                sleep ${WAIT_TIME_LONG}
                python ${mldp}/post_recal_rerun.py -ip all -v -ss $INPUT_FILES_DIR/sub_vasp_xtra.sh > log.recal_test 2>&1
                line_count=$(wc -l < "$file")    # Count the number of lines in the file
                if [ "$line_count" -le 14 ]; then
                    break
                fi
            done
            
            if [ "$line_count" -gt 14 ]; then
                echo "*** There is some problem ($((${line_count}-14))) with recalculations. Take a closer look! ***"
                bad_exit_update
                exit 1
            fi
        fi
    fi


    # code only goes here if the recalculations are successful, else exits ...
    echo "Recal phase: success."
    echo
    echo
    echo
    echo "##################################################################"
    echo "##################################################################"
    echo "##################################################################"
    echo
    echo
    echo
    echo "Testing the previously trained potential against recalculated frames."
    echo
    echo
    echo

    #inside recal to create new deepmd files
    python ${mldp}/merge_out.py -o OUTCAR
    python ${mldp}/extract_deepmd.py -d deepmd -ttr 10000

    # # for testing how good or bad the training potential is wrt recalculated frames
    # dp test -m ${TRAINED_POTENTIAL_DIR} -d dp_test -n ${NUM_RECAL_FRAMES}
    # python ${mldp}/model_dev/analysis.py -tf . -mp dp_test -rf . -euc 10 -fuc 10 -flc 0.4 -elc 0.02
    
    # echo
    # echo
    # echo
    # echo "##################################################################"
    # echo "##################################################################"
    # echo "##################################################################"
    # echo
    # echo
    # echo

    # # Define the input file with folder names
    # input_file="dp_test_vid_e_or_f"

    # # Check the number of lines in the input file
    # line_count=$(wc -l < "$input_file")


    # # Only proceed if the number of lines is less than 90% of NUM_RECAL_FRAMES
    # # Calculate 90% of NUM_RECAL_FRAMES
    # threshold_high=$(( NUM_RECAL_FRAMES * 90 / 100 ))
    # threshold_low=$(( NUM_RECAL_FRAMES * 5 / 100 ))

    # if [ "$line_count" -lt "$threshold_high" ]; then
    #     echo "Number of lines in dp_test_vid_e_and_f is less than ${threshold_high}. Backing up old files ..."

    #     cd ..
    #     cp -r recal old_recal

    #     ### IMPORANT !!! ###
    #     ### Can't use this right now as the temperature mode is also changed every time! So maybe we will have to figure that out first -- i.e., getting rid of composition file for tempearture mode X should not affect temperature mode Y and so on.
    #     ### IMPORANT !!! ###
    #     # if [ "$line_count" -lt "$threshold_low" ]; then

    #     #     echo "Number of lines in dp_test_vid_e_and_f is less than ${threshold_low}. Exiting and erasing this composition."
    #     #     cd ${current_master_dir}
    #     #     rm -r ${RECAL_DIR}
    #     #     echo $RECAL_DIR >> ${current_master_dir}/md_run_counter
    #         # echo $current_master_dir > ${current_master_dir}/md_done
    #     #     exit 1

    #     # else


    #         echo "Now cleaning up for a new recal+deepmd folder on the basis of this analysis."

    #         cd recal
    #         rm -r deepmd OUTCAR

    #         # Read the folder names from the file, skipping the first line
    #         folders_to_keep=$(tail -n +2 "$input_file")

    #         # Convert the list of folders to an array
    #         IFS=$'\n' read -d '' -r -a keep_array <<< "$folders_to_keep"

    #         # Get the list of all directories in the current directory
    #         all_folders=($(ls -d */))

    #         # Initialize counters
    #         delete_count=0
    #         keep_count=0

    #         # Iterate through all directories
    #         for folder in "${all_folders[@]}"; do
    #             # Remove the trailing slash
    #             folder=${folder%/}
    #             # Check if the current folder is in the list of folders to keep
    #             if [[ ! " ${keep_array[@]} " =~ " ${folder} " ]]; then
    #                 # echo "Deleting folder: $folder"
    #                 rm -rf "$folder"
    #                 ((delete_count++))
    #             else
    #                 # echo "Keeping folder: $folder"
    #                 ((keep_count++))
    #             fi
    #         done

    #         # Print the summary
    #         echo "Deleted folders: $delete_count"
    #         echo "Kept folders: $keep_count"

    #         #inside recal to create new deepmd files
    #         python $mldp/merge_out.py -o OUTCAR
    #         python $mldp/extract_deepmd.py -d deepmd -ttr 10000
        
    #         echo "New recal+deepmd folder created"

    #     # fi

    # else

    #     echo "Number of lines in dp_test_vid_e_and_f is greater than ${threshold_high}. No new recal folder will be created."

    # fi

    echo
    echo
    echo
    echo "##################################################################"
    echo "##################################################################"
    echo "##################################################################"
    echo
    echo
    echo "Done working on ${RECAL_DIR} at $(date)! Check the log file for more details."
    echo "Done working on ${RECAL_DIR} at $(date)! Check the log file for more details." >> ${current_master_dir}/../log.RECAL_PHASE_MASTER 2>&1
    echo
    echo
    echo
    echo "##################################################################"
    echo "##################################################################"
    echo "##################################################################"
fi

#append full directory address of the configuration simulated to file md_run_counter in $current_master_dir (if error raised, then append to md_run_counter_error too -- done earlier)
echo ${recal_counter_input} >> ${current_master_dir}/../recal_counter
echo $current_master_dir > ${current_master_dir}/recal_done
rm ${current_master_dir}/recal_running

exit 0