#!/bin/bash

# set -euo pipefail

# Usage: 
# (for new run): source $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc.sh > log.isobar__create_KP1x_hp_calc 2>&1 &
#                nohup $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc.sh > log.isobar__create_KP1x_hp_calc 2>&1 &
# (for rerun with same parameters): source $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc.sh 1 > log.isobar__create_KP1x_hp_calc 2>&1 &
#                                   nohup $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc.sh 1 > log.isobar__create_KP1x_hp_calc 2>&1 &
# Author: Akash Gupta



rerun=${1:-0}  # Default to 0 if not provided; 1 for a rerun with same parameters


# echo process ID
echo "Process ID: $$"
echo ""

echo "Rerun flag: $rerun"

# if rerun is 1, announce it
if [[ $rerun -eq 1 ]]; then
    echo "Rerunning with the same parameters -- only those that did not finish. W updated INCAR_SPC or something to make sure convergence this time ?"
else
    echo "Running for the first time..."
fi



# Scaling parameter for alchemical transformations
SCALEE_CHOSEN=1.0            # Lambda scaling factor
MLDP_SCRIPTS="/projects/BURROWS/akashgpt/misc_libraries/scripts_Jie/mldp"


# Physical constants
kB=0.00008617333262145       # Boltzmann constant (eV/K)
# Compute sigma = k_B * T (for smearing)
# SIGMA_CHOSEN=$(echo "${kB} * ${TEMP_CHOSEN}" | bc -l)

#-------------------------
# Working directory
#-------------------------
ISOBAR_CALC_dir=$(pwd)
CONFIG_dir=$(dirname "$ISOBAR_CALC_dir") # CONFIG_dir is the parent directory of isobar_calc
PT_dir=$(dirname "$CONFIG_dir") # PT_dir is the parent directory of CONFIG_dir
PT_dir_name=$(basename "$PT_dir")

# parent directory
COMPOSITION_dir=$(dirname "$PT_dir")
COMPOSITION_dir_name=$(basename "$COMPOSITION_dir")

echo "Current time: $(date)"
echo "PT directory: $PT_dir"
echo "PT directory name: $PT_dir_name"
echo "COMPOSITION directory: $COMPOSITION_dir"
echo "COMPOSITION directory name: $COMPOSITION_dir_name"
echo "ISOBAR_CALC_dir: $ISOBAR_CALC_dir"
echo ""



SETUP_dir=$PT_dir/master_setup_TI
# LOCAL_SETUP_dir=$CONFIG_dir/setup_TI


# read all the above from input.calculate_GFE file where each line is a key-value pair, e.g. TEMP_CHOSEN=13000
PARAMETER_FILE=${SETUP_dir}/input.calculate_GFE
if [ -f "$PARAMETER_FILE" ]; then
    while IFS='=' read -r key value; do
        case $key in
            TEMP_CHOSEN) TEMP_CHOSEN="$value" ;;
            PSTRESS_CHOSEN_GPa) PSTRESS_CHOSEN_GPa="$value" ;;
            NPAR_CHOSEN) NPAR_CHOSEN="$value" ;;
            POTIM_CHOSEN) POTIM_CHOSEN="$value" ;;
            NBANDS_CHOSEN) NBANDS_CHOSEN="$value" ;;
            KPAR_CHOSEN_111) KPAR_CHOSEN_111="$value" ;;
            KPAR_CHOSEN_222) KPAR_CHOSEN_222="$value" ;;
            WAIT_TIME_VLONG) WAIT_TIME_VLONG="$value" ;;
            WAIT_TIME_LONG) WAIT_TIME_LONG="$value" ;;
            WAIT_TIME_SHORT) WAIT_TIME_SHORT="$value" ;;
            N_FRAMES_hp_calculations) N_FRAMES_hp_calculations="$value" ;;
        esac
    done < "$PARAMETER_FILE"
else
    echo "Parameter file not found: $PARAMETER_FILE"
    exit 1
fi


# if TEMP_CHOSEN and PSTRESS_CHOSEN_GPa are not a number, exit
if ! [[ "$TEMP_CHOSEN" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "TEMP_CHOSEN is not a number: $TEMP_CHOSEN. Exiting. Please check the input.calculate_GFE file."
    exit 1
fi
if ! [[ "$PSTRESS_CHOSEN_GPa" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "PSTRESS_CHOSEN_GPa is not a number: $PSTRESS_CHOSEN_GPa. Exiting. Please check the input.calculate_GFE file."
    exit 1
fi

# Extract TEMP_CHOSEN_ARRAY from the names of the directories in the current directory -- their format is "T<TEMP_CHOSEN_i>"
TEMP_CHOSEN_ARRAY=($(ls -d T* | sed 's/T//g' | sort -n))
echo "TEMP_CHOSEN_ARRAY: ${TEMP_CHOSEN_ARRAY[@]}"

SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN" | bc -l)  # Gaussian smearing sigma

PSTRESS_CHOSEN=$(echo "$PSTRESS_CHOSEN_GPa * 10" | bc -l)  # Convert GPa to kBar

########################################
NPAR_CHOSEN_SPC=7 # For single point calculations, set NPAR_CHOSEN to 2; KPAR is 4 for hp_calculations
########################################


#-------------------------
# Print the parameters
echo "------------------------"
echo "Simulation parameters:"
echo "TEMP_CHOSEN_ARRAY: ${TEMP_CHOSEN_ARRAY[@]}"
echo "PSTRESS_CHOSEN_GPa: $PSTRESS_CHOSEN_GPa"
echo "PSTRESS_CHOSEN (kBar): $PSTRESS_CHOSEN"
echo "NPAR_CHOSEN: $NPAR_CHOSEN"
echo "POTIM_CHOSEN: $POTIM_CHOSEN"
echo "NBANDS_CHOSEN: $NBANDS_CHOSEN"
echo "KPAR_CHOSEN_111: $KPAR_CHOSEN_111"
echo "KPAR_CHOSEN_222: $KPAR_CHOSEN_222"
echo "WAIT_TIME_VLONG: $WAIT_TIME_VLONG"
echo "WAIT_TIME_LONG: $WAIT_TIME_LONG"
echo "WAIT_TIME_SHORT: $WAIT_TIME_SHORT"
echo "N_FRAMES_hp_calculations: $N_FRAMES_hp_calculations"
echo "SCALEE_CHOSEN: $SCALEE_CHOSEN"
echo "MLDP_SCRIPTS: $MLDP_SCRIPTS"
echo "-------------------------"
echo ""



# Check if the folder name contains both TEMP_CHOSEN and PSTRESS_CHOSEN_GPa
if [[ "$PT_dir_name" == *"${TEMP_CHOSEN}"* && "$PT_dir_name" == *"${PSTRESS_CHOSEN_GPa}"* ]]; then
    echo "Folder name contains TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
else
    echo "Folder name does NOT contain both TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
    exit 1
fi

counter_incomplete_runs=0


## Check if KP1a, KP1b, KP1c, etc. directories span the PSTRESS_CHOSEN_GPa or not

# create array of P_RUN values
P_RUN_array=()
counter_P_not_spanned=0

#--------------------------------------------------------------
# Loop over every directory named exactly 'KP1'
#--------------------------------------------------------------
while IFS= read -r -d '' parent; do
    echo "#----------------------------------------"
    echo "#----------------------------------------"
    echo ""
    echo "Processing parent directory: ${parent}"
    echo ""
    echo "#----------------------------------------"
    echo "#----------------------------------------"

    # Iterate over immediate subdirectories beginning with 'KP1'
    for child in "${parent}"/KP1*; do
        if [ -d "${child}" ]; then
            # Enter the child directory
            cd "${child}" || exit # KP1a, KP1b, etc.
            KP1x_dir=$(pwd)
            echo " → Entered ${KP1x_dir}"



            if [[ $rerun -eq 0 ]]; then # only in this case checking if KP1x sims completed or not
                # echo "###########################"
                # echo 
                # echo "###########################"
                # echo "UNCOMMENT THE FOLLOWING LINES IN THE CODE!!!"
                # echo "###########################"
                echo ""
                echo "###########################"
                # 0) file log.create_KP1 has numbers next to "JOB_ID_KP1:" -- grab all those numbers and check if they are running
                #    if they are running, sleep and wait for them to finish
                #    if they are not running, continue to the next step
                #    if there are no numbers, prompt error and exit
                # 0.1) Extract all tokens after "JOB_ID_KP1:" 
                mapfile -t job_ids < <(grep -oP 'JOB_ID_KP1x:\s*\K\S+' $ISOBAR_CALC_dir/log.isobar__create_KP1x)

                # # 0.2) Error if none found
                if [ ${#job_ids[@]} -eq 0 ]; then
                    echo "ERROR: no JOB_ID_KP1x entries found in log.isobar__create_KP1x" >&2
                    exit 1
                fi

                # # 0.3) Validate that each is a pure integer
                for jid in "${job_ids[@]}"; do
                    if ! [[ $jid =~ ^[0-9]+$ ]]; then
                        echo "ERROR: invalid job ID '$jid' extracted from log.isobar__create_KP1x" >&2
                        exit 1
                    fi
                done

                echo "Found JOB_ID_KP1x IDs: ${job_ids[*]}"

                # # 0.4) For each valid job ID, wait until SLURM no longer lists it
                for jid in "${job_ids[@]}"; do
                    echo -n "Waiting for SLURM job $jid to finish"
                    while squeue -h -j "$jid" &>/dev/null; do
                        echo -n "."      # progress dot
                        sleep "$WAIT_TIME_LONG"
                    done
                    echo " done."
                done

                echo "All JOB_ID_KP1x jobs have completed; proceeding with the next steps."
                echo
                echo
                #################
                # UNCOMMENT THE ABOVE LINES!!!











                # # backup check
                # # 1) extract JOB_ID_KP1x from the first line of log.run_sim
                # JOB_ID_KP1x=$(awk 'NR==1 {print $3}' log.run_sim)

                # if [[ -z "$JOB_ID_KP1x" ]]; then
                #     echo "ERROR: could not read JOB_ID_KP1x from log.run_sim"
                #     exit 1
                # fi

                # echo "JOB_ID_KP1x: $JOB_ID_KP1x"
                # echo -n "Waiting for job $JOB_ID_KP1x to finish "

                # # 2) loop until squeue no longer reports it
                # while squeue -h -j "$JOB_ID_KP1x" >/dev/null; do
                #     # print a dot and sleep
                #     echo -n "."
                #     sleep "$WAIT_TIME_LONG"
                # done

                # echo    # newline after the dots
                # echo "Job $JOB_ID_KP1x has completed."
                # echo ""
            elif [[ $rerun -eq 1 ]]; then
                echo "Rerun mode is ON. Skipping the check for completed jobs in ${child}."
            fi


            # only if rerun is 0, check if the directory has already been processed
            if [[ $rerun -eq 0 ]]; then
                # Copy and source analysis script
                cp "${HELP_SCRIPTS_vasp}/data_4_analysis.sh" "${KP1x_dir}/"
                source data_4_analysis.sh
            fi

            # Check for average pressure file
            if [[ ! -f analysis/peavg.out ]]; then
                echo "ERROR: analysis/peavg.out not found in ${child}" >&2
                exit 1
            fi

            # Extract the third field (pressure) from peavg.out
            P_RUN=$(grep Pressure analysis/peavg.out | awk '{print $3}')

            # create array of P_RUN values
            P_RUN_array+=("$P_RUN")
            echo "P_RUN @ $child: $P_RUN GPa"   
            echo ""
            cd $ISOBAR_CALC_dir || exit   
        fi          
    done

    # Check if P_RUN values span the PSTRESS_CHOSEN_GPa
    min_P_RUN=$(printf '%s\n' "${P_RUN_array[@]}" | sort -n | head -n 1)
    max_P_RUN=$(printf '%s\n' "${P_RUN_array[@]}" | sort -n | tail -n 1)
    echo
    echo
    echo "Minimum P_RUN: $min_P_RUN GPa"
    echo "Maximum P_RUN: $max_P_RUN GPa"
    if (( $(echo "$min_P_RUN < $PSTRESS_CHOSEN_GPa" | bc -l) )) && (( $(echo "$max_P_RUN > $PSTRESS_CHOSEN_GPa" | bc -l) )); then
        echo "P_RUN values span the PSTRESS_CHOSEN_GPa = $PSTRESS_CHOSEN_GPa GPa @ $parent"
        echo ""
        echo ""
    else
        counter_P_not_spanned=$((counter_P_not_spanned + 1))
        echo "P_RUN values do NOT span the PSTRESS_CHOSEN_GPa = $PSTRESS_CHOSEN_GPa GPa @ $parent"
        echo "counter_P_not_spanned set to 1"
        echo ""
        echo ""
    fi

done < <(find . -type d -name KP1 -print0)

if [[ $counter_P_not_spanned -eq 0 ]]; then
    echo
    echo "========================="
    echo "========================="
    echo "Across all CONFIGs, KP1x span the PSTRESS_CHOSEN_GPa ($PSTRESS_CHOSEN_GPa). Proceeding with the script."
    echo "========================="
    echo "========================="
    echo ""
    echo ""
    echo ""
else
    echo ""
    echo "========================="
    echo "========================="
    echo "Across all CONFIGs, KP1x do NOT span the PSTRESS_CHOSEN_GPa ($PSTRESS_CHOSEN_GPa). Exiting."
    echo "counter_P_not_spanned = $counter_P_not_spanned (number of CONFIGs that do not span the PSTRESS_CHOSEN_GPa)"
    echo "========================="
    echo "========================="
    echo ""
    echo ""
    echo ""
    exit 1
fi















#--------------------------------------------------------------
# Loop over every directory named exactly 'KP1'
#--------------------------------------------------------------
while IFS= read -r -d '' parent; do
    echo "#----------------------------------------"
    echo "#----------------------------------------"
    echo ""
    echo "Processing parent directory: ${parent}"
    echo ""
    echo "#----------------------------------------"
    echo "#----------------------------------------"
    cd "$parent" || exit

    KP1_dir=$(pwd)
    V_est_dir=$(dirname "$KP1_dir") # parent directory is V_est_dir
    ISOBAR_T_dir=$(dirname "$V_est_dir") # parent directory is ISOBAR_T_dir
    ISOBAR_T_dir_name=$(basename "$ISOBAR_T_dir") # name of the ISOBAR_T_dir directory
    ISOBAR_CALC_dir__test=$(dirname "$ISOBAR_T_dir") # parent directory is ISOBAR_CALC_test_dir

    # Extract TEMP_CHOSEN_ISOBAR from the name of the ISOBAR_T_dir directory (ISOBAR_T_dir_name) -- the format is "T<TEMP_CHOSEN_i>"
    TEMP_CHOSEN_ISOBAR=$(echo "$ISOBAR_T_dir_name" | sed 's/T//g')
    echo "KP1_dir: $KP1_dir"
    echo "V_est_dir: $V_est_dir"
    echo "ISOBAR_T_dir: $ISOBAR_T_dir"
    echo "ISOBAR_T_dir_name: $ISOBAR_T_dir_name"
    echo "==========================="
    echo "TEMP_CHOSEN_ISOBAR: $TEMP_CHOSEN_ISOBAR"
    echo "==========================="
    echo ""

    # if ISOBAR_CALC_dir__test is not the same as ISOBAR_CALC_dir, exit
    if [ "$ISOBAR_CALC_dir__test" != "$ISOBAR_CALC_dir" ]; then
        echo "ERROR: ISOBAR_CALC_dir__test ($ISOBAR_CALC_dir__test) is not the same as ISOBAR_CALC_dir ($ISOBAR_CALC_dir)"
        exit 1
    else
        echo "ISOBAR_CALC_dir__test ($ISOBAR_CALC_dir__test) is the same as ISOBAR_CALC_dir ($ISOBAR_CALC_dir)"
    fi

    cd $ISOBAR_CALC_dir || exit

    # Iterate over immediate subdirectories beginning with 'KP1'
    for child in "${parent}"/KP1*; do
        if [ -d "${child}" ]; then
            # Enter the child directory
            cd "${child}" || exit # KP1a, KP1b, etc.
            KP1x_dir=$(pwd)
            


            #----------------------------------------------------------
            # check how many simulations have already been submitted to SLURM by $USER
            # if this number is greater than 1500 on either the " short" or "vshort" queure, than sleep for another 10 minutes
            # and then check again

            # Get the number of jobs in the "short" and "vshort" queues
            sqpmy_test='squeue -o "%.18i %Q %.9q %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | grep $USER' #priority rating
            short_jobs=$(eval $sqpmy_test | grep " short" | wc -l)
            vshort_jobs=$(eval $sqpmy_test | grep "vshort" | wc -l)

            echo "Checking the job submission levels to normalize..."
            echo "Current job counts - short: $short_jobs, vshort: $vshort_jobs"
            echo "Checking every 10 minutes..."

            while true; do
                # Get the number of jobs in the "short" and "vshort" queues
                short_jobs=$(eval $sqpmy_test | grep " short" | wc -l)
                vshort_jobs=$(eval $sqpmy_test | grep "vshort" | wc -l)

                # Print a dot to indicate progress
                echo -n "."

                # Check if either queue has more than 1500 jobs
                if (( short_jobs > 1500 || vshort_jobs > 1500 )); then
                    echo -n "."      # progress dot
                    sleep $WAIT_TIME_LONG  # Sleep for 10 minutes
                else
                    break
                fi
            done
            echo ""
            echo "Job submission levels are normal (short: $short_jobs, vshort: $vshort_jobs)."
            echo ""
            #----------------------------------------------------------







            # check if $child name contains "KP1a", "KP1b", etc., if not exit with error
            if [[ ! "$child" =~ KP1[a-z] ]]; then
                echo "ERROR: $child does not match KP1[a-z] pattern. Exiting."
                exit 1
            fi

            # only if rerun is 0, check if the directory has already been processed
            if [[ $rerun -eq 0 ]]; then
                # Copy and source analysis script
                # cp "${HELP_SCRIPTS_vasp}/data_4_analysis.sh" "${KP1x_dir}/"
                # source data_4_analysis.sh

                # Check for average pressure file
                if [[ ! -f analysis/peavg.out ]]; then
                    echo "ERROR: analysis/peavg.out not found in ${child}" >&2
                    exit 1
                fi
                # Extract the third field (pressure) from peavg.out
                P_RUN=$(grep Pressure analysis/peavg.out | awk '{print $3}')


                # Run ASE setup (assumed alias/function)
                module load anaconda3/2024.6; conda activate ase_env
                # Generate high-precision calculations using eos script
                rm -rf hp_calculations
                python ${HELP_SCRIPTS_vasp}/eos* -p ${P_RUN} -m 0 -e 0.1 -hp 1 -nt ${N_FRAMES_hp_calculations}
                module purge
            else
                echo "Skipping data_4_analysis.sh+eos_fit__V_at_P.py in ${child} as rerun is set to 1."
            fi

            #----------------------------------------------------------
            # Now enter hp_calculations for KPOINTS_222 runs
            #----------------------------------------------------------
            cd hp_calculations
            hp_calculations_dir=$(pwd)
            echo " → In hp_calculations at ${hp_calculations_dir}"

            # Copy VASP run scripts and input templates
            cp ${SETUP_dir}/RUN_VASP_SPC.sh \
                ${hp_calculations_dir}/RUN_VASP.sh

            # cp ${SETUP_dir}/POTCAR \
                # ${hp_calculations_dir}/POTCAR
            CONFIG_dir_name=$(basename "$CONFIG_dir")
            # if "_0" in the CONFIG_dir_name and COMPOSITION_dir_name contains "MgSiO3", copy corresponding POTCARs
            if [[ "$CONFIG_dir_name" == *_0* && "$COMPOSITION_dir_name" == *"MgSiO3"* ]]; then
                echo "Copying POTCAR for MgSiO3"
                cp $SETUP_dir/POTCAR_MgSiO3 "$hp_calculations_dir/POTCAR"
            elif [[ "$CONFIG_dir_name" == *_0* && "$COMPOSITION_dir_name" == *"Fe"* ]]; then
                echo "Copying POTCAR for Fe"
                cp $SETUP_dir/POTCAR_Fe "$hp_calculations_dir/POTCAR"
            else
                echo "Copying default POTCAR"
                cp $SETUP_dir/POTCAR "$hp_calculations_dir/POTCAR"
            fi

            cp ${SETUP_dir}/KPOINTS_222 \
                ${hp_calculations_dir}/KPOINTS
            cp ${SETUP_dir}/INCAR_SPC \
                ${hp_calculations_dir}/INCAR

            # Substitute placeholders in INCAR
            sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN_ISOBAR}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN_SPC}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_222}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" ${hp_calculations_dir}/INCAR

            # Distribute input files to all subdirectories
            find . -type d | xargs -I {} cp INCAR KPOINTS POTCAR RUN_VASP.sh {}

            # Clean up templates in current and analysis directories
            rm -f KPOINTS INCAR POTCAR RUN_VASP.sh
            cd analysis
            rm -r KPOINTS INCAR POTCAR RUN_VASP.sh
            cd ${hp_calculations_dir} || exit

            # if rerun is 1, check if the jobs are already completed successfully
            if [[ $rerun -eq 1 ]]; then
                module purge
                module load anaconda3/2024.6; conda activate deepmd
                python ${MLDP_SCRIPTS}/post_recal_rerun.py -ip all -v -ss $INPUT_FILES_DIR/sub_vasp_xtra.sh > log.recal_test 2>&1
                module purge
                # wait for log.recal_test to be created
                while [ ! -f ${hp_calculations_dir}/log.recal_test ]; do
                    sleep 1
                done
                # grab all numbers from the log file that come after the line with [Folder Path] and before line with "Run 'source rerun'". Only grab the first number from each line though.
                awk '
                    /\[Folder Path\]/  { flag = 1; next }          # start copying *after* this line
                    /Run '\''source rerun'\''/ { flag = 0 }        # stop when this line appears
                    flag {                                         # only while inside the block
                        if (match($0, /[0-9]+/))                   # grab first number on the line
                            print substr($0, RSTART, RLENGTH)
                    }
                '  log.recal_test > rerun_folders.dat            

                logfile="log.recal_test"
                line_count=$(wc -l < "$logfile")    # Count the number of lines in the file
                num_failed_recal_frames=$((${line_count}-14)) # name says it all ...
                echo "Number of failed recal frames: $num_failed_recal_frames"
                counter_incomplete_runs=$(($counter_incomplete_runs + $num_failed_recal_frames))

            fi

            #----------------------------------------------------------
            # Submit VASP jobs in each subfolder
            #----------------------------------------------------------
            for subchild in *; do
                if [ -d "${subchild}" ]; then
                    cd "${subchild}" || exit
                    rm -f WAVE* CHG*

                    # if subchild is "analysis", skip it
                    if [[ "${subchild}" == "analysis" ]]; then
                        echo "Skipping analysis directory in ${child}"
                        cd ${hp_calculations_dir} || exit
                        continue
                    fi

                    # if rerun is 1, grep "Total CPU time used" in OUTCAR. If it is not found, then submit the job
                    if [[ $rerun -eq 1 ]]; then
                        subchild_basename=$(basename "$subchild")
                        # Check if the subchild directory is in the rerun_folders.dat file. If yes, then rerun the job
                        if grep -q "${subchild_basename}" ${hp_calculations_dir}/rerun_folders.dat; then
                            echo "Subdirectory ${subchild_basename} is in rerun_folders.dat. Rerunning the job with ALGO = Normal."
                            # find . -type f -name 'INCAR' -exec sed -i 's/ALGO   = F/ALGO   = N/g' {} +
                            sbatch RUN_VASP.sh
                            LAST_JOB_ID=$(squeue -u $USER -h -o "%i" | sort -n | tail -1)
                            echo "   → Started VASP in ${subchild}"
                            echo "JOB_ID_hp_calc: $LAST_JOB_ID"
                            echo ""
                        else
                            echo "VASP job already completed in ${subchild}. Skipping submission."
                        fi
                    else
                        # If rerun is 0, just submit the job
                        sbatch RUN_VASP.sh
                        LAST_JOB_ID=$(squeue -u $USER -h -o "%i" | sort -n | tail -1)
                        echo "   → Started VASP in ${subchild}"
                        echo "JOB_ID_hp_calc: $LAST_JOB_ID"
                        echo ""
                    fi

                    # Return to the hp_calculations directory
                    cd ${hp_calculations_dir} || exit

                fi
            done

            echo "Started hp_calculations in ${child}"
            echo ""
            # Return to the original driver directory
            cd ${ISOBAR_CALC_dir} || exit
        fi
    done
    echo ""
    echo "++++++++++++++++++++++++++++++"
    echo ""
    echo ""
done < <(find . -type d -name KP1 -print0)

# Final confirmation message
echo ""
echo ""
echo ""
echo "################################"
echo "################################"
echo "################################"
echo "Parameter rerun = ${rerun}"
echo "All ISOBAR VASP jobs started in all KP1/*/hp_calculations directories under ${ISOBAR_CALC_dir} by $(date)."
echo "If you want to rerun the jobs with ALGO = N, run the script again with rerun=1."
echo "Number of incomplete runs across compositions (ignore if rerun=0): ${counter_incomplete_runs}"
echo "################################"
echo "################################"
echo "################################"
echo ""
echo ""
echo ""
