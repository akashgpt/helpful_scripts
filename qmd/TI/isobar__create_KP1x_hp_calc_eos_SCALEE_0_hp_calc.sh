#!/bin/bash
# set -euo pipefail

# Usage: 
# (for new run): source $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc.sh > log.isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc 2>&1 &
# (for new run): nohup bash $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc.sh > log.isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc 2>&1 &
# (for rerun with same parameters): source $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc.sh 1 > log.isobar__create_KP1x_hp_calc_eos_SCALEE_0_hp_calc 2>&1 &
# Author: Akash Gupta


rerun=${1:-0}  # Default to 0 if not provided; 1 for a rerun with same parameters

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
SCALEE_CHOSEN=1.0 # Default value for SCALEE_CHOSEN, can be overridden by the input file
# Compute sigma = k_B * T (for smearing)
# SIGMA_CHOSEN=$(echo "${kB} * ${TEMP_CHOSEN}" | bc -l)


# for each folder in the current directory, except, master_setup_TI, go inside and run source $HELP_SCRIPTS_TI/calculate_GFE_v2.sh 1 2 > log.calculate_GFE 2>&1 &


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

PSTRESS_CHOSEN=$(echo "$PSTRESS_CHOSEN_GPa * 10" | bc -l)  # Convert GPa to kBar


########################################
########################################
########################################
NPAR_CHOSEN_SPC=7 # For single point calculations, set NPAR_CHOSEN to 2; KPAR is 4 for hp_calculations
N_FRAMES_hp_calculations__SCALEE_1=$(( ${N_FRAMES_hp_calculations} * 2 ))  # Number of frames for high-precision calculations at SCALEE=1
########################################
########################################
########################################


#-------------------------
# Print the parameters
echo "------------------------"
echo "Simulation parameters:"
echo "TEMP_CHOSEN: $TEMP_CHOSEN"
echo "PSTRESS_CHOSEN_GPa: $PSTRESS_CHOSEN_GPa"
echo "PSTRESS_CHOSEN: $PSTRESS_CHOSEN (kBar)"
echo "NPAR_CHOSEN: $NPAR_CHOSEN_SPC (special case for single point calculations)"
echo "POTIM_CHOSEN: $POTIM_CHOSEN"
echo "NBANDS_CHOSEN: $NBANDS_CHOSEN"
echo "KPAR_CHOSEN_111: $KPAR_CHOSEN_111"
echo "KPAR_CHOSEN_222: $KPAR_CHOSEN_222"
echo "WAIT_TIME_VLONG: $WAIT_TIME_VLONG"
echo "WAIT_TIME_LONG: $WAIT_TIME_LONG"
echo "WAIT_TIME_SHORT: $WAIT_TIME_SHORT"
echo "N_FRAMES_hp_calculations: $N_FRAMES_hp_calculations__SCALEE_1 (special case for single point calculations for SCALEE 1)"
echo "SCALEE_CHOSEN: $SCALEE_CHOSEN"
echo "MLDP_SCRIPTS: $MLDP_SCRIPTS"
echo "-------------------------"



# Check if the folder name contains both TEMP_CHOSEN and PSTRESS_CHOSEN_GPa
if [[ "$PT_dir_name" == *"${TEMP_CHOSEN}"* && "$PT_dir_name" == *"${PSTRESS_CHOSEN_GPa}"* ]]; then
    echo "Folder name contains TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
else
    echo "Folder name does NOT contain both TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
    exit 1
fi




counter_incomplete_runs=0



echo ""
echo ""
echo ""




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

    cd "${parent}" || exit
    parent_abs=$(pwd)
    SCALEE_0_dir=$parent_abs
    echo " → Entered ${SCALEE_0_dir}"

    ISOBAR_T_dir=$(dirname "$SCALEE_0_dir")
    ISOBAR_T_dir_name=$(basename "$ISOBAR_T_dir")
    ISOBAR_CALC_dir__test=$(dirname "$ISOBAR_T_dir") # parent directory is ISOBAR_CALC_test_dir
    V_est_dir=$ISOBAR_T_dir/V_est
    KP1_dir=$V_est_dir/KP1

    LOCAL_SETUP_dir=$ISOBAR_T_dir/setup_TI




    # Check if the parent directory has already been processed "done_hp_calculations" in hp_calculations directory
    if [[ -f "${SCALEE_0_dir}/hp_calculations/done_hp_calculations" ]]; then
        echo ""
        echo "----------------------------------------"
        echo "Parent directory ${SCALEE_0_dir} has already been processed. Found \"done_hp_calculations\". Skipping."
        echo "----------------------------------------"
        echo ""
        cd $ISOBAR_CALC_dir || exit 1 # Return to ISOBAR_CALC_dir
        continue
    fi

    # Check if "to_RUN__correction" or "to_RUN__hp_calc" exists in SCALEE_0_dir and skip if it does not
    if [[ ! -f "${SCALEE_0_dir}/to_RUN__correction" && ! -f "${SCALEE_0_dir}/to_RUN__hp_calc" ]]; then
        echo ""
        echo "----------------------------------------"
        echo "Parent directory ${SCALEE_0_dir} does NOT have a file \"to_RUN__correction\" or \"to_RUN__hp_calc\". Skipping."
        echo "----------------------------------------"
        echo ""
        cd "$ISOBAR_CALC_dir" || exit 1 # Return to ISOBAR_CALC_dir
        continue
    fi



    # Extract TEMP_CHOSEN_ISOBAR from the name of the ISOBAR_T_dir directory (ISOBAR_T_dir_name) -- the format is "T<TEMP_CHOSEN_i>"
    TEMP_CHOSEN_ISOBAR=$(echo "$ISOBAR_T_dir_name" | sed 's/T//g')
    SIGMA_CHOSEN_ISOBAR=$(echo "$kB * $TEMP_CHOSEN_ISOBAR" | bc -l)  # Gaussian smearing sigma
    echo "KP1_dir: $KP1_dir"
    echo "V_est_dir: $V_est_dir"
    echo "ISOBAR_T_dir: $ISOBAR_T_dir"
    echo "ISOBAR_T_dir_name: $ISOBAR_T_dir_name"
    echo 
    echo "==========================="
    echo "TEMP_CHOSEN_ISOBAR: $TEMP_CHOSEN_ISOBAR"
    echo "SIGMA_CHOSEN_ISOBAR: $SIGMA_CHOSEN_ISOBAR"
    echo "==========================="
    echo ""


    # if ISOBAR_CALC_dir__test is not the same as ISOBAR_CALC_dir, exit
    if [ "$ISOBAR_CALC_dir__test" != "$ISOBAR_CALC_dir" ]; then
        echo ""
        echo "ERROR: ISOBAR_CALC_dir__test ($ISOBAR_CALC_dir__test) is not the same as ISOBAR_CALC_dir ($ISOBAR_CALC_dir)"
        echo ""
        exit 1
    else
        echo ""
        echo "ISOBAR_CALC_dir__test ($ISOBAR_CALC_dir__test) is the same as ISOBAR_CALC_dir ($ISOBAR_CALC_dir)"
        echo ""
    fi


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



    child=$parent # to make the previous script work, we need to set child to parent
    child_abs=$SCALEE_0_dir

    # only if rerun is 0, check if the directory has already been processed
    if [[ $rerun -eq 0 ]]; then
        # Copy and source analysis script
        cp "${HELP_SCRIPTS_vasp}/data_4_analysis.sh" "${child_abs}/"
        source data_4_analysis.sh

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
        python ${HELP_SCRIPTS_vasp}/eos_fit__V_at_P* -p ${P_RUN} -m 0 -e 0.1 -hp 1 -nt ${N_FRAMES_hp_calculations__SCALEE_1} # creates hp_calculations directory
        module purge
        echo "Executed eos_fit__V_at_P.py in ${child} with P_RUN = ${P_RUN} GPa."
    else
        echo "Skipping data_4_analysis.sh+eos_fit__V_at_P.py in ${child} as rerun is set to 1."
    fi

    #----------------------------------------------------------
    # Now enter hp_calculations for KPOINTS_222 runs
    #----------------------------------------------------------
    cd hp_calculations || exit
    hp_calculations_dir=$(pwd)
    echo " → In hp_calculations at ${hp_calculations_dir}"

    # Copy VASP run scripts and input templates
    cp ${PT_dir}/master_setup_TI/RUN_VASP_SPC.sh \
        ${hp_calculations_dir}/RUN_VASP.sh
    
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


    cp ${PT_dir}/master_setup_TI/KPOINTS_222 \
        ${hp_calculations_dir}/KPOINTS
    cp ${PT_dir}/master_setup_TI/INCAR_SPC \
        ${hp_calculations_dir}/INCAR

    # Substitute placeholders in INCAR
    sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN_ISOBAR}/" ${hp_calculations_dir}/INCAR
    sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" ${hp_calculations_dir}/INCAR
    sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" ${hp_calculations_dir}/INCAR
    sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN_SPC}/" ${hp_calculations_dir}/INCAR
    sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_222}/" ${hp_calculations_dir}/INCAR
    sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN_ISOBAR}/" ${hp_calculations_dir}/INCAR
    sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" ${hp_calculations_dir}/INCAR

    # Distribute input files to all subdirectories
    find . -type d | xargs -I {} cp INCAR KPOINTS POTCAR RUN_VASP.sh {}

    # Clean up templates in current and analysis directories
    rm -f KPOINTS INCAR POTCAR RUN_VASP.sh
    cd analysis || exit
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
    echo ""
    echo "++++++++++++++++++++++++++++++"
    echo ""
    echo ""
    #     fi
    # done
done < <(find . -type d -name SCALEE_0 -print0)

# Final confirmation message
echo ""
echo ""
echo ""
echo "################################"
echo "################################"
echo "################################"
echo "Parameter rerun = ${rerun}"
echo "All VASP jobs started in all KP1/*/hp_calculations directories under ${ISOBAR_CALC_dir}."
echo "If you want to rerun the jobs with ALGO = N, run the script again with rerun=1."
echo "Number of incomplete runs across compositions (ignore if rerun=0): ${counter_incomplete_runs}"
echo "################################"
echo "################################"
echo "################################"
echo ""
echo ""
echo ""
