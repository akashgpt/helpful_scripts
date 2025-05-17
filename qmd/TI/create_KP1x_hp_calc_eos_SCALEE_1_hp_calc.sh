#!/bin/bash
# set -euo pipefail

# Usage: 
# (for new run): source $HELP_SCRIPTS_TI/create_KP1x_hp_calc_eos_SCALEE_1_hp_calc.sh > log.create_KP1x_hp_calc_eos_SCALEE_1_hp_calc 2>&1 &
# (for rerun with same parameters): source $HELP_SCRIPTS_TI/create_KP1x_hp_calc_eos_SCALEE_1_hp_calc.sh 1 > log.create_KP1x_hp_calc_eos_SCALEE_1_hp_calc 2>&1 &
# Author: Akash Gupta


# while IFS= read -r -d '' parent; do
#     # …look for its immediate subdirectories starting with “KP1”
#     for child in "$parent"/KP1*; do
#         if [ -d "$child" ]; then
#         touch "$child/done_KP1x"
#         # P_RUN=$(grep Pressure analysis/peavg.out  | awk '{print $3}')
#         # module load anaconda3/2024.6; conda activate ase_env; python $HELP_SCRIPTS_vasp/eos* -p $P_RUN -m 0 -e 0.01 -hp 1
#         echo "Touched $child/done_KP1x"
#         fi
#     done
# done < <(find . -type d -name KP1 -print0)



#---------------------------    -----------------------------------
# Driver script to set up and launch VASP runs in KP1*/hp_calculations
#--------------------------------------------------------------

# # Thermostat and MD settings
# TEMP_CHOSEN=13000            # Target temperature (K)
# PSTRESS_CHOSEN_GPa=1000     # Pressure offset (GPa), not used here
# NBANDS_CHOSEN=784            # Number of electronic bands (e.g., 560 for Fe-He; 784 for MgSiO3-He)
# POTIM_CHOSEN=0.5             # MD timestep (fs)
# NPAR_CHOSEN=14               # Parallelization: cores/node × nodes (e.g., TIGER3=14, STELLAR=16)

# # K-point parallelization choices
# KPAR_CHOSEN_111=1            # KPAR for 1×1×1 k-point mesh
# KPAR_CHOSEN_222=4            # KPAR for 2×2×2 k-point mesh

# # Job wait-time thresholds (mins)
# WAIT_TIME_VLONG=600          # Very long jobs
# WAIT_TIME_LONG=60            # Long jobs
# WAIT_TIME_SHORT=10           # Short jobs

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
# Compute sigma = k_B * T (for smearing)
# SIGMA_CHOSEN=$(echo "${kB} * ${TEMP_CHOSEN}" | bc -l)

# Save the current working directory for later
PT_dir=$(pwd)
PT_dir_name=$(basename "$PT_dir")

COMPOSITION_dir=$(dirname "$PT_dir")
COMPOSITION_dir_name=$(basename "$COMPOSITION_dir")

echo "Current time: $(date)"
echo "Current PT directory: $PT_dir"
echo "Current PT directory name: $PT_dir_name"
echo "Current COMPOSITION directory: $COMPOSITION_dir"
echo "Current COMPOSITION directory name: $COMPOSITION_dir_name"
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

SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN" | bc -l)  # Gaussian smearing sigma


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
    echo " → Entered ${parent_abs}"
    # cp $HELP_SCRIPTS_vasp/data_4_analysis.sh .
    # source data_4_analysis.sh 

    child=$parent # to make the previous script work, we need to set child to parent
    child_abs=$parent_abs

    # Iterate over immediate subdirectories beginning with 'KP1'
    # for child in "${parent}"/KP1*; do
    #     if [ -d "${child}" ]; then
    #         # Enter the child directory
    #         cd "${child}" || exit
    #         child_abs=$(pwd)
    #         echo " → Entered ${child_abs}"

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
        python ${HELP_SCRIPTS_vasp}/eos* -p ${P_RUN} -m 0 -e 0.1 -hp 1 -nt ${N_FRAMES_hp_calculations__SCALEE_1} # creates hp_calculations directory
        module purge
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
    cp ${PT_dir}/master_setup_TI/POTCAR \
        ${hp_calculations_dir}/POTCAR
    cp ${PT_dir}/master_setup_TI/KPOINTS_222 \
        ${hp_calculations_dir}/KPOINTS
    cp ${PT_dir}/master_setup_TI/INCAR_SPC \
        ${hp_calculations_dir}/INCAR

    # Substitute placeholders in INCAR
    sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" ${hp_calculations_dir}/INCAR
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
    cd analysis || exit
    rm -r KPOINTS INCAR POTCAR RUN_VASP.sh
    cd ${hp_calculations_dir} || exit

    # if rerun is 1, check if the jobs are already completed successfully
    if [[ $rerun -eq 1 ]]; then
        l_deepmd
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
    # Return to the original driver directory
    cd ${PT_dir} || exit
    #     fi
    # done
done < <(find . -type d -name SCALEE_1 -print0)

# Final confirmation message
echo ""
echo ""
echo ""
echo "################################"
echo "################################"
echo "################################"
echo "Parameter rerun = ${rerun}"
echo "All VASP jobs started in all KP1/*/hp_calculations directories under ${PT_dir}."
echo "If you want to rerun the jobs with ALGO = N, run the script again with rerun=1."
echo "Number of incomplete runs across compositions (ignore if rerun=0): ${counter_incomplete_runs}"
echo "################################"
echo "################################"
echo "################################"
echo ""
echo ""
echo ""
