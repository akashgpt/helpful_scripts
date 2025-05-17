#!/bin/bash
# set -euo pipefail

# Usage: source $HELP_SCRIPTS_TI/create_KP1x.sh > log.create_KP1x 2>&1 &
#        nohup bash $HELP_SCRIPTS_TI/create_KP1x.sh > log.create_KP1x 2>&1 &
# Author: Akash Gupta

#--------------------------------------------------------------
# Script to generate KP1* directories and submit NVT VASP runs
#--------------------------------------------------------------

# big warning message saying better to do 4 - KP 222 calculations rather than 111
echo ""
echo ""
echo ""
echo "#==========================================#"
echo "#==========================================#"
echo "#==========================================#"
echo "WARNING: It is better to do 4 {KP 222} calculations rather than 4 {KP 111} + hp_calculcations"
echo "Not sure though ..."
echo "Thermal pressure correction is likely larger than a simple external pressure offset between KPOINTS 222 vs 111. Not 100% sure though."
echo "#==========================================#"
echo "#==========================================#"
echo "#==========================================#"
echo ""
echo ""
echo ""




#-------------------------
# Simulation parameters
#-------------------------
# TEMP_CHOSEN=6500                     # Target temperature (K)
# PSTRESS_CHOSEN_GPa=250               # Pressure offset (GPa), not used here
# NBANDS_CHOSEN=784                    # Number of bands: 784 for MgSiO3-He; use 560 for Fe-He
# POTIM_CHOSEN=0.5                     # MD time step (fs)
# NPAR_CHOSEN=14                       # Parallelization: cores per node; TIGER3=14, STELLAR=16
# KPAR_CHOSEN_111=1                    # K-point parallelization for KPOINTS 1×1×1
# KPAR_CHOSEN_222=4                    # K-point parallelization for KPOINTS 2×2×2

# #-------------------------
# # Timing controls (in minutes)
# #-------------------------
# WAIT_TIME_VLONG=600                  # Very long jobs
# WAIT_TIME_LONG=60                    # Long jobs
# WAIT_TIME_SHORT=10                   # Short jobs

#-------------------------
# Free energy scaling factor
#-------------------------
SCALEE_CHOSEN=1.0                    # Scale factor for free energy calculations

#-------------------------
# Derived constants
#-------------------------
kB=0.00008617333262145               # Boltzmann constant in eV/K
# SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN" | bc -l)  # Gaussian smearing sigma

#-------------------------
# Working directory
#-------------------------
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



#-------------------------
# Print the parameters
echo "------------------------"
echo "Simulation parameters:"
echo "TEMP_CHOSEN: $TEMP_CHOSEN"
echo "PSTRESS_CHOSEN_GPa: $PSTRESS_CHOSEN_GPa"
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
echo "-------------------------"
#-------------------------


# Check if the folder name contains both TEMP_CHOSEN and PSTRESS_CHOSEN_GPa
if [[ "$PT_dir_name" == *"${TEMP_CHOSEN}"* && "$PT_dir_name" == *"${PSTRESS_CHOSEN_GPa}"* ]]; then
    echo "Folder name contains TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
else
    echo "Folder name does NOT contain both TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
    exit 1
fi

#-------------------------
# Main loop: find KP1 dirs
#-------------------------
while IFS= read -r -d '' parent; do
    echo
    echo
    echo
    echo "=========================="
    echo "Processing $parent"
    # Enter parent directory
    cd "$parent" || exit

    echo "###########################"
    echo 
    echo "###########################"
    echo "UNCOMMENT THE FOLLOWING LINES IN THE CODE!!!"
    echo "###########################"
    echo
    echo "###########################"
    # 0) file log.create_KP1 has numbers next to "JOB_ID_KP1:" -- grab all those numbers and check if they are running
    #    if they are running, sleep and wait for them to finish
    #    if they are not running, continue to the next step
    #    if there are no numbers, prompt error and exit
    # 0.1) Extract all tokens after "JOB_ID_KP1:" 
    # mapfile -t job_ids < <(grep -oP 'JOB_ID_KP1:\s*\K\S+' log.create_KP1)

    # # 0.2) Error if none found
    # if [ ${#job_ids[@]} -eq 0 ]; then
    # echo "ERROR: no JOB_ID_KP1 entries found in log.create_KP1" >&2
    # exit 1
    # fi

    # # 0.3) Validate that each is a pure integer
    # for jid in "${job_ids[@]}"; do
    # if ! [[ $jid =~ ^[0-9]+$ ]]; then
    #     echo "ERROR: invalid job ID '$jid' extracted from log.create_KP1" >&2
    #     exit 1
    # fi
    # done

    # echo "Found JOB_ID_KP1 IDs: ${job_ids[*]}"

    # # 0.4) For each valid job ID, wait until SLURM no longer lists it
    # for jid in "${job_ids[@]}"; do
    # echo -n "Waiting for SLURM job $jid to finish"
    # while squeue -h -j "$jid" &>/dev/null; do
    #     echo -n "."      # progress dot
    #     sleep "$WAIT_TIME_LONG"
    # done
    # echo " done."
    # done

    # echo "All JOB_ID_KP1 jobs have completed; proceeding with the next steps."
    # echo
    # echo
    #################
    # UNCOMMENT THE ABOVE LINES!!!



    # backup check
    # 1) extract JOB_ID_KP1 from the first line of log.run_sim
    JOB_ID_KP1=$(awk 'NR==1 {print $3}' log.run_sim)

    if [[ -z "$JOB_ID_KP1" ]]; then
        echo "ERROR: could not read JOB_ID_KP1 from log.run_sim"
        exit 1
    fi

    echo "JOB_ID_KP1: $JOB_ID_KP1"
    echo -n "Waiting for job $JOB_ID_KP1 to finish "

    # 2) loop until squeue no longer reports it
    while squeue -h -j "$JOB_ID_KP1" >/dev/null; do
        # print a dot and sleep
        echo -n "."
        sleep "$WAIT_TIME_LONG"
    done

    echo    # newline after the dots
    echo "Job $JOB_ID_KP1 has completed."
    echo


    # perform analysis
    cp $HELP_SCRIPTS_vasp/data_4_analysis.sh .
    source data_4_analysis.sh

    # Generate KP1* subdirectories with EOS script
    module load anaconda3/2024.6; conda activate ase_env
    python $HELP_SCRIPTS_vasp/eos* \
        -p $PSTRESS_CHOSEN_GPa -m 0 -e 0.025 -hp -1  # create KP1a, KP1b, etc.

    # Return to root
    cd $PT_dir || exit

    # Loop over each KP1* subdirectory
    for child in "$parent"/KP1*; do
        if [ -d "$child" ]; then
            cd "$child" || exit
            child_abs=$(pwd)

            # Copy VASP run scripts and input templates
            cp ${SETUP_dir}/RUN_VASP_NVT.sh $child_abs/RUN_VASP.sh
            cp ../POTCAR           $child_abs/POTCAR
            cp ../KPOINTS         $child_abs/KPOINTS   # KPOINTS 1×1×1
            cp ${SETUP_dir}/INCAR_SCALEE $child_abs/INCAR

            # Populate INCAR placeholders
            sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" $child_abs/INCAR
            sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" $child_abs/INCAR
            sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" $child_abs/INCAR
            sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" $child_abs/INCAR
            sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_111}/" $child_abs/INCAR
            sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" $child_abs/INCAR
            sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" $child_abs/INCAR

            # Submit the VASP job
            sbatch RUN_VASP.sh
            LAST_JOB_ID=$(squeue -u $USER -h -o "%i" | sort -n | tail -1)
            echo "   → Started VASP in ${child}"
            echo "JOB_ID_KP1x: $LAST_JOB_ID"
            echo ""

            # Return to root directory for next iteration
            cd $PT_dir || exit
        fi
    done
    echo "++++++++++++++++++++++++++++++"
    echo 
    echo

done < <(find . -type d -name KP1 -print0)

# Final status message
echo ""
echo "=========================="
echo "All VASP jobs started in KP1 directories in ${PT_dir}."
echo "=========================="
echo ""