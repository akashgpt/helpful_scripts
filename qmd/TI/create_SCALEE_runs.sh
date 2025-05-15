#!/bin/bash

# Summary:
# This script creates directories for different SCALEE values and copies the necessary files for VASP simulations.

run_switch=${1:-0} # 0: create directories, 1: create directories and run VASP simulations

# Input parameters
SCALEE=(1.0 0.71792289 0.3192687 0.08082001 0.00965853 0.00035461 0.00000108469)
TEMP_CHOSEN=3000
POTIM_CHOSEN=0.5
NPAR_CHOSEN=14 # choose based on CLUSTER || the number of cores per node and the number of nodes; Preferred values: TIGER3: 14, STELLAR: 16

parent_dir=$(pwd)

# Constants
kB=0.00008617333262145  # Boltzmann constant in eV/K

# Calculate SIGMA_CHOSEN based on the chosen temperature
SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN" | bc -l)

CLUSTER_NAME=$(scontrol show config | grep ClusterName | awk '{print $3}')

echo "SCALEE array has ${#SCALEE[@]} elements"
echo "TEMP_CHOSEN: ${TEMP_CHOSEN}"
# echo "SIGMA_CHOSEN: ${SIGMA_CHOSEN}"
echo "POTIM_CHOSEN: ${POTIM_CHOSEN}"
echo "NPAR_CHOSEN: ${NPAR_CHOSEN}"
echo "CLUSTER_NAME: ${CLUSTER_NAME}"
echo ""
echo "Creating/Updating dir for:"

# echo "SCALEE elements are:"

counter=0

for SCALEE_CHOSEN in "${SCALEE[@]}"; do

    counter=$((counter+1))
    echo "SCALEE_CHOSEN ${counter}: ${SCALEE_CHOSEN}"

    mkdir -p "SCALEE_${counter}"
    cp setup_TI/* "SCALEE_${counter}"
    cd "SCALEE_${counter}" || exit 1

    # Replace __SCALEE_CHOSEN__ with the SCALEE_CHOSEN
    sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" INCAR

    # Replace __TEMP_CHOSEN__ and __SIGMA_CHOSEN__ based on the chosen temperature
    sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" INCAR
    sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" INCAR
    sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" INCAR

    # Replace __NPAR_CHOSEN__ with the chosen NPAR
    sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" INCAR

    if [ $run_switch -eq 1 ]; then
        sbatch RUN_VASP.sh
    fi

    cd $parent_dir || exit 1

done

echo "Done."