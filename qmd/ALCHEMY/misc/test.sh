#!/bin/bash

# Assuming these variables are set
SECONDARY_CLUSTER_ID="tiger3"  # Replace with your actual cluster ID
SECONDARY_CLUSTER_REL_ADDRESS="BURROWS/akashgpt"  # Replace with your relative address
sim_data_dir="/scratch/gpfs/ag5805/qmd_data/NH3_MgSiO3/sim_data_ML_test__DELLAxTIGER3"  # Example path containing $USER

# Check if SECONDARY_CLUSTER_ID equals tiger3
if [[ "$SECONDARY_CLUSTER_ID" == "tiger3" ]]; then
    # Replace $USER in sim_data_dir with $SECONDARY_CLUSTER_REL_ADDRESS
    sim_data_dir_SECONDARY_CLUSTER="${sim_data_dir//"$USER"/"$SECONDARY_CLUSTER_REL_ADDRESS"}"
fi

# Print the updated path for verification
echo "sim_data_dir: $sim_data_dir"
echo "Updated sim_data_dir_SECONDARY_CLUSTER: $sim_data_dir_SECONDARY_CLUSTER"