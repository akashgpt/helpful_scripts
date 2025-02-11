#!/bin/bash


cd ..
cp -r recal old_recal
cd recal || { echo "Failed to enter recal directory. Exiting."; exit 1; }
rm -r deepmd OUTCAR

# Define the input file with folder names
input_file="dp_test_vid_e_or_f"

# Read the folder names from the file, skipping the first line
folders_to_keep=$(tail -n +2 "$input_file")

# Convert the list of folders to an array
IFS=$'\n' read -d '' -r -a keep_array <<< "$folders_to_keep"

# Get the list of all directories in the current directory
all_folders=($(ls -d */))

# Iterate through all directories
for folder in "${all_folders[@]}"; do
    # Remove the trailing slash
    folder=${folder%/}
    # Check if the current folder is in the list of folders to keep
    if [[ ! " ${keep_array[@]} " =~ " ${folder} " ]]; then
        echo "Deleting folder: $folder"
        rm -rf "$folder"
    else
        echo "Keeping folder: $folder"
    fi
done

#inside recal to create new deepmd files
python $mldp/merge_out.py -o OUTCAR
python $mldp/extract_deepmd.py -d deepmd -ttr 10000
