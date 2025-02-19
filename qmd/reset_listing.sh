#!/bin/bash

# ========================================================================================
# Script: Reset Folder Listing
#
# Summary:
# This script scans the current directory for folders whose names start with a digit, 
# sorts them numerically, and renames them sequentially starting from "1".
# 
# Usage:
# - Run this script in a directory containing numbered folders.
# - Folders will be renamed in increasing order, ensuring they are sequential.
#
# Steps:
# 1. Retrieve all directories that start with a digit.
# 2. Sort them in natural order.
# 3. Rename each folder sequentially, starting from 1.
#
# Notes:
# - The script will not modify non-numeric folders.
# - If no matching folders are found, it exits gracefully.
# - Uses `sort -V` for natural sorting (e.g., 1, 2, 10 instead of 1, 10, 2).
#
# Author: akashgpt
# ========================================================================================

echo ""
echo "==============================="
echo "Resetting folder listing..."
echo "Current folder: $(pwd)"

# Step 1: Get a sorted list of directories that start with a digit
folders=($(ls -d [0-9]* 2>/dev/null | sort -V))

# Step 2: If no folders match, exit
if [ ${#folders[@]} -eq 0 ]; then
    echo "No folders found that start with a digit."
    exit 1
fi

# Step 3: Rename the folders sequentially starting from 1
new_index=1  # Start renaming from folder "1"
counter=0    # Counter for renamed folders

for folder in "${folders[@]}"; do
    # Skip renaming if the folder is already in correct order
    if [ "$folder" != "$new_index" ]; then
        mv "$folder" "$new_index"
        echo "Renamed: $folder -> $new_index"
        ((counter++))
    fi
    ((new_index++))  # Move to the next number
done

if [ $counter -eq 0 ]; then
    echo "No folders were renamed."
else
    echo "Renamed $counter folders."
fi
echo "==============================="
echo ""