#!/bin/bash

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

for folder in "${folders[@]}"; do
    # Skip renaming if the folder is already in correct order
    if [ "$folder" != "$new_index" ]; then
        mv "$folder" "$new_index"
        echo "Renamed: $folder -> $new_index"
    fi
    ((new_index++))  # Move to the next number
done
