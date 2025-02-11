#!/bin/bash

####################################################################################################
# Summary:
# This script adds new system entries to an existing JSON file.
# Usage:
#   ./add_systems.sh path/to/your.json path/to/new_systems.txt
#   - The JSON file should contain a key "training.training_data.systems" that holds an array.
#   - The new_systems.txt file should have a header on the first line, followed by one system per line.
#
# The script reads new system entries (skipping the header), converts them into a JSON array, and
# prepends them to the "systems" array in the JSON file. The updated JSON content then replaces the original file.
#
# Author: akashgpt
####################################################################################################

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 path/to/your.json path/to/new_systems.txt"
    exit 1
fi

json_file=$1
new_systems_file=$2

# Temporary file to hold updated JSON
temp_file=$(mktemp)

# Read new systems into an array, skipping the first line
mapfile -t new_systems < <(tail -n +2 "$new_systems_file")

# Convert array to JSON array format
new_systems_json=$(printf '%s\n' "${new_systems[@]}" | jq -R . | jq -s .)

# Use jq to add new systems to the top of the existing systems array
jq --argjson newSystems "$new_systems_json" '.training.training_data.systems = ($newSystems + .training.training_data.systems)' "$json_file" > "$temp_file"

# Move the temporary file to the original file
mv "$temp_file" "$json_file"

echo "New systems added to the top of the systems array."