#!/bin/bash

# ========================================================================================
# Script: Find and Reset Listings in "recal" Folders
#
# Summary:
# - Finds all folders named "recal" at any level within the current directory.
# - Changes into each "recal" directory.
# - Sources the script "$HELP_SCRIPTS/qmd/reset_listing.sh" within each.
#
# Usage:
# - Run this script from the root directory where you want to search.
# - The script will recursively find "recal" directories and reset their listings.
#
# Notes:
# - `find` ensures all "recal" folders are processed.
# - `source` runs the script in the current shell (ensure it's executable).
# - If no "recal" folders exist, a message is displayed and the script exits.
#
# Author: Akash Gupta
#
# ========================================================================================

echo "Searching for all 'recal' folders and sourcing reset_listing.sh..."

# Find all directories named "recal"
recal_folders=$(find . -type d -name "recal")

# Check if any "recal" folders were found
if [ -z "$recal_folders" ]; then
    echo "No 'recal' folders found in the current directory."
    exit 1
fi

# Loop through each "recal" folder
for dir in $recal_folders; do
    echo "Processing: $dir"
    cd "$dir" || { echo "Failed to enter $dir"; continue; }
    
    # Source the reset listing script
    if [ -f "$HELP_SCRIPTS/qmd/reset_listing.sh" ]; then
        source "$HELP_SCRIPTS/qmd/reset_listing.sh"
        echo "Successfully sourced reset_listing.sh in $dir"
    else
        echo "Error: $HELP_SCRIPTS/qmd/reset_listing.sh not found!"
    fi

    cd - > /dev/null  # Return to the previous directory
done

echo "Completed processing all 'recal' folders."
