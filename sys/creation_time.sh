#!/bin/bash
# This script lists the creation time of each immediate subfolder in a specified parent folder.
# If the creation time is not available, it falls back to the last modification time.
# Usage: source $HELP_SCRIPTS/sys/creation_time.sh

#TARGET_DIR= current dir unless provided "/path/to/parent_folder"
TARGET_DIR="${1:-.}"
# Check if the target directory exists
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Target directory '$TARGET_DIR' does not exist."
    exit 1
fi
# Check if the target directory is a directory
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: '$TARGET_DIR' is not a directory."
    exit 1
fi

# Loop over immediate subfolders only:
find "$TARGET_DIR" -maxdepth 1 -mindepth 1 -type d -print0 |
while IFS= read -r -d '' subdir; do
    # Try to get the birth time (%w). If %w is “-” (unknown), fall back to modification time (%y).
    btime=$(stat --format='%w' "$subdir")
    if [[ "$btime" == "-" ]]; then
        # creation not supported → use last-modified instead
        btime=$(stat --format='%y' "$subdir")
        echo "$subdir   (no birth time; using mtime) → $btime"
    else
        echo "$subdir   (birth) → $btime"
    fi
done