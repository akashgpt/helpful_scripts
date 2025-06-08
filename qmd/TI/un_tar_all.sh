#!/usr/bin/env bash

# This script is used to untar all `.tar.*` files found in the current directory and its subdirectories.
# The names of the extracted directories will be derived from the tar file names by removing the first ".tar" and anything that follows.
# It will also remove the original tar files after extraction.
# un_tar_all.sh
#
# Usage: nohup $HELP_SCRIPTS_TI/un_tar_all.sh > log.un_tar_all 2>&1 &


set -euo pipefail
echo "Starting un-tar process at $(date)"
echo "Current directory: $(pwd)"
echo ""

counter=0
find . -type f -name '*.tar.*' -print0 |
while IFS= read -r -d '' archive; do
    parent="$(dirname "$archive")"
    base="$(basename "$archive")"
    # Remove the first “.tar” and anything that follows, e.g. “foo.tar.gz” → “foo”
    prefix="${base%%.tar.*}"
    target="${parent}/${prefix}"

    echo "Extracting $archive → $target/"
    mkdir -p "$target"
    tar -xf "$archive" -C "$target"

    # # remove the archive after extraction
    # echo "Removing archive: $archive"
    # rm -f "$archive"
    # echo "Done extracting $archive"
    
    counter=$((counter + 1))
    echo "Counter: $counter"
    echo ""

done

echo ""
echo "Un-tar-ing completed at $(date)"