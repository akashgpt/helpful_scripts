#!/usr/bin/env bash



# for all KP1 folders in BACKUP_DIR="/tigerdata/burrows/planet_evo/akashgpt/qmd_data_backup", with local address $LOCAL_KP1_PATH (needs to be found) and thus a global address BACKUP_KP1_DIR=${BACKUP_DIR}${LOCAL_KP1_PATH}, 
# copy BACKUP_KP1_DIR to ${HOME_DIR}${LOCAL_KP1_PATH} where HOME_DIR="/scratch/gpfs/BURROWS/akashgpt/qmd_data"
#
# Usage: nohup $HELP_SCRIPTS_TI/backup_files.sh > log.backup_files 2>&1 &

set -euo pipefail

# Base paths
BACKUP_DIR="/tigerdata/burrows/planet_evo/akashgpt/qmd_data_backup/Fe_MgSiO3_H"
HOME_DIR="/scratch/gpfs/BURROWS/akashgpt/qmd_data/Fe_MgSiO3_H"

# Find every “KP1” folder under BACKUP_DIR, determine its path relative to BACKUP_DIR,
# then copy that entire KP1 folder into the analogous location under HOME_DIR.

counter=0
echo "Starting KP1 backup process at $(date)"
echo "Current directory: $(pwd)"
echo ""

find "$BACKUP_DIR" -type d -name "KP1" -print0 |
while IFS= read -r -d '' backup_kp1_dir; do
    # Derive the local path (everything after BACKUP_DIR), e.g.:
    #   backup_kp1_dir="/tigerdata/.../foo/bar/KP1"
    #   local_path="/foo/bar/KP1"
    local_path="${backup_kp1_dir#$BACKUP_DIR}"
    echo "Found local KP1 directory: $local_path"
    # The parent folder under HOME_DIR where “KP1” should live:
    #   dest_parent="/scratch/.../foo/bar"
    dest_parent="$HOME_DIR$(dirname "$local_path")"

    # Skip if isobar_calc is in the path
    if [[ "$local_path" == *"isobar_calc"* ]]; then
        echo "Skipping directory: $backup_kp1_dir (contains isobar_calc)"
        echo ""
        continue
    fi

    echo "Copying $backup_kp1_dir → $dest_parent/"

    # # Ensure the parent exists
    mkdir -p "$dest_parent"

    # # Copy the entire “KP1” directory (and its contents) into dest_parent
    cp -r "$backup_kp1_dir" "$dest_parent"
    echo "Done copying $backup_kp1_dir to $dest_parent"

    # confirm that the directory sizes are the same
    backup_size=$(du -sh "$backup_kp1_dir" | cut -f1)
    dest_size=$(du -sh "$dest_parent/KP1" | cut -f1)
    if [[ "$backup_size" == "$dest_size" ]]; then
        echo "Sizes match: $backup_size"
    else
        echo "*********************************************"
        echo "WARNING: Sizes do not match: $backup_size vs $dest_size"
        echo "*********************************************"
    fi

    counter=$((counter + 1))
    echo "Counter: $counter"
    echo ""

done

echo ""
echo "KP1 Backup process completed at $(date)"
























echo ""
echo "======================================================="
echo "======================================================="
echo "======================================================="
echo ""




counter=0
echo "Starting hp_calculations backup process at $(date)"
echo "Current directory: $(pwd)"
echo ""

find "$BACKUP_DIR" -type d -name "hp_calculations" -print0 |
while IFS= read -r -d '' backup_hp_dir; do
    # Derive the local path (everything after BACKUP_DIR), e.g.:
    #   backup_hp_dir="/tigerdata/.../foo/bar/hp_calculations"
    #   local_path="/foo/bar/hp_calculations"
    local_path="${backup_hp_dir#$BACKUP_DIR}"
    echo "Found local hp_calculations directory: $local_path"
    # The parent folder under HOME_DIR where “hp_calculations” should live:
    #   dest_parent="/scratch/.../foo/bar"
    dest_parent="$HOME_DIR$(dirname "$local_path")"

    # Skip if isobar_calc is in the path
    if [[ "$local_path" == *"isobar_calc"* ]]; then
        echo "Skipping directory: $backup_hp_dir (path contains isobar_calc)"
        echo ""
        continue
    fi

    # Skip if SCALEE_1 is not in the path
    if [[ "$local_path" != *"SCALEE_1"* ]]; then
        echo "Skipping directory: $backup_hp_dir (path does not contain SCALEE_1)"
        echo ""
        continue
    fi

    echo "Copying $backup_hp_dir → $dest_parent/"

    # # Ensure the parent exists
    mkdir -p "$dest_parent"

    # # Copy the entire “hp_calculations” directory (and its contents) into dest_parent
    cp -r "$backup_hp_dir" "$dest_parent"
    echo "Done copying $backup_hp_dir to $dest_parent"

    # confirm that the directory sizes are the same
    backup_size=$(du -sh "$backup_hp_dir" | cut -f1)
    dest_size=$(du -sh "$dest_parent/hp_calculations" | cut -f1)
    if [[ "$backup_size" == "$dest_size" ]]; then
        echo "Sizes match: $backup_size"
    else
        echo "*********************************************"
        echo "WARNING: Sizes do not match: $backup_size vs $dest_size"
        echo "*********************************************"
    fi

    counter=$((counter + 1))
    echo "Counter: $counter"
    echo ""

done

echo ""
echo "hp_calculations Backup process completed at $(date)"