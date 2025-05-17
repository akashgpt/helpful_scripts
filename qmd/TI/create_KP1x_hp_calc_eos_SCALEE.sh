#!/usr/bin/env bash
shopt -s nullglob   # so that `for dir in */` does nothing if there are no folders

# Usage: source $HELP_SCRIPTS_TI/create_KP1x_hp_calc_eos_SCALEE.sh > log.create_KP1x_hp_calc_eos_SCALEE 2>&1 &

# for each folder in the current directory, except, master_setup_TI, go inside and run source $HELP_SCRIPTS_TI/calculate_GFE_v2.sh 1 2 > log.calculate_GFE 2>&1 &

# Save the current working directory for later
PT_dir=$(pwd)
PT_dir_name=$(basename "$PT_dir")

COMPOSITION_dir=$(dirname "$PT_dir")
COMPOSITION_dir_name=$(basename "$COMPOSITION_dir")

echo "Current time: $(date)"
echo "Current PT directory: $PT_dir"
echo "Current PT directory name: $PT_dir_name"
echo "Current COMPOSITION directory: $COMPOSITION_dir"
echo "Current COMPOSITION directory name: $COMPOSITION_dir_name"
echo ""


for dir in */; do
    # skip the master_setup_TI folder
    [[ "$dir" == "master_setup_TI/" ]] && continue

    (
        cd "$dir" || exit 1
        dir_address=$(pwd)
        # source the helper script with args "1 2",
        # redirect both stdout+stderr into the log file,
        # and put it in the background.
        source "$HELP_SCRIPTS_TI/calculate_GFE_v2.sh" 1 2 > log.calculate_GFE 2>&1 &
        echo "Sourcing calculate_GFE_v2.sh in $dir_address"
        echo ""
        cd "$PT_dir" || exit 1
    )
done

echo ""
echo "All calculations started in the background by $(date)."
echo ""