#!/bin/bash


# navigate to all isobar_calc directories and run script X.sh
# Usage: nohup $HELP_SCRIPTS_TI/run_isobar__create.sh > log.run_isobar__create 2>&1 &

home_dir=$(pwd)
echo ""
echo "Current directory: $home_dir"
echo "Current time: $(date)"
echo ""

counter=0

find . -type d -name "isobar_calc" | while read dir; do
    echo "Running script in $dir"

    (
        cd "$dir" 

        # nohup bash $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos.sh > log.isobar__create_KP1x_hp_calc_eos 2>&1 &
        
        # to not run SCALEE sims
        nohup bash $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos_SCALEE.sh 0 > log.isobar__create_KP1x_hp_calc_eos_SCALEE 2>&1 &

        # to run SCALEE sims
        # nohup bash $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos_SCALEE.sh 1 > log.isobar__create_KP1x_hp_calc_eos_SCALEE 2>&1 &

        echo "Started isobar__create_ ... .sh in $dir"
        echo "Process ID: $!"
        counter=$((counter + 1))
        echo "Running counter: $counter"
        echo ""

        cd "$home_dir" || exit 1
    )

    sleep 1 # wait for 1 minute before proceeding to the next directory
done

echo "All scripts started. Check individual log files in each isobar_calc directory."
echo "Current time: $(date)"