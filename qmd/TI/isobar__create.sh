#!/bin/bash
# isobar__create.sh

# Usage: source $HELP_SCRIPTS_TI/isobar__create.sh > log.isobar__create 2>&1 &
#        nohup $HELP_SCRIPTS_TI/isobar__create.sh > log.isobar__create 2>&1 &

ISOBAR_CALC_dir=$(pwd)

# if this directory is not isobar_calc, then error and exit
if [[ $(basename "$ISOBAR_CALC_dir") != isobar_calc ]]; then
    printf 'Error: run this script from "isobar_calc" (current: %s)\n' \
        "$ISOBAR_CALC_dir" >&2
    exit 1
fi

echo "Current time: $(date)"
echo "ISOBAR_CALC_dir: $ISOBAR_CALC_dir"
echo ""

source $HELP_SCRIPTS_TI/isobar__create_KP1.sh > log.isobar__create_KP1 2>&1

sleep 3600 # wait for 1 hour before proceeding to the next step

source $HELP_SCRIPTS_TI/isobar__create_KP1x.sh > log.isobar__create_KP1x 2>&1

sleep 3600 # wait for 1 hour before proceeding to the next step

source $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc.sh > log.isobar__create_KP1x_hp_calc 2>&1

