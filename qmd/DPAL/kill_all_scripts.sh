#!/bin/bash

################################################################################
# Summary:
# This script stops running training jobs by killing processes that match specific
# training script names and canceling any pending SLURM jobs for the current user.
#
# Usage:
#   ./stop_training.sh
#
# It uses 'pkill' with the '-f' option to target processes whose command lines
# contain the given script names, and 'scancel' to cancel all SLURM jobs belonging
# to the current user.
################################################################################

pkill -f TRAIN_MLMD_LEVEL_1__TEST.sh  # Kill processes running TRAIN_MLMD_LEVEL_1__TEST.sh
pkill -f TRAIN_MLMD_LEVEL_1.sh          # Kill processes running TRAIN_MLMD_LEVEL_1.sh
pkill -f TRAIN_MLMD_LEVEL_2.sh          # Kill processes running TRAIN_MLMD_LEVEL_2.sh
pkill -f TRAIN_MLMD_LEVEL_3.sh          # Kill processes running TRAIN_MLMD_LEVEL_3.sh
pkill -f TRAIN_MLMD_LEVEL_4.sh          # Kill processes running TRAIN_MLMD_LEVEL_4.sh

scancel -u $USER                       # Cancel all SLURM jobs for the current user