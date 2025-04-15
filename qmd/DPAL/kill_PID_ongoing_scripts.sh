#!/bin/bash

################################################################################
# Summary:
# This script terminates processes whose process IDs (PIDs) are listed in the file
# "PID_ongoing_scripts". It uses xargs to pass each PID from the file to the kill command.
#
# Usage:
#   1. Ensure that the file "PID_ongoing_scripts" exists in the current directory.
#      The file should contain one PID per line.
#   2. Run this script to kill the processes corresponding to those PIDs.
#
# Note: This script does not cancel any pending SLURM jobs.
#
# Author: akashgpt
################################################################################

xargs kill < PID_ongoing_scripts