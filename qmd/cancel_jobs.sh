#!/bin/bash


####################################################################################################
# Summary:
# This script cancels a range of job IDs using the `scancel` command.
# Usage:
#   ./cancel_jobs.sh
#   - Modify the START_ID and END_ID variables to specify the range of job IDs to cancel.
#
# Author: akashgpt
####################################################################################################

# Define the start and end of the job ID range
START_ID=57912585
END_ID=57912700

# Loop through the range and cancel each job
for job_id in $(seq $START_ID $END_ID); do
    scancel $job_id
done
