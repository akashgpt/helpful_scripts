#!/bin/bash

# # SBATCH --job-name=msd_calc      # create a short name for your job
# # SBATCH --nodes=1                # node count
# # SBATCH --ntasks-per-node=32      # total number of tasks across all nodes
# # SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
# # SBATCH --mem=4G                # memory per cpu-core (4G is default)
# # SBATCH --time=1:00:00           # total run time limit (HH:MM:SS)
# # #SBATCH --mail-type=begin       # send email when job begins
# # #SBATCH --mail-type=end         # send email when job ends
# # #SBATCH --mail-user=ag5805@princeton.edu
# # SBATCH --exclude=della-r4c[1-4]n[1-16],della-r1c[3,4]n[1-16]
# ~# ##

####################################################################################################
# go to each folder in the immediate parent directory, copy-paste data_4_analysis.sh from run_scripts, and then source this script

# Define the source script path
SOURCE_SCRIPTD_DIR="/scratch/gpfs/ag5805/qmd_data/H2O_H2/sim_data_convergence/crystalline_or_not/run_scripts/"
SOURCE_SCRIPT="${SOURCE_SCRIPTD_DIR}data_4_analysis.sh"
DATA_DIR=$SOURCE_SCRIPTD_DIR/..

# Check if the source script exists
if [ ! -f "$SOURCE_SCRIPT" ]; then
    echo "Error: $SOURCE_SCRIPT does not exist."
    # exit 1
fi

cd $DATA_DIR
# Loop over each directory that starts with "a0" in the immediate parent directory
for dir in a0*/; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        dir_name=$(basename $dir)
        echo "Processing $dir_name ..."
        
        # Copy the script into the directory
        cp $SOURCE_SCRIPT $dir/
        
        # Navigate into the directory, source the script, then return to the parent directory
        (
            cd "$dir"
            if [ ! -f "OUTCAR" ]; then
                chmod +x data_4_analysis.sh

                # check if "MSD_combined.jpg" exists, if not, then run the script otherwise skip
                if [ ! -f "MSD_combined.jpg" ]; then
                    source data_4_analysis.sh
                fi
            fi
            cd /scratch/gpfs/ag5805/qmd_data/H2O_H2/sim_data/
        )
        
        echo "Sourced data_4_analysis.sh in $dir_name"
    fi
done


# # a0621
# # a0623
# # a0625e
# # a0631
# # a0632
# # a0633
# # a0635

# # a0651a
# # a0652
# # a0653-5

# runid=a0401
# cd ../a0${runid} && source data_4_analysis.sh  > log.data_4_analysis 2>&1 &