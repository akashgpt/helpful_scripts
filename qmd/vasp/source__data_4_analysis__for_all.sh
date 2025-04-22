#!/bin/bash

# Summary:
# This script sources the data_4_analysis.sh script in each directory in the current directory.

source_sequentially=${1:-0} # 0: source in parallel, 1: source sequentially
update_data_4_analysis_script=${2:-1} # 0: do not update the script, 1: update the script


for dir in */; do 

    # if directory does not have have OUTCAR, echo warning
    if [ ! -f "$dir/OUTCAR" ]; then
        echo ""
        echo "################################"
        echo "Warning: OUTCAR not found in $dir. Skipping this directory."
        echo "################################"
        echo ""
        continue
    fi
    echo ""
    echo "Processing $dir"
    cd "$dir"
    
    if [ $update_data_4_analysis_script -eq 0 ]; then
        echo "Not updating data_4_analysis.sh script from $HELP_SCRIPTS/qmd/vasp/"
    else
        # update the data_4_analysis.sh script from $HELP_SCRIPTS/qmd/vasp/
        cp $HELP_SCRIPTS/qmd/vasp/data_4_analysis.sh .
    fi

    if [ $source_sequentially -eq 1 ]; then
        echo "Sourcing data_4_analysis.sh sequentially"
        source data_4_analysis.sh
    else
        echo "Sourcing data_4_analysis.sh in parallel"
        source data_4_analysis.sh &
    fi

    echo ""
    cd ..
    
done