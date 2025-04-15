#!/bin/bash

parent_dir=$(pwd)
parent_dir_name=$(basename "$parent_dir")


module purge
echo $ENV_for_MSD > setting_env.sh #for mda analysis and the "bc" command required for peavg.sh
source setting_env.sh
rm setting_env.sh


echo "################################"
echo "Running data_4_analysis.sh for $parent_dir_name"
# echo "Runtime: $runtime seconds"
echo "################################"

# figure out how long the scripts takes to run
# start=$(date +%s)  # Start time in seconds


MSD_python_file="${SCRATCH}/qmd_data/H2O_H2/sim_data_convergence/crystalline_or_not/run_scripts/msd_calc_v2.py"
ENV_for_MSD="module load anaconda3/2024.6; conda activate mda_env"


echo "Updating data for 'analysis/' ..."

mkdir -p analysis

grep "total pressure" OUTCAR | awk '{print $4}' > analysis/evo_total_pressure.dat
grep external OUTCAR | awk '{print $4}' > analysis/evo_external_pressure.dat
grep -a "volume of cell :" OUTCAR | awk '{print $5}' > analysis/evo_cell_volume.dat
sed -i '1,2d' analysis/evo_cell_volume.dat

grep "free  energy" OUTCAR | awk '{print $5}' > analysis/evo_free_energy.dat
grep ETOTAL OUTCAR | awk '{print $5}' > analysis/evo_total_energy.dat
grep "energy  without entropy" OUTCAR | awk '{print $4}' > analysis/evo_internal_energy.dat

grep "mean temperature" OUTCAR | awk '{print $5}' > analysis/evo_mean_temp.dat

source peavg.sh OUTCAR

# append a line to analysis/peavg_numbers.out with $parent_dir
echo "$parent_dir" > analysis/peavg_summary.out
# second and fourth line from peavg_numbers.out to analysis/peavg_summary.out
sed -n '1p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #TEMP
sed -n '4p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #NIONS
sed -n '24p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #Pressure
sed -n '25p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #Pressure error
sed -n '10p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #Internal energy
sed -n '11p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #Internal energy error
grep ENCUT $parent_dir/INCAR | awk '{print $3}' >> analysis/peavg_summary.out #ENCUT
grep GGA $parent_dir/INCAR | awk '{print $3}' >> analysis/peavg_summary.out #XC
grep "TITEL" $parent_dir/POTCAR | awk '{print $4}' >> analysis/peavg_summary.out #POTCAR 


# Diffusion calculcation
# cp $MSD_python_file .
# python msd_calc_v2.py
# module purge




# end=$(date +%s)  # End time in seconds
# runtime=$((end - start))
# runtime in proper format
# run_mins=$((runtime / 60))

echo "################################"
echo "Done with data_4_analysis.sh for $parent_dir_name"
# echo "Runtime: $runtime seconds"
echo "################################"
echo

module purge

#exit