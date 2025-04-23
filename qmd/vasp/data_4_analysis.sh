#!/bin/bash

#############################################################
# Summary: This script is used to analyze the output of VASP simulations.
# It extracts relevant data from the OUTCAR file, performs calculations, 
# and generates plots for analysis.
#
# Usage: source data_4_analysis.sh
#
# Author: Akash Gupta
#############################################################

parent_dir=$(pwd)
parent_dir_name=$(basename "$parent_dir")


# module purge
# echo $ENV_for_MSD > setting_env.sh #for mda analysis and the "bc" command required for peavg.sh
# source setting_env.sh
# rm setting_env.sh
# MSD_python_file="${SCRATCH}/qmd_data/H2O_H2/sim_data_convergence/crystalline_or_not/run_scripts/msd_calc_v2.py"
# ENV_for_MSD="module load anaconda3/2024.6; conda activate mda_env"


echo "################################"
echo "Running data_4_analysis.sh for $parent_dir_name"
# echo "Runtime: $runtime seconds"
echo "################################"

# figure out how long the scripts takes to run
# start=$(date +%s)  # Start time in seconds





echo "Updating data for 'analysis/' ..."

mkdir -p analysis

grep "total pressure" OUTCAR | awk '{print $4}' > analysis/evo_total_pressure.dat
grep external OUTCAR | awk '{print $4}' > analysis/evo_external_pressure.dat
grep "kinetic pressure" OUTCAR | awk '{print $7}' > analysis/evo_kinetic_pressure.dat
grep "Pullay stress" OUTCAR | awk '{print $9}' > analysis/evo_pullay_stress.dat
grep -a "volume of cell :" OUTCAR | awk '{print $5}' > analysis/evo_cell_volume.dat
sed -i '1,2d' analysis/evo_cell_volume.dat

grep "free  energy" OUTCAR | awk '{print $5}' > analysis/evo_free_energy.dat
grep ETOTAL OUTCAR | awk '{print $5}' > analysis/evo_total_energy.dat
grep "energy  without entropy" OUTCAR | awk '{print $4}' > analysis/evo_internal_energy.dat

# grep "mean temperature" OUTCAR | awk '{print $5}' > analysis/evo_mean_temp.dat
grep "(temperature" OUTCAR | sed -E 's/.*temperature[[:space:]]*([0-9]+\.[0-9]+).*/\1/' > analysis/evo_mean_temp.dat


echo "Sourcing peavg.sh"
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



######################################
# echo "Running MSD calculation ..."
echo "Diffusion calculation deactivated."
# Diffusion calculcation
# cp $MSD_python_file .
# python msd_calc_v2.py
# module purge
######################################




######################################
echo "Plotting some relevant data."
# call python to create a plot of the following in 1 figure, 4 X 1 panels
# 1. data in evo_total_pressure vs time-step
# 2. data in evo_total_energy vs time-step
# 3. data in evo_cell_volume vs time-step
# 4. data in evo_mean_temp vs time-step
python << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.getcwd()
analysis_dir = os.path.join(current_dir, "analysis")

# Load data from each file.
total_pressure = np.loadtxt("analysis/evo_total_pressure.dat")
external_pressure = np.loadtxt("analysis/evo_external_pressure.dat")
total_energy   = np.loadtxt("analysis/evo_total_energy.dat")
internal_energy = np.loadtxt("analysis/evo_internal_energy.dat")
free_energy   = np.loadtxt("analysis/evo_free_energy.dat")
cell_volume    = np.loadtxt("analysis/evo_cell_volume.dat")
mean_temp      = np.loadtxt("analysis/evo_mean_temp.dat")

# make internal_energy and total_energy the same length as each other and get ride of the extra lines whichever has it
if len(internal_energy) > len(total_energy):
    internal_energy = internal_energy[:len(total_energy)]
elif len(total_energy) > len(internal_energy):
    total_energy = total_energy[:len(internal_energy)]

# same with free_energy and total_energy
if len(free_energy) > len(total_energy):
    free_energy = free_energy[:len(total_energy)]
elif len(total_energy) > len(free_energy):
    total_energy = total_energy[:len(free_energy)]

# pressure kBar to GPa
total_pressure = total_pressure * 0.1
external_pressure = external_pressure * 0.1

# Create a time-step array based on the number of data points.
time_steps_pressure = np.arange(1, len(total_pressure) + 1)
time_steps_external_pressure = np.arange(1, len(external_pressure) + 1)
time_steps_energy = np.arange(1, len(total_energy) + 1)
time_steps_volume = np.arange(1, len(cell_volume) + 1)
time_steps_temp = np.arange(1, len(mean_temp) + 1)

# Create a figure with 4 vertical subplots.
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 12))
fig.subplots_adjust(hspace=0.5)

# Panel 1: evo_total_pressure vs time-step
axs[0].plot(time_steps_pressure, total_pressure, 'b-', alpha=0.5)
axs[0].axhline(np.mean(total_pressure), color='b', linestyle='--', label=f'Mean: {np.mean(total_pressure):.2f} +/- {np.std(total_pressure):.2f} GPa')
axs[0].set_ylabel('Total Pressure (GPa)')
leg = axs[0].legend(loc='upper left')
for text in leg.get_texts():
    text.set_color('b')
axs[0].grid()
# twinx axis for external pressure
ax1 = axs[0].twinx()
ax1.plot(time_steps_external_pressure, external_pressure, 'r-', alpha=0.5)
ax1.axhline(np.mean(external_pressure), color='r', linestyle='--', label=f'Mean: {np.mean(external_pressure):.2f} +/- {np.std(external_pressure):.2f} GPa')
ax1.set_ylabel('External Pressure (GPa)')
# color the axis red
# ax1.tick_params(axis='y', labelcolor='r')
leg = ax1.legend(loc='upper right')
for text in leg.get_texts():
    text.set_color('r')


# Panel 2: evo_total_energy vs time-step
axs[1].plot(time_steps_energy, total_energy, 'g-', alpha=0.5)
axs[1].axhline(np.mean(total_energy), color='g', linestyle='--', label=f'Mean: {np.mean(total_energy):.2f} +/- {np.std(total_energy):.2f} eV')
# axs[1].plot(time_steps_energy, free_energy, 'm:', label='Free Energy')
axs[1].set_ylabel('Total Energy (ETOTAL; eV)')
axs[1].grid()
leg = axs[1].legend(loc='upper left')
for text in leg.get_texts():
    text.set_color('g')
# twinx axis for internal energy
ax2 = axs[1].twinx()
ax2.plot(time_steps_energy, internal_energy, 'r-',alpha=0.5)
ax2.axhline(np.mean(internal_energy), color='r', linestyle='--', label=f'Mean: {np.mean(internal_energy):.2f} +/- {np.std(internal_energy):.2f} eV')
ax2.set_ylabel('Internal Energy (energy without entropy; eV)')
# color the axis red
# ax2.tick_params(axis='y', labelcolor='r')
leg = ax2.legend(loc='upper right')
for text in leg.get_texts():
    text.set_color('r')

# Panel 3: evo_cell_volume vs time-step
axs[2].plot(time_steps_volume, cell_volume, 'r-', alpha=0.5)
axs[2].axhline(np.mean(cell_volume), color='r', linestyle='--', label=f'Mean: {np.mean(cell_volume):.2f} +/- {np.std(cell_volume):.2f} Å³')
axs[2].set_ylabel('Cell Volume (Å³)')
axs[2].grid()
axs[2].legend()

# Panel 4: evo_mean_temp vs time-step
axs[3].plot(time_steps_temp, mean_temp, 'm-', alpha=0.5)
axs[3].axhline(np.mean(mean_temp), color='m', linestyle='--', label=f'Mean: {np.mean(mean_temp):.2f} +/- {np.std(mean_temp):.2f} K')
axs[3].set_xlabel('Time-step')
axs[3].set_ylabel('Temperature (K)')
axs[3].legend()
axs[3].grid()


# Improve layout to prevent overlapping labels
plt.tight_layout()
# plt.show()
# Save the figure to a file
plt.savefig("analysis/plot_evo_data.png", dpi=300)

# create a log file with all means
with open("analysis/log.plot_evo_data", "w") as log_file:
    log_file.write(f"Mean Total Pressure: {np.mean(total_pressure):.2f} +/- {np.std(total_pressure):.2f} GPa\n")
    log_file.write(f"Mean External Pressure: {np.mean(external_pressure):.2f} +/- {np.std(external_pressure):.2f} GPa\n")
    log_file.write(f"Mean Total Energy: {np.mean(total_energy):.2f} +/- {np.std(total_energy):.2f} eV\n")
    log_file.write(f"Mean Internal Energy: {np.mean(internal_energy):.2f} +/- {np.std(internal_energy):.2f} eV\n")
    log_file.write(f"Mean Free Energy: {np.mean(free_energy):.2f} +/- {np.std(free_energy):.2f} eV\n")
    log_file.write(f"Mean Cell Volume: {np.mean(cell_volume):.2f} +/- {np.std(cell_volume):.2f} Å³\n")
    log_file.write(f"Mean Temperature: {np.mean(mean_temp):.2f} +/- {np.std(mean_temp):.2f} K\n")

EOF




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