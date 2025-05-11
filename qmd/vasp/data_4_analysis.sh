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


# Count the number of lines matching the two patterns
scaled_count=$(grep "SCALED FREE ENERGIE" OUTCAR | wc -l)
free_count=$(grep "free  energy" OUTCAR | wc -l)
half_free=$(echo "0.5 * $free_count" | bc)
# make half_free an integer
half_free=${half_free%.*}

# Define TI_mode: 1 if scaled_count equals half_free, 0 otherwise.
if [ "$scaled_count" -eq "$half_free" ]; then
    TI_mode=1
    # echo "TI_mode switched on."
else
    TI_mode=0
fi

echo "TI_mode is: $TI_mode" #; scaled_count is: $scaled_count, free_count is: $free_count, half_free is: $half_free"




grep "total pressure" OUTCAR | awk '{print $4}' > analysis/evo_total_pressure.dat
grep external OUTCAR | awk '{print $4}' > analysis/evo_external_pressure.dat
grep "kinetic pressure" OUTCAR | awk '{print $7}' > analysis/evo_kinetic_pressure.dat
grep "Pullay stress" OUTCAR | awk '{print $9}' > analysis/evo_pullay_stress.dat
grep -a "volume of cell :" OUTCAR | awk '{print $5}' > analysis/evo_cell_volume.dat
sed -i '1,2d' analysis/evo_cell_volume.dat

grep "free  energy" OUTCAR | awk '{print $5}' > analysis/evo_free_energy.dat
grep ETOTAL OUTCAR | awk '{print $5}' > analysis/evo_total_energy.dat
grep "free  energy   TOTEN" OUTCAR | awk '{print $5}' > analysis/evo_TOTEN.dat
grep "energy  without entropy" OUTCAR | awk '{print $4}' > analysis/evo_internal_energy.dat

# grep "mean temperature" OUTCAR | awk '{print $5}' > analysis/evo_mean_temp.dat
grep "(temperature" OUTCAR | sed -E 's/.*temperature[[:space:]]*([0-9]+\.[0-9]+).*/\1/' > analysis/evo_mean_temp.dat

# if TI_mode is 1, then
if [ "$TI_mode" -eq 1 ]; then
    awk 'NR%2==0' analysis/evo_internal_energy.dat > analysis/temp
    mv analysis/temp analysis/evo_internal_energy.dat
    awk 'NR%2==0' analysis/evo_free_energy.dat > analysis/temp
    mv analysis/temp analysis/evo_free_energy.dat
    awk 'NR%2==0' analysis/evo_TOTEN.dat > analysis/temp
    mv analysis/temp analysis/evo_TOTEN.dat
fi

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

axis_low_limit = 0.90
axis_high_limit = 1.10

# Load data from each file.
total_pressure = np.loadtxt("analysis/evo_total_pressure.dat")
external_pressure = np.loadtxt("analysis/evo_external_pressure.dat")
total_energy   = np.loadtxt("analysis/evo_total_energy.dat")
TOTEN        = np.loadtxt("analysis/evo_TOTEN.dat")
internal_energy = np.loadtxt("analysis/evo_internal_energy.dat")
free_energy   = np.loadtxt("analysis/evo_free_energy.dat")
cell_volume    = np.loadtxt("analysis/evo_cell_volume.dat")
mean_temp      = np.loadtxt("analysis/evo_mean_temp.dat")

# check if a ratio file exists
if os.path.exists("ratio"):
    ratio = np.loadtxt("ratio")
    print(f"Ratio file found: {ratio}")
else:
    print("No ratio file found. Using default value of 4")
    ratio = 4

# choose the last (1 - (1/ratio)) of the data
stat_total_pressure = total_pressure[int(len(total_pressure) * (1 - (1 / ratio))):]
stat_external_pressure = external_pressure[int(len(external_pressure) * (1 - (1 / ratio))):]
stat_total_energy = total_energy[int(len(total_energy) * (1 - (1 / ratio))):]
stat_TOTEN = TOTEN[int(len(TOTEN) * (1 - (1 / ratio))):]
stat_internal_energy = internal_energy[int(len(internal_energy) * (1 - (1 / ratio))):]
stat_free_energy = free_energy[int(len(free_energy) * (1 - (1 / ratio))):]
stat_cell_volume = cell_volume[int(len(cell_volume) * (1 - (1 / ratio))):]
stat_mean_temp = mean_temp[int(len(mean_temp) * (1 - (1 / ratio))):]


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

# TOTEN and total_energy
if len(TOTEN) > len(total_energy):
    TOTEN = TOTEN[:len(total_energy)]
elif len(total_energy) > len(TOTEN):
    total_energy = total_energy[:len(TOTEN)]

# pressure kBar to GPa
total_pressure = total_pressure * 0.1
external_pressure = external_pressure * 0.1
stat_total_pressure = stat_total_pressure * 0.1
stat_external_pressure = stat_external_pressure * 0.1

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
axs[0].axhline(np.mean(total_pressure), color='b', linestyle='--', label=f'Mean: {np.mean(stat_total_pressure):.2f} +/- {np.std(stat_total_pressure):.2f} GPa')
axs[0].set_ylabel('Total Pressure (GPa)')
leg = axs[0].legend(loc='upper left')
for text in leg.get_texts():
    text.set_color('b')
axs[0].grid()
axs[0].set_ylim(np.min(total_pressure)*axis_low_limit, np.max(total_pressure)*axis_high_limit)
# twinx axis for external pressure
ax1 = axs[0].twinx()
ax1.plot(time_steps_external_pressure, external_pressure, 'r-', alpha=0.5)
ax1.axhline(np.mean(external_pressure), color='r', linestyle='--', label=f'Mean: {np.mean(stat_external_pressure):.2f} +/- {np.std(stat_external_pressure):.2f} GPa')
ax1.set_ylabel('External Pressure (GPa)')
# color the axis red
# ax1.tick_params(axis='y', labelcolor='r')
leg = ax1.legend(loc='upper right')
for text in leg.get_texts():
    text.set_color('r')
ax1.set_ylim(np.min(external_pressure)*axis_low_limit, np.max(external_pressure)*axis_high_limit)


# Panel 2: evo_total_energy vs time-step
axs[1].plot(time_steps_energy, total_energy, 'g-', alpha=0.5)
axs[1].axhline(np.mean(total_energy), color='g', linestyle='--', label=f'Mean: {np.mean(stat_total_energy):.2f} +/- {np.std(stat_total_energy):.2f} eV')
# axs[1].plot(time_steps_energy, TOTEN, 'b-', alpha=0.5)
# axs[1].axhline(np.mean(TOTEN), color='b', linestyle='--', label=f'Mean: {np.mean(TOTEN):.2f} +/- {np.std(TOTEN):.2f} eV')
# axs[1].plot(time_steps_energy, free_energy, 'm:', label='Free Energy')
axs[1].set_ylabel('Total Energy (ETOTAL; eV)')
axs[1].grid()
leg = axs[1].legend(loc='upper left')
for text in leg.get_texts():
    text.set_color('g')
# axs[1].set_ylim(np.min(total_energy)*axis_low_limit, np.max(total_energy)*axis_high_limit)
# twinx axis for TOTEN
ax2 = axs[1].twinx()
ax2.plot(time_steps_energy, TOTEN, 'r-',alpha=0.5)
ax2.axhline(np.mean(TOTEN), color='r', linestyle='--', label=f'Mean: {np.mean(stat_internal_energy):.2f} +/- {np.std(stat_internal_energy):.2f} eV')
ax2.set_ylabel('TOTEN (El. Helmholtz free energy; eV)')
# color the axis red
# ax2.tick_params(axis='y', labelcolor='r')
leg = ax2.legend(loc='upper right')
for text in leg.get_texts():
    text.set_color('r')
# if np.max(TOTEN) > 0 and np.min(TOTEN) < 0:
#     ax2.set_ylim(np.min(TOTEN)*axis_high_limit, np.max(TOTEN)*axis_high_limit)
# elif np.max(TOTEN) > 0 and np.min(TOTEN) > 0:
#     ax2.set_ylim(np.min(TOTEN)*axis_low_limit, np.max(TOTEN)*axis_high_limit)
# elif np.max(TOTEN) < 0 and np.min(TOTEN) < 0:
#     ax2.set_ylim(np.min(TOTEN)*axis_high_limit, np.max(TOTEN)*axis_low_limit)
# else:
#     ax2.set_ylim(np.min(TOTEN)*axis_low_limit, np.max(TOTEN)*axis_high_limit)

# Panel 3: evo_cell_volume vs time-step
axs[2].plot(time_steps_volume, cell_volume, 'r-', alpha=0.5)
axs[2].axhline(np.mean(cell_volume), color='r', linestyle='--', label=f'Mean: {np.mean(stat_cell_volume):.2f} +/- {np.std(stat_cell_volume):.2f} Å³')
axs[2].set_ylabel('Cell Volume (Å³)')
axs[2].grid()
axs[2].legend()
# axs[2].set_ylim(np.min(cell_volume)*axis_low_limit, np.max(cell_volume)*axis_high_limit)

# Panel 4: evo_mean_temp vs time-step
axs[3].plot(time_steps_temp, mean_temp, 'm-', alpha=0.5)
axs[3].axhline(np.mean(mean_temp), color='m', linestyle='--', label=f'Mean: {np.mean(stat_mean_temp):.2f} +/- {np.std(stat_mean_temp):.2f} K')
axs[3].set_xlabel('Time-step')
axs[3].set_ylabel('Temperature (K)')
axs[3].legend()
axs[3].grid()
# axs[3].set_ylim(np.min(mean_temp)*axis_low_limit, np.max(mean_temp)*axis_high_limit)

# plot title
plt.suptitle(f"Analysis of VASP Simulation Data: {os.path.basename(current_dir)} (ratio: {ratio})", fontsize=12)

# Improve layout to prevent overlapping labels
plt.tight_layout()
# plt.show()
# Save the figure to a file
plt.savefig("analysis/plot_evo_data.png", dpi=300)


# create a log file with all means
with open("analysis/log.plot_evo_data", "w") as log_file:
    log_file.write(f"Mean Total Pressure: {np.mean(stat_total_pressure):.2f} +/- {np.std(stat_total_pressure):.2f} GPa\n")
    log_file.write(f"Mean External Pressure: {np.mean(stat_external_pressure):.2f} +/- {np.std(stat_external_pressure):.2f} GPa\n")
    log_file.write(f"Mean Total Energy: {np.mean(stat_total_energy):.2f} +/- {np.std(stat_total_energy):.2f} eV\n")
    log_file.write(f"Mean Internal Energy: {np.mean(stat_internal_energy):.2f} +/- {np.std(stat_internal_energy):.2f} eV\n")
    log_file.write(f"Mean Free Energy: {np.mean(stat_free_energy):.2f} +/- {np.std(stat_free_energy):.2f} eV\n")
    log_file.write(f"Mean Cell Volume: {np.mean(stat_cell_volume):.2f} +/- {np.std(stat_cell_volume):.2f} Å³\n")
    log_file.write(f"Mean Temperature: {np.mean(stat_mean_temp):.2f} +/- {np.std(stat_mean_temp):.2f} K\n")

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