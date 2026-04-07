# Define the main project directory
AG_GLOBAL = "/projects/BURROWS/akashgpt"

# Import essential libraries
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from MDAnalysis.analysis import msd
from scipy.stats import linregress
import csv

# Add paths for additional modules
sys.path.insert(1, "/projects/BURROWS/akashgpt/misc_libraries/vatic-master")

# Import custom modules from vatic
from vatic_others import XDATCAR_toolkit as xdt
from vatic_interpy import write_pdb

# MDAnalysis and pymatgen for molecular dynamics and materials science analysis
import MDAnalysis as mda
from MDAnalysis.analysis import rdf, rms, distances
from pymatgen.io.vasp import Poscar, Incar, Xdatcar
from pymatgen.core import trajectory
from pymatgen.analysis.eos import EOS, BirchMurnaghan

# Physical constants for vibrational periods (in ps)
c_light = 299792458
H_H_vibrational_period = 1E-3 * 1E15 / (c_light * 4342 * 1E2)  # >> 7.68 fs
O_H_vibrational_period = 1E-3 * 1E15 / (c_light * 3506 * 1E2)  # >> 9.51 fs
O_O_vibrational_period = 1E-3 * 1E15 / (c_light * 2061 * 1E2)  # >> 16.18 fs

# Get current working directory
curr_dir = os.getcwd()
print(f"Current working directory: {curr_dir}")

# Load structural information from VASP input files
poscar = Poscar.from_file('POSCAR')
incar = Incar.from_file('INCAR')
a_box_dimn, b_box_dimn, c_box_dimn = poscar.structure.lattice.a, poscar.structure.lattice.b, poscar.structure.lattice.c

# Generate trajectory file using XDATCAR toolkit
dt_sim = incar['POTIM'] * 1.0E-3  # Set timestep in ps

# Create MDAnalysis universes for wrapped and unwrapped trajectories
os.system(f'python {AG_GLOBAL}/misc_libraries/XDATCAR_toolkit/XDATCAR_toolkit.py -p -t {incar["POTIM"]}')
u = mda.Universe("XDATCAR.pdb", dt=dt_sim)
# os.system(f'python {AG_GLOBAL}/misc_libraries/XDATCAR_toolkit/XDATCAR_toolkit.py -p -t {incar["POTIM"]} --pbc')
# u_unwrapped = mda.Universe("XDATCAR.pdb", dt=dt_sim)

################################################################
u_ref = u#u_unwrapped
################################################################

# Parse analysis data
runID_dir_analysis = os.path.join(os.getcwd(), "analysis/")
with open(runID_dir_analysis + 'peavg_numbers.out') as f1:
    peavg_numbers = np.array([float(line) if line else -np.inf for line in f1.read().splitlines()])
run_pressure, run_timesteps = peavg_numbers[23], peavg_numbers[31]

# Select atom types
type_O, type_H = "type O", "type H"
O_atoms = u_ref.select_atoms(type_O)
H_atoms = u_ref.select_atoms(type_H)

# Simulation parameters for plot titles
run_temperature = incar['TEBEG']
run_title_4_figures = f': t = {int(run_timesteps)}, T = {run_temperature}K, P = {str(run_pressure)[:5]}GPa, N={{ {len(O_atoms)} H₂O, {int((len(H_atoms) - 2 * len(O_atoms)) / 2)} H₂ }}'

# Select all frames
t_select_bool = np.full(len(u_ref.trajectory), False)
t_beg, t_end = -1, len(u_ref.trajectory)
t_select_bool[t_beg:t_end] = True

# Extract unique atom types and format for MSD calculation
atom_types = np.unique(u_ref.atoms.types)
str_atom_types = ["type " + atype for atype in atom_types]
print(str_atom_types)

# Colors for different atom types in the MSD plot
colors = ['teal', 'darkorange', 'purple', 'blue', 'red']




################################################################
################################################################
#time-averaged histograms
################################################################
################################################################

hist_num_bins = 100
vol_block = a_box_dimn*b_box_dimn*c_box_dimn/hist_num_bins
hist_bins = np.linspace(0,c_box_dimn,hist_num_bins)

X_atoms = O_atoms
Y_atoms = H_atoms

X_hist_x = np.zeros(shape=(len(hist_bins)-1))
X_hist_y = np.zeros(shape=(len(u.trajectory), len(hist_bins)-1))
Y_hist_x = np.zeros(shape=(len(hist_bins)-1))
Y_hist_y = np.zeros(shape=(len(u.trajectory), len(hist_bins)-1))


for i2 in range(len(u.trajectory)):
    u.trajectory[i2].frame

    hist, bin_edges = np.histogram(X_atoms.positions[:,2], bins=hist_bins)
    for i1 in range(len(hist_bins)-1):
        X_hist_y[i2,i1] = hist[i1]
        if (i2 ==0):
            X_hist_x[i1] = 0.5*(bin_edges[i1] + bin_edges[i1+1])

    hist, bin_edges = np.histogram(Y_atoms.positions[:,2], bins=hist_bins)
    for i1 in range(len(hist_bins)-1):
        Y_hist_y[i2,i1] = hist[i1]
        if (i2 ==0):
            Y_hist_x[i1] = 0.5*(bin_edges[i1] + bin_edges[i1+1])



file_2log = open("hist_time_averaged.csv", 'w') 

csv_writer = csv.writer(file_2log, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

fields = ['hist_x','hist_y_O','hist_y_H']
data_output = np.transpose(np.vstack([X_hist_x, X_hist_y, Y_hist_y]))

# for i1 in range(len(wc_field_density_points_vector)):
csv_writer.writerow(fields) 
csv_writer.writerows(data_output)

file_2log.close()       


################################################################
# amount of species
################################################################
time_step = 0
time_step_block = len(u.trajectory) - time_step #in steps

fig, ax = plt.subplots(figsize=(14, 6))

hist_y_time_step_block = np.sum(X_hist_y[time_step:time_step+time_step_block,:], axis=0)/time_step_block**1
plt.plot(X_hist_x,hist_y_time_step_block/vol_block,color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=10, label='O atom')

hist_y_time_step_block = np.sum(Y_hist_y[time_step:time_step+time_step_block,:], axis=0)/time_step_block**1
plt.plot(Y_hist_x,hist_y_time_step_block/vol_block,color='red', marker='o', linestyle='dashed', linewidth=2, markersize=10, label='H atom')

plt.xlabel("z (Å)")
plt.ylabel("# per unit volume")

plt.legend(loc="best")
plt.title(run_title_4_figures+"; Histogram - O, H atoms for time-steps=["+str(time_step)+':'+str(time_step+time_step_block)+']')
plt.xlim([0,c_box_dimn])
plt.grid(True, linestyle='--', alpha=0.7)
fig.savefig("amount_species_hist_time_averaged.jpg", dpi=300, bbox_inches='tight')        



################################################################
#ratio of species
################################################################
time_step = 0
time_step_block = len(u.trajectory) - time_step #in steps
fig, ax = plt.subplots(figsize=(14, 6))

X_hist_y_time_step_block = np.sum(X_hist_y[time_step:time_step+time_step_block,:], axis=0)/time_step_block**1
Y_hist_y_time_step_block = np.sum(Y_hist_y[time_step:time_step+time_step_block,:], axis=0)/time_step_block**1
plt.plot(X_hist_x,X_hist_y_time_step_block/(X_hist_y_time_step_block + Y_hist_y_time_step_block),color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=10, label='O/(O+H)')

plt.xlabel("z (Å)")
plt.ylabel("O/(O+H)")

plt.legend(loc="best")
plt.title(run_title_4_figures+": Histogram - O, H atoms for time-steps=["+str(time_step)+':'+str(time_step+time_step_block)+']')
plt.xlim([0,c_box_dimn])
plt.grid(True, linestyle='--', alpha=0.7)
fig.savefig("ratio_species_hist_time_averaged.jpg", dpi=300, bbox_inches='tight')        


