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

# name of curr_dir
runID = curr_dir.split('/')[-1]


with open(runID_dir_analysis + 'peavg_numbers.out') as f1:
    peavg_numbers = np.array([float(line) if line else -np.inf for line in f1.read().splitlines()])
run_pressure, run_timesteps = peavg_numbers[23], peavg_numbers[31]

# Select atom types
type_O, type_H = "type O", "type H"
O_atoms = u_ref.select_atoms(type_O)
H_atoms = u_ref.select_atoms(type_H)

# Simulation parameters for plot titles
run_temperature = incar['TEBEG']
run_title_4_figures = f'{runID}: t = {int(run_timesteps)}, T = {run_temperature}K, P = {str(run_pressure)[:5]}GPa, N={{ {len(O_atoms)} H₂O, {int((len(H_atoms) - 2 * len(O_atoms)) / 2)} H₂ }}'

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
# radial distribution function
################################################################
################################################################

nbins_rdf = 100
rdf_range_max = min(u.dimensions[:3])/2.0

rdf_vasp_H_O = rdf.InterRDF(H_atoms, O_atoms, verbose=True,  nbins=nbins_rdf, range=(0.,rdf_range_max))
rdf_vasp_H_O.run()

rdf_vasp_O_O = rdf.InterRDF(O_atoms, O_atoms, exclusion_block=(1,1), verbose=True, nbins=nbins_rdf, range=(0.,rdf_range_max))
rdf_vasp_O_O.run()

rdf_vasp_H_H = rdf.InterRDF(H_atoms, H_atoms, exclusion_block=(1,1), verbose=True, nbins=nbins_rdf, range=(0.,rdf_range_max))
rdf_vasp_H_H.run()

# for ts in u.trajectory[3999:4005]:
#     rdf_vasp_H_O = rdf.InterRDF(H_atoms, O_atoms, verbose=True, nbins=nbins_rdf, range=(0.,rdf_range_max))
#     rdf_vasp_H_O.run()
    


file_2log = open("rdf.csv", 'w') 

csv_writer = csv.writer(file_2log, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

fields = ['rdf_bins','rdf_H_O','rdf_O_O', 'rdf_H_H']
data_output = np.transpose(np.vstack([rdf_vasp_H_O.results.bins, rdf_vasp_H_O.results.rdf, rdf_vasp_O_O.results.rdf, rdf_vasp_H_H.results.rdf]))

# for i1 in range(len(wc_field_density_points_vector)):
csv_writer.writerow(fields) 
csv_writer.writerows(data_output)

file_2log.close()       




# fig = plt.figure(figsize=(30, 15))
# plt.xlabel("r (Å)")
# plt.ylabel("RDF")
# plt.plot(rdf_vasp_H_O.results.bins, rdf_vasp_H_O.results.rdf, label='H-O')
# plt.plot(rdf_vasp_H_H.results.bins, rdf_vasp_H_H.results.rdf, label='H-H')
# plt.plot(rdf_vasp_O_O.results.bins, rdf_vasp_O_O.results.rdf, label='O-O')
# plt.plot(np.linspace(0,7),1*np.linspace(0,7)**0., 'k:')
# plt.legend(loc="best")
# plt.title(run_title_4_figures)
# plt.axis([0,a_box_dimn/2,0,8])
# # plt.axis([0,3.625,0,8])
# plt.grid(True, linestyle='--', alpha=0.7)
# fig.savefig("RDF_all.png", dpi=300, bbox_inches='tight')
# plt.show()

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))  # Adjust size as needed

# Plot the RDFs
ax.plot(rdf_vasp_H_O.results.bins, rdf_vasp_H_O.results.rdf, label='H-O', linewidth=2, linestyle='-', color='teal')
ax.plot(rdf_vasp_H_H.results.bins, rdf_vasp_H_H.results.rdf, label='H-H', linewidth=2, linestyle='--', color='orange')
ax.plot(rdf_vasp_O_O.results.bins, rdf_vasp_O_O.results.rdf, label='O-O', linewidth=2, linestyle='-.', color='purple')

# Add reference line at RDF = 1
ax.plot(np.linspace(0, 7), np.ones_like(np.linspace(0, 7)), 'k:', linewidth=1.5, label='Reference (RDF=1)')

# Customize axes labels and title
ax.set_xlabel(r"$r$ (Å)", fontsize=16, labelpad=10)
ax.set_ylabel("Radial Distribution Function (RDF)", fontsize=16, labelpad=10)
plt.title(run_title_4_figures+"; RDF of H-O, H-H, and O-O", fontsize=18, pad=15)
# plt.title(run_title_4_figures+"; Histogram - O, H atoms for time-steps=["+str(time_step)+':'+str(time_step+time_step_block)+']')

# Set axis limits
# ax.set_xlim([0, a_box_dimn / 2])
ax.set_xlim([0, 4.0])
# ax.set_ylim([0, 8])

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=14)

# Add grid with light styling
ax.grid(True, linestyle='--', alpha=0.5)

# Add legend with styling
ax.legend(loc='best', fontsize=14, frameon=True, shadow=True, fancybox=True)

# Save and show the figure
plt.tight_layout()  # Adjust layout to avoid clipping
fig.savefig("RDF_all.png", dpi=300, bbox_inches='tight')
# plt.show()






