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
# os.system(f'python {AG_GLOBAL}/misc_libraries/XDATCAR_toolkit/XDATCAR_toolkit.py -p -t {incar["POTIM"]}')
# u = mda.Universe("XDATCAR.pdb", dt=dt_sim)
os.system(f'python {AG_GLOBAL}/misc_libraries/XDATCAR_toolkit/XDATCAR_toolkit.py -p -t {incar["POTIM"]} --pbc')
u_unwrapped = mda.Universe("XDATCAR.pdb", dt=dt_sim)

################################################################
u_ref = u_unwrapped
################################################################

# Parse analysis data
runID_dir_analysis = os.path.join(os.getcwd(), "analysis/")
with open(runID_dir_analysis + 'peavg_numbers.out') as f1:
    peavg_numbers = np.array([float(line) if line else -np.inf for line in f1.read().splitlines()])
run_pressure, run_timesteps = peavg_numbers[23], peavg_numbers[31]




# Select all frames
t_select_bool = np.full(len(u_ref.trajectory), False)
t_beg, t_end = -1, len(u_ref.trajectory)
t_select_bool[t_beg:t_end] = True




# Extract unique atom types and format for MSD calculation
atom_types = np.unique(u_ref.atoms.types)
str_atom_types = ["type " + atype for atype in atom_types]
print(f"Atom types in the system: {str_atom_types}")

# Simulation parameters for plot titles
run_temperature = incar['TEBEG']
run_title_4_figures = f': t = {int(run_timesteps)}, T = {run_temperature}K, P = {str(run_pressure)[:5]}GPa, Atom types in the system: {str_atom_types}'


# Colors for different atom types in the MSD plot
colors = ['teal', 'darkorange', 'purple', 'blue', 'red', 'green', 'brown', 'pink', 'gray', 'cyan']

# Delete the existing diffusion coefficient file if it exists
os.system("rm -f diffusion_coefficient.txt")



# MSD Plot Setup
fig, ax1 = plt.subplots(figsize=(14, 10))

# Loop over each atom type and calculate MSD
for idx, (str_atom_type, atom_type) in enumerate(zip(str_atom_types, atom_types)):
    # Run MSD calculation
    MSD = msd.EinsteinMSD(u_unwrapped, select=str_atom_type, msd_type='xyz', fft=True)
    MSD.run()

    # Extract MSD timeseries and define lag times
    msd_data = MSD.results.timeseries
    nframes = MSD.n_frames
    timestep = u_ref.trajectory.dt  # Time step in fs
    lagtimes = np.arange(nframes) * timestep  # Lag time axis in fs

    # Plot MSD vs. time (linear scale)
    ax1.plot(lagtimes, msd_data, color=colors[idx % len(colors)], linestyle='--', linewidth=2, label=f"{atom_type}")

    # Calculate the diffusion coefficient
    start_index = int(nframes / 5)  # Start index for linear fit
    end_index = nframes  # End index for linear fit
    slope = linregress(lagtimes[start_index:end_index], msd_data[start_index:end_index]).slope

    # Calculate diffusion coefficient in Å²/fs and convert to cm²/s
    diffusion_coefficient = slope / (2 * MSD.dim_fac)  # Result in Å²/fs
    diffusion_coefficient_cm2_s = diffusion_coefficient * 1e-1  # Conversion to cm²/s

    # Save the diffusion coefficient to a file
    print(f"Self-diffusion coefficient for {atom_type}:  {diffusion_coefficient:.4e} Å²/fs, {diffusion_coefficient_cm2_s:.4e} cm²/s")
    
    diffusion_file_path = os.path.join(runID_dir_analysis, "diffusion_coefficient.txt")
    with open(diffusion_file_path, "a") as f:
        f.write(f"Self-diffusion coefficient for {atom_type}:\n")
        f.write(f"{diffusion_coefficient:.4e} Å²/fs\n")
        f.write(f"{diffusion_coefficient_cm2_s:.4e} cm²/s\n\n")

# Customize the MSD plot
ax1.set_xlabel("Time (fs)", fontsize=14)
ax1.set_ylabel("MSD (Å²)", fontsize=14)
ax1.set_title(f"Mean Square Displacement (MSD) Analysis {run_title_4_figures}", fontsize=16, pad=10)
ax1.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
ax1.legend(loc="best", fontsize=10)
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)

#tite for plot
# plt.suptitle(f"Mean Square Displacement (MSD) Analysis {run_title_4_figures}", fontsize=18, y=1.02)

# Save the plot
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("MSD_combined.jpg", dpi=300, bbox_inches='tight')
plt.show()
