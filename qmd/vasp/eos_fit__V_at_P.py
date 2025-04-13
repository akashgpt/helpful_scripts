# this script reads evo_total_pressure.dat, evo_cell_volume.dat and line 2 from peavg_summary.out. All in analysis directory in the home directory.
# then find pressures within epsilon_fP of the target_P and find the corresponding volumes.
# then use this to fit a Birch-Murnaghan equation of state to get the bulk modulus and its pressure derivative.
# the Birch-Murnaghan equation of state is given by:
# f = 0.5*( ((V0/V_data)**(2/3)) - 1);    P_est = ( 3*K0*f*((1+ 2*f)**2.5)*(1 + (1.5*(K0p - 4)*f)) ) where P_est has to be close to P_target.
# Then, estimate the volume at the target pressure using the Birch-Murnaghan equation of state.


#!/usr/bin/env python3

# print info on conda environment
import os, sys
print("Python version:", sys.version)
print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'Not in a conda environment')}")


import numpy as np
from scipy.optimize import curve_fit, fsolve
import argparse
import sys
from ase.io import read, write



# Read input optionally via parse_args
# Inputs:
parser = argparse.ArgumentParser(description="Estimate volume at target pressure using Birch-Murnaghan EOS.")
parser.add_argument("-p", "--target_P", type=float, required=True, help="Target pressure in GPa.")
parser.add_argument("-e", "--epsilon_fP", type=float, default=0.1, help="Fractional olerance for pressure matching (default: 0.10 GPa).")
parser.add_argument("-m", "--mode", type=int, default=0, help="Mode: 0 for trajectory based, 1 for relaxation calculation based.")
args = parser.parse_args()
target_P = args.target_P  # Target pressure in GPa
epsilon_fP = args.epsilon_fP  # Tolerance for pressure matching in GPa
mode = args.mode  # Mode: 0 for trajectory based, 1 for relaxation calculation based

print(f"Target pressure: {target_P} GPa")
print(f"Tolerance for pressure matching: {epsilon_fP*target_P} GPa")

# ---------------------------
# 1. Define file paths
# ---------------------------
current_dir = os.getcwd()
home_dir = current_dir
analysis_dir = os.path.join(home_dir, "analysis")

pressure_file = os.path.join(analysis_dir, "evo_total_pressure.dat")
volume_file   = os.path.join(analysis_dir, "evo_cell_volume.dat")
# summary_file  = os.path.join(analysis_dir, "peavg_summary.out")

if mode == 1:
    pressure_file = os.path.join(analysis_dir, "pressure.dat")
    volume_file   = os.path.join(analysis_dir, "volume.dat")

# ---------------------------
# 2. Read Data
# ---------------------------
# Assumes each file has one numerical value per line.
P_data = np.loadtxt(pressure_file)  # total pressure data array
V_data = np.loadtxt(volume_file)    # cell volume data array

# only consider the second half of the data
# P_data = P_data[int(P_data.size/2):]
# V_data = V_data[int(V_data.size/2):]

# Conver P_data from kBar to GPa
P_data = P_data / 10.0  # Convert from kBar to GPa

# ---------------------------
# 3. Filter Data
# ---------------------------
# Set epsilon_fP as a tolerance (here, 10% of target_P; adjust as needed)
mask = np.abs(P_data - target_P) < epsilon_fP*target_P
P_filtered = P_data[mask]
V_filtered = V_data[mask]
Cell_size_filtered = V_filtered ** (1/3)  # Convert volume to cell size

print(f"mask = {mask}")
# time_steps corresponding to mask
time_steps = np.arange(len(P_data))
time_steps_filtered = time_steps[mask]
# only keep time steps that are > 0.5 * len(P_data)
time_steps_filtered = time_steps_filtered[time_steps_filtered > 0.5 * len(P_data)]
# randomly chose 100 time steps
if time_steps_filtered.size > 100:
    np.random.seed(0)  # For reproducibility
    indices = np.random.choice(time_steps_filtered.size, 100, replace=False)
    time_steps_selected = time_steps_filtered[indices]
    print(f"Using 100 random time steps.")
else:
    np.random.seed(0)  # For reproducibility
    indices = np.random.choice(time_steps_filtered.size, 100, replace=True)
    time_steps_selected = time_steps_filtered[indices]
    print(f"Warning: Less than 100 time steps available. Using all available time steps.")
print(f"Selected time steps: {time_steps_selected}")
# print to file
filtered_file = os.path.join(analysis_dir, "selected_time_steps.txt")
with open(filtered_file, 'w') as f:
    for step in time_steps_selected:
        f.write(f"{step}\n")



# Print the range of filtered pressures
print("")
print(f"Filtered pressures: {P_filtered.min()} to {P_filtered.max()}")
print(f"Stddev of filtered pressures: {P_filtered.std()}")
print(f"Filtered volumes: {V_filtered.min()} to {V_filtered.max()}")
print(f"Stddev of filtered volumes: {V_filtered.std()}")
print(f"Corresponding cell sizes: {V_filtered.min()**(1/3)} to {V_filtered.max()**(1/3)}")
print(f"Mean cell size: {Cell_size_filtered.mean()}")
print(f"Stddev of filtered cell sizes: {Cell_size_filtered.std()}")
print("")

if P_filtered.size == 0:
    raise ValueError("No data points found within epsilon_fP of target pressure.")

print(f"Using {len(P_filtered)} data points for EOS fitting.")

# ---------------------------
# 4. Define the Birch–Murnaghan EOS function
# ---------------------------
def BM_pressure(V, V0, K0, K0p):
    """
    Third-order Birch–Murnaghan equation of state.
    f = 0.5*((V0/V)**(2/3) - 1)
    P_est = 3*K0*f*((1 + 2*f)**2.5) * (1 + 1.5*(K0p - 4)*f)
    All pressures are in the same units as P_data.
    """
    # print("V0/V = ", V0/V)
    f_val = 0.5 * ((V0/V)**(2/3) - 1.0)
    return 3.0 * K0 * f_val * ((1.0 + 2.0 * f_val)**2.5) * (1.0 + 1.5*(K0p - 4.0)*f_val)

# # ---------------------------
# # 5. Fit the EOS to the filtered data
# # ---------------------------
# # Initial guesses: V0 as median volume, K0 ~ 200, K0p ~ 4 (adjust units as needed)
# min_V = V_filtered.min()/10
# max_V = V_filtered.max()*10
# min_K0 = P_filtered.min()/100
# max_K0 = P_filtered.max()*100
# min_K0p = 3.
# max_K0p = 5.0
# lower_bounds = [min_V, min_K0, min_K0p]
# upper_bounds = [max_V, max_K0, max_K0p]
# p0 = [np.median(V_filtered), 200.0, 4.0]
# params, cov = curve_fit(BM_pressure, V_filtered, P_filtered, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)
# V0_fit, K0_fit, K0p_fit = params

# print("Fitted EOS parameters:")
# print("V0 =", V0_fit)
# print("K0 =", K0_fit)
# print("K0p =", K0p_fit)
# print("")
# # # ---------------------------
# # # 6. Estimate Volume at the Target Pressure
# # # ---------------------------
# def f_to_solve(V):
#     return BM_pressure(V, V0_fit, K0_fit, K0p_fit) - target_P

# # # Use fsolve starting from the fitted V0.
# V_est = fsolve(f_to_solve, V0_fit)[0]
# cell_size_est = V_est ** (1/3)  # Convert volume to cell size
# print("Estimated volume at target pressure:", V_est)
# print("Estimated cell size at target pressure:", cell_size_est)







from scipy.optimize import differential_evolution, fsolve

# Define bounds for parameters (adjust these as needed)
min_V   = V_filtered.min() / 1.0
max_V   = V_filtered.max() * 100.0
min_K0  = P_filtered.min() / 100.0
max_K0  = P_filtered.max() * 100.0
min_K0p = 4.0
max_K0p = 7.0

bounds = [(min_V, max_V), (min_K0, max_K0), (min_K0p, max_K0p)]

# Define an objective function: sum of squared residuals.
def objective(params):
    V0, K0, K0p = params
    residuals = BM_pressure(V_filtered, V0, K0, K0p) - P_filtered
    return np.sum(residuals**2)

# Use differential evolution to find the best-fit parameters.
result = differential_evolution(objective, bounds, maxiter=100000,tol=1e-3,polish=True)
V0_fit, K0_fit, K0p_fit = result.x

print("Fitted EOS parameters:")
print("V0 =", V0_fit)
print("K0 =", K0_fit)
print("K0p =", K0p_fit)
print("")

# # Define a function for fsolve to find the volume at which the predicted pressure equals target_P.
# def f_to_solve(V):
#     return BM_pressure(V, V0_fit, K0_fit, K0p_fit) - target_P

# Use fsolve, starting from the fitted V0
from scipy.optimize import least_squares

# Define f(V) as before
def residual(V):
    return BM_pressure(V, V0_fit, K0_fit, K0p_fit) - target_P
V0_guess = V_filtered.mean()  # Initial guess for V0
V_lower = V_filtered.min()
V_upper = V_filtered.max()
res = least_squares(residual, V0_guess, bounds=(V_lower, V_upper), xtol=1e-6)
V_est = res.x[0]
cell_size_est = V_est ** (1/3)  # Calculate cubic cell size

print("Estimated volume at target pressure:", V_est)
print("Estimated cell size at target pressure:", cell_size_est)

# V_est2 = 



# plot the fit with the data
import matplotlib.pyplot as plt
plt.figure()
cell_size = V_filtered ** (1/3)  # Convert volume to cell size
plt.scatter(cell_size, P_filtered, label='Filtered Data', color='blue')
plt.xlabel('Cell Size (A)')
# plt.scatter(V_filtered, P_filtered, label='Filtered Data', color='blue')
V_fit = np.linspace(V_filtered.min(), V_filtered.max(), 100)
P_fit = BM_pressure(V_fit, V0_fit, K0_fit, K0p_fit)
cell_size_fit = V_fit ** (1/3)  # Convert volume to cell size
plt.plot(cell_size_fit, P_fit, label='Fitted EOS', color='red')
# plt.xlabel('Volume (A^3)')
plt.xlabel('Cell Size (A)')
plt.ylabel('Pressure (GPa)')
plt.title('Birch-Murnaghan EOS Fit')
plt.axhline(y=target_P, color='green', linestyle='--', label=f'Target Pressure: {target_P} GPa')
plt.axvline(x=cell_size_est, color='orange', linestyle='--', label=f'Estimated Cell Size: {cell_size_est:.3f} A')
# plt.axvline(x=Cell_size_filtered.mean(), color='red', linestyle='--', label='Mean Cell Size from filtered data')
plt.ylim(0.99*P_filtered.min(), 1.01*P_filtered.max())
# plt.xlim(cell_size.min(), cell_size.max())
plt.legend()
plt.grid()
plt.savefig(os.path.join(analysis_dir, 'BM_EOS_fit.png'))


# create a log file log.eos_fit and save fitted parameters and estimated volume; write values and texts in different lines
log_file = os.path.join(analysis_dir, 'log.eos_fit')
with open(log_file, 'w') as f:
    f.write(f"Fitted EOS parameters:\n")
    f.write(f"V0 = {V0_fit}\n")
    f.write(f"K0 = {K0_fit}\n")
    f.write(f"K0p = {K0p_fit}\n")
    f.write(f"Target pressure: {target_P} GPa\n")
    f.write(f"Fractional tolerance for pressure matching: {epsilon_fP}\n")
    f.write(f"Number of data points used for fitting: {len(P_filtered)}\n")
    f.write(f"Estimated volume at target pressure: {V_est}\n")
    f.write(f"Estimated cell size at target pressure: {cell_size_est}\n")







######################################################################
######################################################################
######################################################################
# Specify the timesteps (images) you want to extract.
# Note: In ASE, the first image has index 0.
# timesteps = [0, 100, 200]  # Adjust as needed.

# Read the entire trajectory from XDATCAR
# The 'index=":"' syntax means "all frames."
# (Alternatively, you can do: read('XDATCAR@:', format='vasp'))
# images = read('XDATCAR', index=':')

# create folder in home_dir called hp_calculations
os.makedirs(os.path.join(home_dir, "hp_calculations"), exist_ok=True)
hp_dir = os.path.join(home_dir, "hp_calculations")

# Loop over the selected timesteps, create folders, and write POSCAR files.
counter=0
for step in time_steps_selected:
    counter += 1
    # print(f"Processing step {step}...")
    image = read('XDATCAR', index=step)  # Read the specific frame
    folder_name = f"{counter}"
    folder_name = os.path.join(hp_dir, folder_name)  # Create a folder for each step
    os.makedirs(folder_name, exist_ok=True)  # Create folder if it doesn't exist
    # image = images[step]                    # Grab the corresponding image
    image.wrap()  # Wrap atoms to unit cell
    write(os.path.join(folder_name, "POSCAR"), image, format='vasp', direct=True)

    # add POTCAR from home_dir in all these folders

print("Done! Created a folder and POSCAR for each requested timestep.")
######################################################################
######################################################################
######################################################################