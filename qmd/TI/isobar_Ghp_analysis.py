#!/usr/bin/env python3

# Usage:                       nohup python $HELP_SCRIPTS_TI/isobar_Ghp_analysis.py > log.isobar_Ghp_analysis 2>&1 &
# or (if no master_setup_TI)   python isobar_Ghp_analysis.py -b <base_dir> -p <pressure> -t <temperature>

# This script analyzes the Gibbs Helmholtz relation for a given isobaric simulation directory.
# It reads the enthalpy and GFE from the primary simulation directory, calculates the Gibbs Helmholtz integrand (H/T^2),
# and then processes secondary simulation directories to compute the Gibbs free energy (GFE) over temperature.
# It also fits a quadratic polynomial to the data and plots the results.



import numpy as np
import pandas as pd
import glob, re
import os
from mc_error import monte_carlo_error # located at $HELP_SCRIPTS/general/mc_error.py


kB = 1.380649e-23  # Boltzmann constant in J/K


# home directory for the analysis
ISOBAR_CALC_dir = os.getcwd()
CONFIG_dir      = os.path.dirname(ISOBAR_CALC_dir)
CONFIG_dirname  = os.path.basename(CONFIG_dir)
PT_dir          = os.path.dirname(CONFIG_dir)
PT_dirname      = os.path.basename(PT_dir)

print("" + "="*50)
print(f"ISOBAR_CALC_dir:    {ISOBAR_CALC_dir}")
print(f"CONFIG_dirname:     {CONFIG_dirname}")
print(f"PT_dirname:         {PT_dirname}")
print("" + "="*50)





# define pandas "CONFIG_data" with columns for temperature, pressure, and enthalpy
CONFIG_data = pd.DataFrame(columns=['tag','T_target', 'P_target', 'T_calc', 'T_calc_err', 'P_calc', 'P_calc_err', 'Enthalpy', 'Enthalpy_err', 'GFE', 'GFE_err',
                                    'H_over_T2', 'H_over_T2_err', 'GFE_over_T_calc', 'GFE_over_T_calc_err'])



# PART 1: primary simulation directory data
# Evaluate SCALEE_1 folder in CONFIG_dir
# strip PT_dir to get the temperature and pressure given the format: P{pressure}_T{temperature}
PT_info = re.search(r'P(\d+)_T(\d+)', CONFIG_dir)
if PT_info:
    pressure = float(PT_info.group(1))  # extract pressure
    temperature = float(PT_info.group(2))  # extract temperature
    # CONFIG_data['Pressure'] = pressure
    # CONFIG_data['Temperature'] = temperature

# for enthalpy, we will look for the peavg.out file in SCALEE_1/analysis
enthalpy_file = os.path.join(CONFIG_dir, 'SCALEE_1', 'analysis', 'peavg.out')
# print(f"Reading enthalpy from: {enthalpy_file}")
if os.path.exists(enthalpy_file):
    with open(enthalpy_file) as fh:
        counter_enthalpy = 0
        for line in fh:
            if 'Computed temperature' in line:
                # Extract the computed temperature and pressure values
                T_calc = float(line.split()[3])
                T_calc_err = float(line.split()[5])
            if 'Pressure' in line:
                P_calc = float(line.split()[2])
                P_calc_err = float(line.split()[4])
            if 'Enthalpy' in line and counter_enthalpy < 1:
                counter_enthalpy += 1
                print(f"Reading enthalpy from: {enthalpy_file}")
                print(f"Enthalpy line: {line.strip()}")
                # Extract the enthalpy value, as the fifth element in the line
                enthalpy_value = float(line.split()[4])
                enthalpy_err_value = float(line.split()[6])
else:
    print(f"Enthalpy file not found: {enthalpy_file}")
    # exit with error
    raise FileNotFoundError(f"Enthalpy file not found: {enthalpy_file}")

# if T_calc and P_calc are not set, use the target values
if 'T_calc' not in locals():
    T_calc = temperature
    T_calc_err = 0.0
if 'P_calc' not in locals():
    P_calc = pressure
    P_calc_err = 0.0

# GFE by reading log.Ghp_analysis file at CONFIG_dir
log_file = os.path.join(CONFIG_dir, 'log.Ghp_analysis')
counter_GFE = 0
if os.path.exists(log_file):
    with open(log_file) as fh:
        for line in fh:
            if 'G_hp' in line and counter_GFE < 1:
                counter_GFE += 1
                # Extract the GFE value, which is the last element in the line
                GFE_value = float(line.split()[2])
            if 'G_hp_err' in line:
                GFE_err_value = float(line.split()[2])
                break



# GFE_over_T_calc, GFE_over_T_calc_err
part_Gibbs_Helmholtz_Integral = lambda G, T: G / T
means = []
sigmas = []
# mean/sigma for GFE_over_T_calc
for G, T, G_err, T_err in zip([GFE_value], [T_calc], [GFE_err_value], [T_calc_err]):
    mean, sigma = monte_carlo_error(
        part_Gibbs_Helmholtz_Integral,
        [G, T],
        [G_err, T_err]
    )
    means.append(mean)
    sigmas.append(sigma)


# **append** as a new row**
next_idx = len(CONFIG_data)
CONFIG_data.loc[next_idx] = {
    'tag': "primary",
    'T_target': temperature,
    'T_calc': T_calc,
    'T_calc_err': T_calc_err,
    'P_target': pressure,
    'P_calc': P_calc,
    'P_calc_err': P_calc_err,
    'Enthalpy': enthalpy_value,
    'Enthalpy_err': enthalpy_err_value,
    'GFE': GFE_value,
    'GFE_err': GFE_err_value,
    'GFE_over_T_calc': means[0],
    'GFE_over_T_calc_err': sigmas[0]
}



# print("Primary simulation directory data:")
# print(f"P_target:       {pressure} GPa")
# print(f"T_target:    {temperature} K")
# print(f"Enthalpy:       {enthalpy_value} eV")
# print(f"Enthalpy_err:   {enthalpy_err_value} eV")
# print("" + "="*50)




# PART 2: Add secondary simulation directories data
# capture this as "CONFIG_enthalpy" array in this python script: grep -m1 Enthalpy  */SCALEE_0/analysis/peavg.out
# 1) Discover all peavg.out files and sort them
paths = sorted(glob.glob('*/SCALEE_0/analysis/peavg.out'))

# 2) Regex to capture the numeric value
pattern = re.compile(r'Enthalpy\s*[:=]\s*([-\d\.Ee+]+)')

# 3) Extract one enthalpy per file
Ts_calc = []
Ts_calc_err = []
Ps_calc = []
Ps_calc_err = []
enthalpies = []
enthalpies_errs = []
for p in paths:
    counter_enthalpy = 0
    with open(p) as fh:
        for line in fh:
            # Check for the computed temperature line
            if 'Computed temperature' in line:
                # Extract the computed temperature and pressure values
                T_calc = float(line.split()[3])
                T_calc_err = float(line.split()[5])
                Ts_calc.append(T_calc)
                Ts_calc_err.append(T_calc_err)
            # Check for the pressure line
            if 'Pressure' in line:
                P_calc = float(line.split()[2])
                P_calc_err = float(line.split()[4])
                Ps_calc.append(P_calc)
                Ps_calc_err.append(P_calc_err)
            # Check for the enthalpy line
            if 'Enthalpy' in line and counter_enthalpy < 1:
                counter_enthalpy += 1
                # Extract the enthalpy value, as the fifth element in the line
                enthalpy_value = float(line.split()[4])
                enthalpy_err_value = float(line.split()[6])
                enthalpies.append(enthalpy_value)
                enthalpies_errs.append(enthalpy_err_value)

# find all directories that start with "T" and strip the rest as the temperature
temp_dirs = sorted(glob.glob('T*'))
temperatures = []
for d in temp_dirs:
    match = re.search(r'T(\d+)', d)
    if match:
        temperatures.append(float(match.group(1)))

# print("Secondary simulation directories enthalpies:")
# for i, enthalpy in enumerate(enthalpies):
#     print(f"#{i+1} isobar sim:")
#     print(f"P_target:       {pressure} GPa")
#     print(f"T_target:    {temperatures[i]} K")
#     print(f"Enthalpy:       {enthalpy} eV")
# print("" + "="*50)

# add new entries for enthalpies, temperatures, and pressure (same in all instances as for primary) to CONFIG_data
for i in range(len(enthalpies)):
    next_idx = len(CONFIG_data)
    CONFIG_data.loc[next_idx] = {
        'tag': "secondary",
        'T_target': temperatures[i],
        'T_calc': Ts_calc[i],
        'T_calc_err': Ts_calc_err[i],
        'P_target': pressure,
        'P_calc': Ps_calc[i],
        'P_calc_err': Ps_calc_err[i],
        'Enthalpy': enthalpies[i],
        'Enthalpy_err': enthalpies_errs[i],
        # 'GFE': None, # empty for secondary simulations
        # 'GFE_err': None # empty for secondary simulations
    }
# sort CONFIG_data by Temperature
CONFIG_data.sort_values(by='T_target', inplace=True)







########################################################
########################################################
# PART 3: Calculate Enthalpy/T^2 with Monte Carlo error propagation
# Define the function for enthalpy over temperature squared
# H/T^2 = H / T^2 (Gibbs_Helmholtz_Integrand)
# where H is enthalpy and T is temperature
# We will use the monte_carlo_error function to propagate the uncertainty
Gibbs_Helmholtz_Integrand = lambda H, T: H / T**2
means   = []
sigmas  = []

for H, T, H_e, T_e in zip(
    CONFIG_data['Enthalpy'],
    CONFIG_data['T_calc'],
    CONFIG_data['Enthalpy_err'],
    CONFIG_data['T_calc_err'],
):
    # get 95% CI as well:
    mean, sigma = monte_carlo_error( 
        Gibbs_Helmholtz_Integrand, 
        [H, T],
        [H_e, T_e]
    )
    means.append(mean)
    sigmas.append(sigma)

# # now store back into your DataFrame
CONFIG_data['H_over_T2']    = means
CONFIG_data['H_over_T2_err']   = sigmas
########################################################
########################################################




## NOTE: remove all entries with T_target == 10400, 7200, 5200
CONFIG_data = CONFIG_data[~CONFIG_data['T_target'].isin([10400, 7200, 5200])].reset_index(drop=True)
print(f"\n\nNOTE: Removed entries with T_target == 10400, 7200, 5200. Likely crystalline Fe at these T/Ps.\n\n")




############################################################
# PART 4: fit a quadratic line to the data
coeffs__Gibbs_Helmholtz_Integrand = np.polyfit(CONFIG_data['T_calc'], CONFIG_data['H_over_T2'], 2)
T_calc_range = np.linspace(CONFIG_data['T_calc'].min(), CONFIG_data['T_calc'].max(), 100)
fit_line__Gibbs_Helmholtz_Integrand = np.polyval(coeffs__Gibbs_Helmholtz_Integrand, T_calc_range)
print(f"coeffs__Gibbs_Helmholtz_Integrand:      {coeffs__Gibbs_Helmholtz_Integrand}")
# goodness of fit
rmse = np.sqrt(np.mean((CONFIG_data['H_over_T2'] - np.polyval(coeffs__Gibbs_Helmholtz_Integrand, CONFIG_data['T_calc']))**2))
print(f"RMSE for the H/T^2 fit:                 {rmse}")
chi_squared = np.sum( ((CONFIG_data['H_over_T2'] - np.polyval(coeffs__Gibbs_Helmholtz_Integrand, CONFIG_data['T_calc'])) / CONFIG_data['H_over_T2_err'])**2 )
reduced_chi_squared = chi_squared / (len(CONFIG_data) - len(coeffs__Gibbs_Helmholtz_Integrand))
print(f"Chi-squared for the H/T^2 fit:          {chi_squared}")
print(f"Reduced chi-squared for the H/T^2 fit:  {reduced_chi_squared}")
print("" + "="*50)
##############################################################





















#####################################################################
# PART 5: Calculate GFE_over_T_calc for all entries
# a*T^2 + b*T + c
# integration > a*(T^3)/3 + b*(T^2)/2 + c*T + d
# (G1/T1) - (G0/T0) = (a*(T1^3)/3 + b*(T1^2)/2 + c*T1 + d) - (a*(T0^3)/3 + b*(T0^2)/2 + c*T0 + d)
# (G1/T1) - (G0/T0) = (a/3) * (T1^3 - T0^3) + (b/2) * (T1^2 - T0^2) + c * (T1 - T0)
# G1/T1 = (G0/T0) + (a/3) * (T1^3 - T0^3) + (b/2) * (T1^2 - T0^2) + c * (T1 - T0)


# given GFE_value, GFE_err_value at tag=primary, calculate the GFE/T_calc for all entries given coeffs__Gibbs_Helmholtz_Integrand and that integration of H/T^2 is GFE/T1 - GFE/T2
# for all secondary entries, calculate GFE_over_T_calc

# initialze with the size of the secondary entries
# GFE_over_T_calc_0=np.zeros(len(CONFIG_data[CONFIG_data['tag'] == 'secondary']))
# GFE_over_T_calc_err_0=np.zeros(len(CONFIG_data[CONFIG_data['tag'] == 'secondary']))
# T_0 = np.zeros(len(CONFIG_data[CONFIG_data['tag'] == 'secondary']))
# T_err_0 = np.zeros(len(CONFIG_data[CONFIG_data['tag'] == 'secondary']))

GFE_over_T_calc_0 = CONFIG_data.loc[CONFIG_data['tag'] == 'primary', 'GFE_over_T_calc'].values[0]
GFE_over_T_calc_err_0 = CONFIG_data.loc[CONFIG_data['tag'] == 'primary', 'GFE_over_T_calc_err'].values[0]
T_0 = CONFIG_data.loc[CONFIG_data['tag'] == 'primary', 'T_calc'].values[0]
T_err_0 = CONFIG_data.loc[CONFIG_data['tag'] == 'primary', 'T_calc_err'].values[0]


for i in range(len(CONFIG_data)):
    if CONFIG_data.loc[i, 'tag'] == 'secondary':
        T_1 = CONFIG_data.loc[i, 'T_calc']
        T_err_1 = CONFIG_data.loc[i, 'T_calc_err']

        # calculate GFE_over_T_calc for this entry using the coefficients from the primary entry
        GFE_over_T_calc_1 = lambda G_over_T_0, T_0, T_1: G_over_T_0 + (coeffs__Gibbs_Helmholtz_Integrand[0]/3) * (T_1**3 - T_0**3) + (coeffs__Gibbs_Helmholtz_Integrand[1]/2) * (T_1**2 - T_0**2) + coeffs__Gibbs_Helmholtz_Integrand[2] * (T_1 - T_0)
        means, sigmas = [], []
        # for G_over_T_0, T_0, T_1, G_over_T_err_0, T_err_0, T_err_1 in zip(GFE_over_T_calc_0, T_0, T_1, GFE_over_T_calc_err_0, T_err_0, T_err_1):
        mean, sigma = monte_carlo_error(
            GFE_over_T_calc_1,
            [GFE_over_T_calc_0, T_0, T_1],
            [GFE_over_T_calc_err_0, T_err_0, T_err_1]
        )
            # means.append(mean)
            # sigmas.append(sigma)
        CONFIG_data.loc[i, 'GFE_over_T_calc'] = mean
        CONFIG_data.loc[i, 'GFE_over_T_calc_err'] = sigma

        # calculate GFE
        GFE_calculation = lambda G_over_T, T: G_over_T * T
        GFE_mean, GFE_sigma = monte_carlo_error(
            GFE_calculation,
            [mean, T_1],
            [sigma, T_err_1]
        )
        CONFIG_data.loc[i, 'GFE'] = GFE_mean
        CONFIG_data.loc[i, 'GFE_err'] = GFE_sigma
####################################################################




############################################################
# PART 6: fit a quadratic line to the data
coeffs__GFE = np.polyfit(CONFIG_data['T_calc'], CONFIG_data['GFE'], 2)
T_calc_range = np.linspace(CONFIG_data['T_calc'].min(), CONFIG_data['T_calc'].max(), 100)
fit_line__GFE = np.polyval(coeffs__GFE, T_calc_range)
print(f"coeffs__GFE:                            {coeffs__GFE}")
# calculate RMSE and reduced chi-squared for the fit
rmse = np.sqrt(np.mean((CONFIG_data['GFE'] - np.polyval(coeffs__GFE, CONFIG_data['T_calc']))**2))
print(f"RMSE for the GFE fit:                   {rmse}")
chi_squared = np.sum( ((CONFIG_data['GFE'] - np.polyval(coeffs__GFE, CONFIG_data['T_calc'])) / CONFIG_data['GFE_err'])**2 )
reduced_chi_squared = chi_squared / (len(CONFIG_data) - len(coeffs__GFE))
print(f"Chi-squared for the GFE fit:            {chi_squared}")
print(f"Reduced chi-squared for the GFE fit:    {reduced_chi_squared}")
##############################################################





#######################################################################
# PART 7: Plot
# plot the enthalpy vs temperature
import matplotlib.pyplot as plt

# two panel plot
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.plot(CONFIG_data['T_calc'], CONFIG_data['Enthalpy'], marker='o', linestyle='--', color='b')
# plt.errorbar(CONFIG_data['T_target'], CONFIG_data['Enthalpy'], yerr=CONFIG_data['Enthalpy_err'], fmt='o', color='b', label='Enthalpy Error')

# another line with T_calc and T_calc_err
# plt.errorbar(CONFIG_data['T_calc'], CONFIG_data['Enthalpy'], xerr=CONFIG_data['T_calc_err'], yerr=CONFIG_data['Enthalpy_err'], fmt='o', color='b', label='Calculated Enthalpy Error')
plt.fill_between(CONFIG_data['T_calc'], 
                    CONFIG_data['Enthalpy'] - CONFIG_data['Enthalpy_err'], 
                    CONFIG_data['Enthalpy'] + CONFIG_data['Enthalpy_err'], 
                    color='blue', alpha=0.2, label='1σ CI')
plt.legend()
# plt.xlabel('T (K)')
plt.ylabel('H (eV)')
plt.grid()



# second subplot for Enthalpy/T_calc^2 with H_over_T2_err
plt.subplot(1, 3, 2)
plt.plot(CONFIG_data['T_calc'], CONFIG_data['H_over_T2'], marker='o', linestyle='', color='r')#, label='H/T^2')
plt.errorbar(CONFIG_data['T_calc'], CONFIG_data['H_over_T2'], yerr=CONFIG_data['H_over_T2_err'], fmt='o', color='r')#, label='H/T^2 Mean ± 1σ')
plt.fill_between(CONFIG_data['T_calc'], 
                    CONFIG_data['H_over_T2'] - CONFIG_data['H_over_T2_err'], 
                    CONFIG_data['H_over_T2'] + CONFIG_data['H_over_T2_err'],  
                    color='red', alpha=0.2, label='1σ CI')
# plt.fill_between(CONFIG_data['T_calc'], 
#                  CONFIG_data['H_over_T2_CI_low_2sigma'], 
#                  CONFIG_data['H_over_T2_CI_high_2sigma'], 
#                  color='red', alpha=0.2, label='95% CI')

# fit a quadratic polynomial to the H/T^2 data
plt.plot(T_calc_range, fit_line__Gibbs_Helmholtz_Integrand, color='red', label='Quadratic Fit', linestyle='--')
# fit_line__Gibbs_Helmholtz_Integrand = np.polyval(coeffs__Gibbs_Helmholtz_Integrand, CONFIG_data['T_calc'])
# plt.plot(CONFIG_data['T_calc'], fit_line__Gibbs_Helmholtz_Integrand, color='orange', label='Quadratic Fit')

plt.xlabel('Temperature (K)')
plt.ylabel('H/T$^2$ (eV/K$^2$)')
plt.grid()
plt.legend()





# plot GFE vs T
plt.subplot(1, 3, 3)
plt.plot(CONFIG_data['T_calc'], CONFIG_data['GFE'], marker='o', linestyle='', color='g')#, label='GFE')
plt.errorbar(CONFIG_data['T_calc'], CONFIG_data['GFE'], yerr=CONFIG_data['GFE_err'], fmt='o', color='g')
plt.fill_between(CONFIG_data['T_calc'], 
                    CONFIG_data['GFE'] - CONFIG_data['GFE_err'], 
                    CONFIG_data['GFE'] + CONFIG_data['GFE_err'], 
                    color='green', alpha=0.2, label='1σ CI')
# plot fit
plt.plot(T_calc_range, fit_line__GFE, color='green', label='Quadratic Fit', linestyle='--')
# plt.xlabel('T (K)')
plt.ylabel('G (eV)')
plt.grid()
plt.legend()



plt.suptitle(f'Gibbs Free Energy (G), Enthalpy (H), Gibbs-Helmholtz integrand (H/T$^2$) and Temperature (T) \n variation at pressure {pressure} GPa and for CONFIG {CONFIG_dirname}', fontsize=14)

# tight layout
plt.tight_layout()

plt.savefig(os.path.join(ISOBAR_CALC_dir, 'GH_analysis.png'), dpi=300)
# plt.show()











# summary
print("" + "="*50)
print("Summary of CONFIG_data:")
print(CONFIG_data)
print("" + "="*50)




# save the CONFIG_data DataFrame to a CSV file
output_csv = os.path.join(ISOBAR_CALC_dir, 'GH_analysis.csv') # Gibbs Helmholtz analysis
CONFIG_data.to_csv(output_csv, index=False)
print(f"CONFIG_data saved to {output_csv}")