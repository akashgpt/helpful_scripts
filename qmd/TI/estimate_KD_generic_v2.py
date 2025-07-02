#!/usr/bin/env python3
"""
Estimate chemical potentials and partition coefficients for {secondary_species} in Fe and MgSiO3 systems from TI data.
Walks through directory structure, parses log.Ghp_analysis files, assembles results into a DataFrame,
computes mixing fractions, fits linear excess chemical potentials, and adds entropy corrections.
Partially based on the discussion in the paper Li, et al. 2022 paper, "Primitive noble gases sampled from ocean island basalts cannot be from the Earth's core".

Usage: python $HELP_SCRIPTS_TI/estimate_KD.py

This script assumes the following directory structure:
Fe_{secondary_species}/
    P50_T3500/
        Config1/
            log.Ghp_analysis, TI_analysis.csv and isobar_calc/GH_analysis.csv
        Config2/
            log.Ghp_analysis, TI_analysis.csv and isobar_calc/GH_analysis.csv
    P100_T4000/
        Config1/
            log.Ghp_analysis, TI_analysis.csv and isobar_calc/GH_analysis.csv
    ...
MgSiO3_{secondary_species}/
    P50_T3500/
        Config1/
            log.Ghp_analysis, TI_analysis.csv and isobar_calc/GH_analysis.csv
        Config2/
            log.Ghp_analysis, TI_analysis.csv and isobar_calc/GH_analysis.csv
    ...
This script will output a CSV file `all_TI_results.csv` with the following columns:
Phase, P_T_folder, Config_folder, P_target, T_target, Atom counts, Atomic masses,
Unique species, Total # of atoms, G_hp, G_hp_error, G_hp_per_atom, G_hp_per_atom_error,
HF_ig, TS, Volume (Å³), Volume per atom (Å³), X_{secondary_species}, mu_excess_{secondary_species}, mu_{secondary_species}_TS_term, mu_{secondary_species},
mu_excess_{secondary_species}, mu_{secondary_species}, KD_sil_to_metal, D_wt, a, b, G_hp_per_atom_w_TS, TS_per_atom

Author: Akash Gupta
"""

import re
import ast
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import os
from mc_error import monte_carlo_error # located at $HELP_SCRIPTS/general/mc_error.py
from scipy.optimize import curve_fit

# Boltzmann constant in eV/K for entropy term
kB = 8.617333262145e-5


# read secondary species from terminal, e.g., He, H, C, etc.
parser = argparse.ArgumentParser(description="Estimate KD for a species in Fe and MgSiO3 systems from TI data.")
parser.add_argument(
    "-s", "--secondary_species",
    type=str,
    default="He",
    help="Secondary species to estimate KD for (default: He)."
)
args = parser.parse_args()
# Use the parsed secondary species
secondary_species = args.secondary_species




current_dir = os.getcwd()
home_dir = current_dir
home_dirname = home_dir.split(os.sep)[-1]
print(f"\nHome directory: {home_dir}\n")
print(f"Home directory name: {home_dirname}\n")



# 1) Define the root directories containing your TI data
ROOT_DIRS = [
    f"Fe_{secondary_species}",
    f"MgSiO3_{secondary_species}"
]

print(f"Using secondary species: {secondary_species}")
# Ensure the root directories exist
for root in ROOT_DIRS:
    if not Path(root).is_dir():
        print(f"Error: Root directory {root} does not exist.")
        sys.exit(1)


# 2) Regex to capture lines of the form "Key : Value"
KV_RE = re.compile(r'^\s*([^:]+?)\s*:\s*(.+)$')

# 3) Parsers to clean raw string values into Python types
int_re   = re.compile(r'-?\d+')
float_re = re.compile(r'[-+]?\d*\.?\d+|\d+')
str_re   = re.compile(r"'([^']*)'|\"([^\"]*)\"")  # capture quoted strings

def parse_log(path):
    """
    Read a log.Ghp_analysis file and extract all key: raw_value pairs.
    Returns a dict of {key: raw_string} for further parsing.
    """
    out = {}
    text = path.read_text().splitlines()
    for line in text:
        m = KV_RE.match(line)
        if not m:
            continue
        key, raw = m.group(1).strip(), m.group(2).strip()
        out[key] = raw
    return out

# Helper functions to convert raw strings to lists or scalars

def parse_int_list(raw):
    """
    Convert a raw string like "[ 0 64]" or "0,64" into a list of ints.
    If already a list/tuple, cast to list.
    """
    if isinstance(raw, str):
        return [int(x) for x in int_re.findall(raw)]
    return list(raw)


def parse_float_list(raw):
    """
    Convert a raw string like "[4.002602 55.845]" into a list of floats.
    If already a list/tuple, cast to list.
    """
    if isinstance(raw, str):
        return [float(x) for x in float_re.findall(raw)]
    return list(raw)


def parse_species_list(raw):
    """
    Convert a raw string representing species list, possibly missing a trailing bracket,
    into a list of strings, e.g. "['He', 'Fe'" -> ['He','Fe'], or "['H', 'Fe']" -> ['H','Fe'].
    """
    if not isinstance(raw, str):
        return list(raw)
    s = raw.strip()
    # add missing closing bracket if needed
    if s.startswith('[') and not s.endswith(']'):
        s += ']'
    try:
        # safe literal_eval to parse Python list syntax
        lst = ast.literal_eval(s)
        return [str(x) for x in lst]
    except Exception:
        # fallback: extract quoted tokens
        return [a or b for a, b in str_re.findall(s)]


def parse_scalar(raw):
    """
    Try to convert raw string to float, otherwise leave as-is.
    """
    try:
        return float(raw)
    except Exception:
        return raw

# 4) Traverse directories and collect parsed log data
records = []
for phase in ROOT_DIRS:
    # iterate pressure_temperature folders (e.g. P50_T3500)
    for pt_dir in Path(phase).iterdir():
        if not pt_dir.is_dir():
            continue
        # iterate configuration subfolders containing log.Ghp_analysis
        for cfg in pt_dir.iterdir():
            logf = cfg / "log.Ghp_analysis"
            if not logf.is_file():
            # if not (cfg / "SCALEE_1").is_dir():
                continue
            # parse the file into a raw dict
            rec = parse_log(logf)
            # annotate provenance columns
            rec["Phase"] = phase
            rec["P_T_folder"] = pt_dir.name
            rec["Config_folder"] = cfg.name
            records.append(rec)

# create pandas DataFrame from list of dicts
df = pd.DataFrame(records)

# 5) Clean and convert columns to appropriate Python types
# list-type columns
if "Atom counts" in df:
    df["Atom counts"] = df["Atom counts"].apply(parse_int_list)
if "Atomic masses" in df:
    df["Atomic masses"] = df["Atomic masses"].apply(parse_float_list)
if "Unique species" in df:
    df["Unique species"] = df["Unique species"].apply(parse_species_list)

# scalar columns to convert
scalar_cols = [
    "Target pressure", "Target temperature",
    "Total # of atoms", "G_hp", "G_hp_error",
    "G_hp_per_atom", "G_hp_per_atom_error",
    "HF_ig", "TS", "Volume (Å³)", "Volume per atom (Å³)"
]
for col in scalar_cols:
    if col in df.columns:
        df[col] = df[col].apply(parse_scalar)


# 6) Extract numeric pressure & temperature values and rename with units
if "Target pressure" in df.columns:
    df["Target pressure (GPa)"] = (
        df["Target pressure"]
            .astype(str)
            .str.extract(r'([-+]?\d*\.?\d+)', expand=False)
            .astype(float)
    )
if "Target temperature" in df.columns:
    df["Target temperature (K)"] = (
        df["Target temperature"]
            .astype(str)
            .str.extract(r'([-+]?\d*\.?\d+)', expand=False)
            .astype(float)
    )



# sort all columns wrt "Phase" and "Target pressure (GPa)"
df.sort_values(by=["Phase", "Target pressure (GPa)"], inplace=True)
# df.sort_values(by=["Target pressure (GPa)"], inplace=True)



# 7) Compute {secondary_species} mole fraction X_{secondary_species} = n_{secondary_species} / total_atoms for each row

def frac_secondary_species(row):
    counts = row["Atom counts"]
    species = row["Unique species"]
    mapping = dict(zip(species, counts))
    total = sum(counts)
    return mapping.get(secondary_species, 0) / total if total else 0.0

df[f"X_{secondary_species}"] = df.apply(frac_secondary_species, axis=1)

# 8) Drop any columns starting with WARNING
warn_cols = [c for c in df.columns if c.startswith("WARNING")]
if warn_cols:
    df.drop(columns=warn_cols, inplace=True)





# TS_per_atom = df["TS"] / df["Total # of atoms"]
df["TS_per_atom"] = df["TS"] / df["Total # of atoms"]


print(f"\nWARNING: Check the equations below with Haiyang!!!\n")
# G_hp_per_atom_w_TS = df["G_hp_per_atom"] + df["TS_per_atom"]
# for all cases except those with n_{secondary_species} = 0
for i, row in df.iterrows():
    if row[f"X_{secondary_species}"] > 0:
        df.at[i, "G_hp_per_atom_w_TS"] = row["G_hp_per_atom"] + row["TS_per_atom"]
    else:
        df.at[i, "G_hp_per_atom_w_TS"] = row["G_hp_per_atom"]
        # df.at[i, "G_hp_per_atom"] = row["G_hp_per_atom"] - row["TS_per_atom"]
        
# df["G_hp_per_atom_w_TS"] = df["G_hp_per_atom"] + df["TS_per_atom"]

print(f"WARNING: the line above was not commented earlier but I just did it now as I can't figure out why it was added in the first place. It seems to be a mistake though doesn't affect results previous or current in any way.\n\n")


df["G_hp_per_atom_w_TS_error"] = df["G_hp_per_atom_error"]  # assuming no error in TS term


# 9) Fit linear excess chemical potential mu_excess = a + b
# Initialize columns
df["intercept"] = np.nan
df["slope"] = np.nan
df["intercept_error"] = np.nan
df["slope_error"] = np.nan

# Group by Phase and P_T_folder to fit separate lines
for (phase, pt), sub in df.groupby(["Phase", "P_T_folder"]):
    x = sub[f"X_{secondary_species}"].values
    y = sub["G_hp_per_atom_w_TS"].values
    y_error = sub["G_hp_per_atom_w_TS_error"].values # 1 sigma error in G_hp_per_atom_w_TS
    # weights = 1/σ_y
    w = 1.0 / y_error
    if len(x) > 1:
        # coeffs, cov = np.polyfit(x, y, 1, cov=True)#,w=w)  # fit y = a + b*x or y = intercept + slope*x
        # slope, intercept = coeffs
        # # Calculate the standard error of the slope and intercept
        # slope_error = np.sqrt(cov[0, 0])
        # intercept_error = np.sqrt(cov[1, 1])
        # # error in slope and intercept

        def fn_linear(x, m, b):
            return m*x + b

        popt, pcov = curve_fit(
            fn_linear, x, y,
            sigma=y_error,          # y‐errors
            absolute_sigma=True   # use true σ to scale covariance
        )

        slope, intercept = popt
        slope_error, intercept_error = np.sqrt(np.diag(pcov))
    else:
        intercept, slope = np.nan, np.nan
        slope_error, intercept_error = np.nan, np.nan
    mask = (df["Phase"] == phase) & (df["P_T_folder"] == pt)
    df.loc[mask, "intercept"] = intercept
    df.loc[mask, "slope"] = slope
    df.loc[mask, "intercept_error"] = intercept_error
    df.loc[mask, "slope_error"] = slope_error
    # print(f"Fitted line for {phase} at {pt}: intercept={intercept:.3f}, slope={slope:.3f}, intercept_error={intercept_error:.3f}, slope_error={slope_error:.3f}")

    # Compute mu_excess = a + b (no X_{secondary_species} factor)
    fn_mu_excess = lambda slope, intercept: slope + intercept
    mu_excess, mu_excess_error = monte_carlo_error(fn_mu_excess, [slope, intercept], [slope_error, intercept_error])
    # mu_excess = fn_mu_excess(slope, intercept)  # compute mu_excess
    df.loc[mask, f"mu_excess_{secondary_species}"] = mu_excess
    df.loc[mask, f"mu_excess_{secondary_species}_error"] = mu_excess_error
    # print(f"Computed mu_excess for {phase} at {pt}: {mu_excess:.3f} ± {mu_excess_error:.3f}")
# # Compute mu_excess = a + b (no X_{secondary_species} factor)
# fn_mu_excess = lambda slope, intercept: slope + intercept
# mu_excess, mu_excess_error = monte_carlo_error(fn_mu_excess, [df["slope"], df["intercept"]], [df["slope_error"], df["intercept_error"]])
# df[f"mu_excess_{secondary_species}"] = mu_excess
# df[f"mu_excess_{secondary_species}_error"] = mu_excess_error

# 10) Compute mixing entropy term mu_TS and total mu_{secondary_species}
# def compute_mu_TS(row):
def compute_mu_TS(T,X):
    """
    Compute the TS term mixing entropy term for {secondary_species}, using different formulas per phase.
    Fe_{secondary_species}: TS = kB * T * ln(X)
    MgSiO3_{secondary_species}: TS = kB * T * ln( X / (5 - 4X) )
    """
    # T = row.get("Target temperature (K)")
    # X = row.get(f"X_{secondary_species}", 0)

    # add floor to X to avoid log(0)
    X = max(X, 1e-10)  # avoid log(0) or negative values

    if T is None or X < 0:
        return np.nan
    if X == 0:
        return 0.0 # assuming no contribution to TS if X is 0
    if row["Phase"] == f"Fe_{secondary_species}":
        # print(f"kB * T * np.log(X) for Fe_{secondary_species} = {kB * T * np.log(X)}")
        return kB * T * np.log(X)
    elif row["Phase"] == f"MgSiO3_{secondary_species}":
        # print(f"kB * T * np.log(X / (5 - 4 * X)) for MgSiO3_{secondary_species} = {kB * T * np.log(X / (5 - 4 * X))}")
        denom = 5 - 4 * X
        return kB * T * np.log(X / denom)# if denom > 0 else np.nan
    else:
        return np.nan

print(f"WARNING: Assuming mu_TS is 0 if X = 0. Okay??\n")
print(f" ############### NOTE: We NEVER USE mu_TS_term or mu for anything -- only mu_excess which comes from the variation in GFE with X ###############\n\n")

# apply TS term and total mu_{secondary_species}
# df[f"mu_{secondary_species}_TS_term"] = df.apply(compute_mu_TS, axis=1)
for i, row in df.iterrows():
    T = row.get("Target temperature (K)")
    X = row.get(f"X_{secondary_species}", 0)
    # print(f"Computing mu_TS_term for row {i}: T={T}, X={X} for {secondary_species}, Phase: {row['Phase']}, Config_folder: {row['Config_folder']}")
    # df.at[i, f"mu_{secondary_species}_TS_term"] = compute_mu_TS(T, X)
    mu_TS_term, mu_TS_term_error = monte_carlo_error(compute_mu_TS, [T, X], [0.0, 0.0])  # assuming no error in T and X
    df.at[i, f"mu_{secondary_species}_TS_term"] = mu_TS_term
    df.at[i, f"mu_{secondary_species}_TS_term_error"] = mu_TS_term_error
    # print(f"Computed mu_TS_term for row {i}: {mu_TS_term:.3f} ± {mu_TS_term_error:.3f}\n")

    fn_mu = lambda mu_excess, mu_TS_term: mu_excess + mu_TS_term
    mu, mu_error = monte_carlo_error(fn_mu, [row[f"mu_excess_{secondary_species}"], mu_TS_term], [row[f"mu_excess_{secondary_species}_error"], mu_TS_term_error])
    df.at[i, f"mu_{secondary_species}"] = mu
    df.at[i, f"mu_{secondary_species}_error"] = mu_error

# df[f"mu_{secondary_species}"] = df[f"mu_excess_{secondary_species}"] + df[f"mu_{secondary_species}_TS_term"]


# print(f"df summary:\
# {df.describe()}\n")



# partiction coefficient: (1.78/5) * np.exp(-(mu_excess_{secondary_species}_for_Fe - mu_excess_{secondary_species}_for_MgSiO3) / (kB * T)) for the same P_T_folder
def compute_KD(row):
    # 1) Identify this row’s phase & P_T group
    phase = row["Phase"]
    pt    = row["P_T_folder"]
    # 2) Determine the other phase
    other_phase = f"MgSiO3_{secondary_species}" if phase == f"Fe_{secondary_species}" else f"Fe_{secondary_species}"
    # 3) Grab that phase’s mu_excess_{secondary_species} for the same P_T_folder and same temperature
    mask = (
            (df["Phase"] == other_phase) &
            (df["P_T_folder"] == pt) &
            (df["Target temperature (K)"] == row["Target temperature (K)"])
            )
    partner__mu_excess = df.loc[mask, f"mu_excess_{secondary_species}"]
    partner__mu_excess_error = df.loc[mask, f"mu_excess_{secondary_species}_error"]

    # Determine the multiplier for the exponent based on phase
    if phase == f"Fe_{secondary_species}":
        mult_factor = 1 # to ensure that KD is always for {secondary_species}_{silicate} -> {secondary_species}_{metal}
    else:
        mult_factor = -1

    if partner__mu_excess.empty or partner__mu_excess_error.empty:
        return np.nan, np.nan
    other__mu_excess = partner__mu_excess.iloc[0]
    other__mu_excess_error = partner__mu_excess_error.iloc[0]
    mu_excess = row[f"mu_excess_{secondary_species}"]
    mu_excess_error = row[f"mu_excess_{secondary_species}_error"]
    if mu_excess is None or np.isnan(mu_excess) or other__mu_excess is None or np.isnan(other__mu_excess):
        return np.nan, np.nan
    # 4) Get the temperature (in K)
    T = row["Target temperature (K)"]
    if np.isnan(T) or T <= 0:
        return np.nan
    # 5) Compute KD
    # solve for KD = (x/y) such that (x/y) = (1/(5-4*y)) * np.exp(-mult_factor*(row[f"mu_excess_{secondary_species}"] - other__mu_excess) / (kB * T))
    # return (1/5.0) * np.exp(-mult_factor*(row[f"mu_excess_{secondary_species}"] - other__mu_excess) / (kB * T)) # assuming y is ~ 0
    fn_KD = lambda mu_excess, other__mu_excess, T, mult_factor: (1/5.0) * np.exp(-mult_factor * (mu_excess - other__mu_excess) / (kB * T))
    KD, KD_error = monte_carlo_error(fn_KD, [mu_excess, other__mu_excess, T, mult_factor], [mu_excess_error, other__mu_excess_error, 0.0, 0.0])

    return KD, KD_error

# Apply it
# df["KD_sil_to_metal"] = df.apply(compute_KD, axis=1)
df["KD_sil_to_metal"] = np.nan
df["KD_sil_to_metal_error"] = np.nan
for i, row in df.iterrows():
    KD, KD_error = compute_KD(row)
    df.at[i, "KD_sil_to_metal"] = KD
    df.at[i, "KD_sil_to_metal_error"] = KD_error
    # print(f"Computed KD_sil_to_metal for row {i}: {KD:.3f} ± {KD_error:.3f} for {row['Phase']} at {row['P_T_folder']}\n")
df["D_wt"] = df["KD_sil_to_metal"] * (100/56)
df["D_wt_error"] = df["KD_sil_to_metal_error"] * (100/56)



# 11) Print summary of results
print(f"\nKD_sil_to_metal for {secondary_species} in Fe and MgSiO3 systems:")
# Loop over each unique (P_T_folder, Phase) combination
for (folder, phase), sub_df in df.groupby(["P_T_folder", "Phase"]):
    # grab all non-NaN KD values
    kd_vals = sub_df["KD_sil_to_metal"].dropna()
    if not kd_vals.empty:
        kd0 = kd_vals.iloc[0]
        mu0 = sub_df[f"mu_excess_{secondary_species}"].iloc[0]
        print(f"{folder}, Phase={phase}:")
        print(f"                                KD_sil_to_metal = {kd0:.4f} ± {sub_df['KD_sil_to_metal_error'].iloc[0]:.4f} (mu_excess = {mu0:.4f} ± {sub_df[f'mu_excess_{secondary_species}_error'].iloc[0]:.4f})")
        print(f"                                slope = {sub_df['slope'].iloc[0]:.4f} ± {sub_df['slope_error'].iloc[0]:.4f}, intercept = {sub_df['intercept'].iloc[0]:.4f} ± {sub_df['intercept_error'].iloc[0]:.4f}")
    else:
        print(f"{folder}, Phase={phase}: "
                "No valid KD_sil_to_metal values found.")


# exit(0)

print(f"="*50)
print(f"Note: It is assumed here that y or X_{secondary_species} in silicates is << 1. If not, multiply K_D and D_wt by (5 / (5-4y)) !")
print(f"="*50)
print("")


# add a new column to df called "tag" = primary -- for all
df["calc_type"] = "TI"










# # increase the display width to 200 characters
pd.set_option('display.max_colwidth', 200)
# print(f"df['Base directory']:\n{df['Base directory'].head()}\n")


# Update Base directory column: find home_dirname in df["Base directory"] and replace that and everything before it with home_dir
df["Base directory"] = df["Base directory"].str.replace(f".*{home_dirname}", home_dir, regex=True)

# # print Base directory and all columns
# # print(f"Base directory: {ROOT_DIRS}")
print(f"All columns: {df.columns.tolist()}\n")


# print(f"df['Base directory']:\n{df['Base directory'].head()}\n")

# exit(0)




# # print unique X_{secondary_species} for each phase
# unique_X = df.groupby("Phase")[f"X_{secondary_species}"].unique()
# print(f"Unique X_{secondary_species} values for each phase:")
# for phase, values in unique_X.items():
#     print(f"{phase}: {values}")


# sort df -- reverserd order of columns
# df.sort_values(by=["Phase", "Target pressure (GPa)"], ascending=[True, False], inplace=True)


counter_GH_analysis = 0
df_superset= pd.DataFrame()




# Group by Phase and P_T_folder to fit separate lines
for (phase, pt), df_subset in df.groupby(["Phase", "P_T_folder"]):

    # create an empty df_isobar_superset
    df_isobar_superset = pd.DataFrame()

    # PT_dir = parent dir of df_subset["Base directory"][0]
    PT_dir = Path(df_subset["Base directory"].iloc[0]).parent
    print(f"Processing Phase: {phase}, P_T_folder: {pt} in directory: {PT_dir}")


    ## if PT_dirname P250_T6500, skip
    PT_dirname = PT_dir.name
    # if PT_dirname.startswith("P250_T6500"):
    #     print(f"#"*50)
    #     print("WARNING: TEMPORARY SKIP")
    #     print(f"*** Skipping {phase} at {pt} as it is in {PT_dirname}. ***")
    #     print(f"#"*50)
    #     continue

    # # print df_subset
    # print(f"Processing {phase} at {pt} with {len(df_subset)} entries.")
    # print(f"df_subset:\n{df_subset}\n")

    # check if there are two GH_analysis.csv files in PT_dir: find . -type f -name "GH_analysis.csv" | wc -l
    num_GH_analysis_files = len(list(PT_dir.glob("**/GH_analysis.csv")))
    
    if num_GH_analysis_files < 2:
        print(f"*** Skipping {phase} at {pt} as there are not enough GH_analysis.csv files (found {num_GH_analysis_files}). ***\n\n")
        continue
    else:
        print(f"Found {num_GH_analysis_files} GH_analysis.csv files.\n\n")

    # print unique df_subset["Base directory"].values
    # print(f"Unique Base directories:\n{df_subset['Base directory'].unique()}\n")

    # for each Base directory, go inside isobar_calc folder if it exists (i.e., <Base_dir>/isobar_calc).
    for CONFIG_base_dir in df_subset["Base directory"]:

        df_base= df_subset[df_subset["Base directory"] == CONFIG_base_dir]
        CONFIG_dirname = Path(CONFIG_base_dir).name



        #isobar_dir = CONFIG_base_dir / "isobar_calc"
        # isobar_dir = os.path.join(CONFIG_base_dir, "isobar_calc") # Use os.path.join for compatibility
        config_base = Path(CONFIG_base_dir)
        isobar_dir  = config_base / "isobar_calc"

        # if the directory name contains *_1H*, *_2H*, *_4H* or *_8H*, skip
        if any(x in CONFIG_dirname for x in ["_1H", "_2H", "_4H"]):
            # print(f"Skipping {CONFIG_dirname}.")
            continue

        if not isobar_dir.is_dir():
            print(f"Skipping {CONFIG_dirname} as it does not contain an isobar_calc directory.\n")
            continue

        print(f"Processing isobar_calc for {CONFIG_dirname} ...")
        print("="*4)
        # print mu_{secondary_species}_TS_term for each row in df_base
        print(f"mu_{secondary_species}_TS_term for {CONFIG_dirname}: {df_base[f'mu_{secondary_species}_TS_term'].values[0]}")


        # Make 5 duplicate rows based on all df_base entries, and save this in df_isobar.
        df_isobar = df_base.copy()
        df_isobar = pd.concat([df_isobar] * 4, ignore_index=True) # 4 new ones/isobar + the original
        # print(f"Created {len(df_isobar)} duplicate rows for isobar_calc processing.")

        # make all values nan
        # df_isobar.loc[:, df_isobar.columns != "Config_folder"] = np.nan

        # 1) grab only “T…” dirs
        temp_dirs = [
            d for d in isobar_dir.iterdir()
            if d.is_dir() and d.name.startswith("T")
        ]

        # 2) sort by the numeric part after “T”
        def _parse_temp(d: Path) -> float:
            try:
                return float(d.name[1:])
            except ValueError:
                return float('inf')  # push bad names to the end

        temp_dirs.sort(key=_parse_temp)
        # print(f"Found {len(temp_dirs)} temperature directories in {isobar_dir}.")

        GH_analysis_file = isobar_dir / "GH_analysis.csv"
        print(f"GH_analysis_file: {GH_analysis_file}")
        if not GH_analysis_file.is_file():
            print(f"Skipping {isobar_dir} as {GH_analysis_file} does not exist.\n\n")
            # make first four rows of df_isobar["mu_excess_{secondary_species}"], mu_excess_{secondary_species}_error, G_hpm G_hp_error, Target temperature (K) = np.nan
            df_isobar.loc[:3, f"mu_excess_{secondary_species}"] = np.nan
            df_isobar.loc[:3, f"mu_excess_{secondary_species}_error"] = np.nan
            df_isobar.loc[:3, "G_hp"] = np.nan
            df_isobar.loc[:3, "G_hp_error"] = np.nan
            df_isobar.loc[:3, "Target temperature (K)"] = np.nan
            continue

        counter_GH_analysis += 1
        GH_df = pd.read_csv(GH_analysis_file)
        # print(f"Heading of GH_analysis.csv:\n{GH_df.head()}\n")
        # columns
        # print(f"GH_analysis.csv columns: {GH_df.columns.tolist()}\n")
        
        # There, go into each directory beginning with "T". Whatever comes after "T" is the temperature in K.
        # 3) enumerate and fill your DataFrame
        for idx, temp_dir in enumerate(temp_dirs):
            temp_val = _parse_temp(temp_dir)
            if temp_val == float('inf'):
                print(f"Skipping {temp_dir!r}: invalid temperature")
                continue

            # print(f"Row {idx}: found {temp_dir.name!r} → {temp_val} K")
            # update temperature in df_isobar
            df_isobar.loc[idx, "Target temperature (K)"] = temp_val

            # calc_type = secondary
            df_isobar.loc[idx, "calc_type"] = "GH"


            # then
            # TS_per_atom = (df_base["TS_per_atom"] / df_base["Target temperature (K)"]) * df_isobar["Target temperature (K)"]
            df_isobar.loc[idx, "TS_per_atom"] = (df_base["TS_per_atom"].iloc[0] / df_base["Target temperature (K)"].iloc[0]) * temp_val
            # read GH_analysis.csv as GH_df and grab the GFE and GFE_error based on corresponding GH_df["T_target"]==df_isobar["Target temperature (K)"] and replace "G_hp", "G_hp_error" in df_isobar
            df_isobar.loc[idx, "G_hp"] = GH_df.loc[GH_df["T_target"] == temp_val, "GFE"].values[0]
            df_isobar.loc[idx, "G_hp_error"] = GH_df.loc[GH_df["T_target"] == temp_val, "GFE_err"].values[0]

            
            # Calculate G_hp_per_atom = G_hp / Total # of atoms and G_hp_per_atom_error = G_hp_per_atom / Total # of atoms, and update df_isobar with these values.
            total_atoms = df_base["Total # of atoms"].iloc[0]
            df_isobar.loc[idx, "G_hp_per_atom"] = df_isobar.loc[idx, "G_hp"] / total_atoms
            df_isobar.loc[idx, "G_hp_per_atom_error"] = df_isobar.loc[idx, "G_hp_error"] / total_atoms

            # Then update "G_hp_per_atom_w_TS" in df_isobar as well, using the same formula as above in this script, i.e., G_hp_per_atom_w_TS = G_hp_per_atom + TS_per_atom if X_{secondary_species} > 0, else just G_hp_per_atom.
            if df_isobar.loc[idx, f"X_{secondary_species}"] > 0:
                df_isobar.loc[idx, "G_hp_per_atom_w_TS"] = df_isobar.loc[idx, "G_hp_per_atom"] + df_isobar.loc[idx, "TS_per_atom"]
            else:
                df_isobar.loc[idx, "G_hp_per_atom_w_TS"] = df_isobar.loc[idx, "G_hp_per_atom"]
            df_isobar.loc[idx, "G_hp_per_atom_w_TS_error"] = df_isobar.loc[idx, "G_hp_per_atom_error"]  # assuming no error in TS term

        # print("\n"*2)

        # sort wrt temperature
        df_isobar.sort_values("Target temperature (K)", inplace=True)
        GH_df.sort_values("T_target", inplace=True)


        print(f"counter_GH_analysis = {counter_GH_analysis}")
        # print(f"df_isobar:\n{df_isobar[['Config_folder', 'G_hp_per_atom', 'G_hp_per_atom_w_TS', 'Target temperature (K)', 'Target pressure (GPa)']]}\n")
        # print(f"df_GH_analysis:\n{GH_df[['GFE', 'T_target', 'P_target']]}\n")

        # append df_isobar to df_isobar_superset
        df_isobar_superset = pd.concat([df_isobar_superset, df_isobar], ignore_index=True)

        print("\n"*4)


    for temp, sub in df_isobar_superset.groupby("Target temperature (K)"):
        # sort sub by X_{secondary_species}
        sub.sort_values(f"X_{secondary_species}", inplace=True)
        # size of sub
        # print(f"Processing temperature {temp} K with {len(sub)} entries for {secondary_species}...")
        # name of the Config_folder
        # print(f"Config_folder: {sub['Config_folder'].values}")

        x = sub[f"X_{secondary_species}"].values
        x_error = x * 0
        y = sub["G_hp_per_atom_w_TS"].values
        y_error = sub["G_hp_per_atom_w_TS_error"].values
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)#, cov=True)  # fit y = a + b*x
            # print(f"1: slope={slope}, intercept={intercept}")
            
            fn_slope = lambda y1, y0, x1, x0: (y1 - y0) / (x1 - x0)  # calculate slope
            slope, slope_error = monte_carlo_error(fn_slope, [y[1], y[0], x[1], x[0]], [y_error[1], y_error[0], x_error[1], x_error[0]])
            fn_intercept = lambda y, slope, x: y - slope * x  # calculate intercept
            intercept, intercept_error = monte_carlo_error(fn_intercept, [y[0], slope, x[0]], [y_error[0], slope_error, x_error[0]])
        else:
            intercept, slope = np.nan, np.nan
            slope_error, intercept_error = np.nan, np.nan
        # Update the df_isobar_superset with a and b
        mask = (df_isobar_superset["Target temperature (K)"] == temp)

        if temp in df_subset["Target temperature (K)"].values:
            # print(f"Skipping T={temp} K as it is a primary temperature.")
            continue
        df_isobar_superset.loc[mask, "intercept"] = intercept
        df_isobar_superset.loc[mask, "slope"] = slope
        df_isobar_superset.loc[mask, "intercept_error"] = intercept_error
        df_isobar_superset.loc[mask, "slope_error"] = slope_error
        
        # Update mu_excess_{secondary_species} = slope + intercept
        fn_mu_excess = lambda slope, intercept: slope + intercept
        mu_excess, mu_excess_error = monte_carlo_error(fn_mu_excess, [slope, intercept], [slope_error, intercept_error])
        df_isobar_superset.loc[mask, f"mu_excess_{secondary_species}"] = mu_excess
        df_isobar_superset.loc[mask, f"mu_excess_{secondary_species}_error"] = mu_excess_error
        # print(f"Computed mu_excess for T={temp} K: {mu_excess:.3f} ± {mu_excess_error:.3f}")

        # # loop through X_{secondary_species} and compute mu_TS_term for each
        # for X in sub[f"X_{secondary_species}"]:
        #     T = sub.loc[sub[f"X_{secondary_species}"] == X, "Target temperature (K)"].values[0]
        #     # X = df_isobar_superset.loc[mask, f"X_{secondary_species}"].values[0]
        #     Config_folder = sub.loc[sub[f"X_{secondary_species}"] == X, "Config_folder"].values[0]
        #     # print(f"Computing mu_TS_term for T={T} K, X={X} for {secondary_species}, Config_folder: {Config_folder}")
        #     mu_TS_term, mu_TS_term_error = monte_carlo_error(compute_mu_TS, [T, X], [0.0, 0.0])  # assuming no error in T and X
        #     df_isobar_superset.loc[mask, f"mu_{secondary_species}_TS_term"] = mu_TS_term
        #     df_isobar_superset.loc[mask, f"mu_{secondary_species}_TS_term_error"] = mu_TS_term_error

        # fn_mu = lambda mu_excess, mu_TS_term: mu_excess + mu_TS_term
        # mu, mu_error = monte_carlo_error(fn_mu, [mu_excess, mu_TS_term], [mu_excess_error, mu_TS_term_error])
        # df_isobar_superset.loc[mask, f"mu_{secondary_species}"] = mu
        # df_isobar_superset.loc[mask, f"mu_{secondary_species}_error"] = mu_error


    # print df_isobar_superset
    # sort wrt Config_folder and then temperature
    df_isobar_superset.sort_values(["Config_folder", "Target temperature (K)"], inplace=True)
    # print(f"df_isobar_superset for {phase} at {pt}:\n{df_isobar_superset[['Config_folder', 'G_hp','G_hp_per_atom', 'G_hp_per_atom_w_TS', 'Target temperature (K)', 'Target pressure (GPa)', f'mu_{secondary_species}', f'mu_{secondary_species}_error']]}\n")
    print(f"df_isobar_superset for {phase} at {pt}:\n{df_isobar_superset[['Config_folder', 'G_hp','G_hp_per_atom', 'G_hp_per_atom_w_TS', f'mu_excess_{secondary_species}', f'mu_excess_{secondary_species}_error']]}\n")


    # check if df_isobar_superset and df have same columns
    if not set(df_isobar_superset.columns).issubset(set(df.columns)):
        print(f"WARNING: df_isobar_superset has columns not in df. This is unexpected. df_isobar_superset columns: {df_isobar_superset.columns.tolist()}\n")
        print(f"df columns: {df.columns.tolist()}\n")
        print("Exiting to avoid data inconsistency.")
        exit(1)
    # else:
    #     print(f"df_isobar_superset has all columns in df. Proceeding to append.\n")

    # append df_isobar_superset to df_superset
    df_superset = pd.concat([df_superset, df_isobar_superset], ignore_index=True)

    print("\n"*8)

    # exit(0)


# run if KD_sil_to_metal and KD_sil_to_metal_error exist in df_superset
if "KD_sil_to_metal" in df_superset.columns and "KD_sil_to_metal_error" in df_superset.columns:
    # calculate KD, D_wt
    for i, row in df_superset.iterrows():
        # print(f"Computing KD_sil_to_metal for row {i}: {row['Phase']} at {row['P_T_folder']}")
        KD, KD_error = compute_KD(row)
        df_superset.at[i, "KD_sil_to_metal"] = KD
        df_superset.at[i, "KD_sil_to_metal_error"] = KD_error
        # print(f"Computed KD_sil_to_metal for row {i}: {KD:.3f} ± {KD_error:.3f} for {row['Phase']} at {row['P_T_folder']}\n")
    df_superset["D_wt"] = df_superset["KD_sil_to_metal"] * (100/56)
    df_superset["D_wt_error"] = df_superset["KD_sil_to_metal_error"] * (100/56)



# add df to df_superset
df_superset = pd.concat([df_superset, df], ignore_index=True)



# sort df_superset by Phase, Config_folder and Target temperature (K)
df_superset.sort_values(["Phase", "Config_folder", "Target temperature (K)"], inplace=True)

df_superset_wo_nan = df_superset.dropna(subset=["G_hp_per_atom_w_TS", "mu_excess_" + secondary_species, "mu_excess_" + secondary_species + "_error", "KD_sil_to_metal", "KD_sil_to_metal_error", "D_wt", "D_wt_error"])

# print(f"df_superset:\n{df_superset[['Phase','Config_folder','G_hp_per_atom', 'G_hp_per_atom_w_TS', 'Target temperature (K)', 'Target pressure (GPa)', f'mu_excess_{secondary_species}', f'mu_excess_{secondary_species}_error', 'KD_sil_to_metal', 'KD_sil_to_metal_error', 'D_wt', 'D_wt_error']]}\n")
print(f"df_superset_wo_nan:\n{df_superset_wo_nan[['Phase','Config_folder','G_hp_per_atom', 'G_hp_per_atom_w_TS', 'Target temperature (K)', 'Target pressure (GPa)', f'mu_excess_{secondary_species}', f'mu_excess_{secondary_species}_error', 'KD_sil_to_metal', 'KD_sil_to_metal_error', 'D_wt', 'D_wt_error']]}\n")

print("\nWARNING: Have a closer look at G_hp/GFE values (isobar_calc) to see if they are monotonic or not!!\n")
print("\nWARNING: Is Ghp from Gibbs-Helmholtz analysis with or without TS? Assuming it is not for now. If it has TS, isobar_Ghp_analysis.sh will also need to be updated.\n")





#########
# df = df_superset.copy()
#########




# 11) Save the assembled table to CSV
pd.set_option("display.width", 200)
df.to_csv("all_TI_results.csv", index=False)
print(f"Wrote all_TI_results.csv with {len(df)} rows.")














# plot X_{secondary_species} vs G_hp_per_atom_w_TS, and color by P_T_folder and size by phase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# 1) Load your assembled TI results (with columns: Phase, P_T_folder, X_{secondary_species}, G_hp_per_atom_w_TS, a, b, etc.)
df = pd.read_csv("all_TI_results.csv")

# 2) Build a discrete colormap only for the P_T_folder categories actually present
folder_cats = df["P_T_folder"].astype("category")
orig_codes  = folder_cats.cat.codes.values            # original integer codes
used_codes  = np.unique(orig_codes)                   # codes that actually appear
remap       = {old: new for new, old in enumerate(used_codes)}
mapped_codes = np.vectorize(remap.get)(orig_codes)    # remapped to 0..N-1

base_cmap = plt.get_cmap("tab10")
colors    = [base_cmap(old) for old in used_codes]
cmap      = ListedColormap(colors)
norm      = BoundaryNorm(np.arange(len(used_codes)+1)-0.5, len(used_codes))

# 3) Marker size and opacity by phase
size_map  = {f"Fe_{secondary_species}": 200, f"MgSiO3_{secondary_species}": 100}
alpha_map = {f"Fe_{secondary_species}": 0.5,   f"MgSiO3_{secondary_species}": 1.0}

# 4) Create figure and axes
fig, ax = plt.subplots(figsize=(8, 20))

# 5) Prepare a ScalarMappable for consistent coloring
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# 6) Scatter plot of the data
for phase, sub in df.groupby("Phase"):
    # get the per‐point colors from your remapped codes
    point_colors = [cmap(norm(code)) for code in mapped_codes[sub.index]]

    # scatter the main points
    ax.scatter(
        sub[f"X_{secondary_species}"],
        sub["G_hp_per_atom_w_TS"],
        color=point_colors,
        s=size_map[phase],
        alpha=alpha_map[phase],
        label=phase
    )

    # now loop to draw each errorbar with its own color
    for x0, y0, err0, c in zip(
        sub[f"X_{secondary_species}"],
        sub["G_hp_per_atom_w_TS"],
        sub["G_hp_per_atom_w_TS_error"],
        point_colors
    ):
        ax.errorbar(
            x0, y0,
            yerr=err0,
            fmt='none',    # no extra marker
            ecolor=c,
            elinewidth=1,
            capsize=3,
            alpha=alpha_map[phase]
        )

# # finish up: legend, colorbar, labels, etc.
# ax.legend(title="Phase")
# cbar = fig.colorbar(sm, ticks=used_codes, ax=ax)
# cbar.ax.set_yticklabels(folder_cats.cat.categories[used_codes])
# ax.set_xlabel(f"X_{secondary_species}")
# ax.set_ylabel("G_hp_per_atom_w_TS")

# 7) Overlay linear fits y = a + b*x for each (Phase, P_T_folder)
for (phase, pt), sub in df.groupby(["Phase", "P_T_folder"]):
    a = sub["intercept"].iloc[0]
    b = sub["slope"].iloc[0]
    x_line = np.linspace(sub[f"X_{secondary_species}"].min(), sub[f"X_{secondary_species}"].max(), 200)
    # use the same norm+cmap via sm.to_rgba on the original code
    orig_code = folder_cats.cat.categories.get_loc(pt)
    line_color = sm.to_rgba(orig_code)
    ax.plot(
        x_line,
        a + b * x_line,
        color=line_color,
        linestyle="--",
        linewidth=2
    )

# 8) Add a discrete colorbar (only showing used categories), shrunk and semi-transparent
cbar = fig.colorbar(
    sm,
    boundaries=np.arange(len(used_codes)+1)-0.5,
    ticks     = np.arange(len(used_codes)),
    ax        = ax,
    shrink    = 1.0,    # 70% length
    aspect    = 40      # thinner bar
)
cbar.solids.set_alpha(0.5)
cbar.ax.set_yticklabels([folder_cats.cat.categories[i] for i in used_codes])
cbar.set_label("P_T_folder")

# 9) Final styling
ax.set_xlabel(f"X_{secondary_species}")
ax.set_ylabel("G_hp_per_atom_w_TS (eV)")
ax.set_title(f"X_{secondary_species} vs G_hp_per_atom_w_TS\nColored by P_T_folder, Sized/Alpha by Phase")
ax.legend(title="Phase")
ax.grid(True)
plt.tight_layout()

# 10) Save and/or show
plt.savefig(f"X_{secondary_species}_vs_G_hp_per_atom_w_TS.png", dpi=300)
# plt.show()








# narrow df to Fe_{secondary_species} phase and P_T_folder = P50_T3500
# df = df[ (df["P_T_folder"] == "P50_T3500")]

# plot mu_{secondary_species} vs X_{secondary_species}, and color by P_T_folder and size by phase
# --- Prepare the discrete colormap for the used P_T_folders ---
folder_cats  = df["P_T_folder"].astype("category")
orig_codes   = folder_cats.cat.codes.values
used_codes   = np.unique(orig_codes)
remap        = {old: new for new, old in enumerate(used_codes)}
mapped_codes = np.vectorize(remap.get)(orig_codes)
base_cmap = plt.get_cmap("tab10")
colors    = [base_cmap(old) for old in used_codes]          
cmap      = ListedColormap(colors)
norm      = BoundaryNorm(np.arange(len(used_codes)+1)-0.5, len(used_codes))
# --- Size & opacity per phase ---
size_map  = {f"Fe_{secondary_species}": 200, f"MgSiO3_{secondary_species}": 100}
alpha_map = {f"Fe_{secondary_species}": 0.5,  f"MgSiO3_{secondary_species}": 1.0}
# --- Make the figure & axes ---
fig, ax = plt.subplots(figsize=(10,10))
# 1) Scatter the raw data, grouping by Phase so we get two sizes/alphas
for phase, sub in df.groupby("Phase"):
    ax.scatter(
        sub[f"X_{secondary_species}"], sub[f"mu_{secondary_species}"],
        # sub[f"X_{secondary_species}"], sub[f"mu_excess_{secondary_species}"],
        c=mapped_codes[sub.index],      # use remapped folder codes
        cmap=cmap, norm=norm,
        s=size_map[phase],
        alpha=alpha_map[phase],
        label=phase
    )
    # print(f"phase = {phase}")
    # print(f"sub['X_{secondary_species}'] = {sub[f'X_{secondary_species}']}")
    # print(f"sub['mu_{secondary_species}'] = {sub[f'mu_{secondary_species}']}")
# # 2) Overlay the fitted lines for each (Phase, P_T_folder)
# #    We pick the same x-range per folder so the line spans the full data
# for (phase, pt), sub in df.groupby(["Phase","P_T_folder"]):
#     a = sub["intercept"].iloc[0]
#     b = sub["slope"].iloc[0]
#     # line x from min to max of that subgroup
#     x_line = np.linspace(sub[f"X_{secondary_species}"].min(), sub[f"X_{secondary_species}"].max(), 200)
#     # get the original code for this folder, then map→0..N-1 for color lookup
#     orig_code = folder_cats.cat.categories.get_loc(pt)
#     mcode     = remap[orig_code]
#     ax.plot(
#         x_line, a + b*x_line,
#         color=cmap(mcode),
#         linestyle="--",
#         linewidth=2#,
#         # label=f"{phase}, {pt} fit"
#     )
# 3) Colorbar for the folders
# 3) Colorbar for the folders, shrunk to 70% height and with 50% opacity
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    boundaries=np.arange(len(used_codes)+1)-0.5,
    ticks     = np.arange(len(used_codes)),
    ax        = ax,
    shrink    = 1.0,    # make it 70% as tall as the axes
    aspect    = 40      # make it thinner (default is ~20)
)

# set only the color patches semi-transparent
cbar.solids.set_alpha(0.5)

# relabel
cbar.ax.set_yticklabels([folder_cats.cat.categories[i] for i in used_codes])
cbar.set_label("P_T_folder")
# diagnol cbar labels
# for label in cbar.ax.get_yticklabels():
#     label.set_rotation(45)
#     label.set_horizontalalignment('right')
#     label.set_verticalalignment('center')
#     label.set_fontsize(8)

# 4) Final styling
ax.set_xlabel(f"X_{secondary_species}")
ax.set_ylabel(f"mu_{secondary_species} (eV)")
ax.set_title(f"X_{secondary_species} vs mu_{secondary_species} with Line Fits by Phase+P_T_folder")
ax.legend()
# ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small", title="Legend")
ax.grid(True)
plt.tight_layout()
plt.savefig(f"X_{secondary_species}_vs_mu_{secondary_species}.png")







fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
# ax1, ax2, ax3, ax4 = axes.flatten()
ax2, ax4 = axes.flatten()

# 2) Common settings
# grab the default color cycle
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# create a base color dictionary associated with "P_T_folder" values
base_color = {
    "P50_T3500": default_colors[0],
    "P250_T6500": default_colors[1],
    "P500_T9000": default_colors[2],
    "P1000_T13000": default_colors[3]
}
marker_opts = dict(marker='o', linestyle='', markersize=10, alpha=0.5)#,color=base_color)
marker_opts_error = dict(marker='o', linestyle='', markersize=10, alpha=0.5, capsize=3, elinewidth=1)
marker_opts__others = dict(marker='s', linestyle='', markersize=15, alpha=0.25)

x_variable = "Target pressure (GPa)"  # x-axis variable for all plots
z_variable = "Target temperature (K)"  # z-axis variable for all plots -- color coding

# # --- Panel 1: KD_sil_to_metal (linear) ---
# ax1.plot(df[x_variable], df["KD_sil_to_metal"], **marker_opts, color=base_color[df["P_T_folder"].iloc[0]])
# ax1.errorbar(df[x_variable], df["KD_sil_to_metal"], yerr=df["KD_sil_to_metal_error"], **marker_opts_error, color=base_color[df["P_T_folder"].iloc[0]])
# ax1.set_ylabel("K_D (silicate → metal)")
# # ax1.set_title("Partition Coefficient (linear)")
# ax1.grid(True)
# # remove x-tick labels on top row
# ax1.tick_params(labelbottom=False)

# # --- Panel 2: D_wt (linear) ---
# ax3.plot(df[x_variable], df["D_wt"], **marker_opts)
# ax3.errorbar(df[x_variable], df["D_wt"], yerr=df["D_wt_error"], **marker_opts_error)
# ax3.set_ylabel("D_wt (silicate → metal)")
# # ax3.set_title("Distribution Coefficient (linear)")
# ax3.grid(True)
# # x label
# ax3.set_xlabel("P, T")
# # rotate bottom x-labels

# --- Panel 3: KD_sil_to_metal (log y) ---
ax2.plot(df[x_variable], df["KD_sil_to_metal"], **marker_opts)
ax2.errorbar(df[x_variable], df["KD_sil_to_metal"], yerr=df["KD_sil_to_metal_error"], **marker_opts_error)
ax2.set_yscale("log")
ax2.set_ylabel("K_D (silicate → metal; log scale)")
# ax2.set_title("Partition Coefficient (log)")
ax2.grid(True)
# ax2.tick_params(labelbottom=False)
ax2.set_xlabel("Pressure (GPa)")


# --- Panel 4: D_wt (log y) ---
ax4.plot(df[x_variable], df["D_wt"], **marker_opts)
ax4.errorbar(df[x_variable], df["D_wt"], yerr=df["D_wt_error"], **marker_opts_error)
ax4.set_yscale("log")
ax4.set_ylabel("D_wt (silicate → metal; log scale)")
# ax4.set_title("Distribution Coefficient (log)")
ax4.grid(True)
ax4.set_xlabel("Pressure (GPa)")

# super title
fig.suptitle(f"Partition Coefficient (K_D) and Weight Distribution Coefficient (D_wt) for {secondary_species} in Fe and MgSiO3. \n Note: Assumption that X_{secondary_species} in silicates is << 1", fontsize=12)


# if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
if secondary_species == "He":
    # ax1.plot(500, 0.032, **marker_opts)
    # ax1.plot(500, 0.07, **marker_opts)
    # ax1.plot(1000, 1, **marker_opts)
    # ax1.plot(1000, 0.32, **marker_opts)
    ax2.plot(500, 0.032, **marker_opts)
    ax2.plot(500, 0.07, **marker_opts)
    # ax2.plot(1000, 1, **marker_opts)
    ax2.plot(1000, 0.32, **marker_opts)
    # ax3.plot(500, 0.032*1.78, **marker_opts)
    # ax3.plot(500, 0.07*1.78, **marker_opts)
    # ax3.plot(1000, 1*1.78, **marker_opts)
    # ax3.plot(1000, 0.32*1.78, **marker_opts)
    ax4.plot(500, 0.032*1.78, **marker_opts)
    ax4.plot(500, 0.07*1.78, **marker_opts)
    # ax4.plot(1000, 1*1.78, **marker_opts)
    ax4.plot(1000, 0.32*1.78, **marker_opts)

# previous studies
if secondary_species == "He":
    # P50_T3500: 2.6E-3
    # P135_T4200: 8.7E-4
    # ax1.plot(50, 2.6E-3, **marker_opts__others, color='red', label="Previous Study: P50_T3500")
    # ax1.plot(135, 8.7E-4, **marker_opts__others, color='red', label="Previous Study: P135_T4200")
    ax2.plot(50, 2.6E-3, **marker_opts__others, color='red', label="Previous Study: P50_T3500")
    ax2.plot(135, 8.7E-4, **marker_opts__others, color='red', label="Previous Study: P135_T4200")
    # ax3.plot(50, 2.6E-3*1.78, **marker_opts__others, color='red', label="Previous Study: P50_T3500")
    # ax3.plot(135, 8.7E-4*1.78, **marker_opts__others, color='red', label="Previous Study: P135_T4200")
    ax4.plot(50, 2.6E-3*1.78, **marker_opts__others, color='red', label="Previous Study: P50_T3500")
    ax4.plot(135, 8.7E-4*1.78, **marker_opts__others, color='red', label="Previous Study: P135_T4200")


if secondary_species == "H":
    # Luo et al.
    # P500_T9000: 10**1.4
    # P1000_T13000: 10**1.8
    # ax1.plot(500, 10**1.4, **marker_opts__others, color='red', label="Luo et al. P500_T9000")
    # ax1.plot(1000, 10**1.8, **marker_opts__others, color='red', label="Luo et al. P1000_T13000")
    ax2.plot(500, 10**1.4, **marker_opts__others, color='red', label="Luo et al. P500_T9000")
    ax2.plot(1000, 10**1.8, **marker_opts__others, color='red', label="Luo et al. P1000_T13000")
    # ax3.plot(500, 10**1.4*1.78, **marker_opts__others, color='red', label="Luo et al. P500_T9000")
    # ax3.plot(1000, 10**1.8*1.78, **marker_opts__others, color='red', label="Luo et al. P1000_T13000")
    ax4.plot(500, 10**1.4*1.78, **marker_opts__others, color='red', label="Luo et al. P500_T9000")
    ax4.plot(1000, 10**1.8*1.78, **marker_opts__others, color='red', label="Luo et al. P1000_T13000")

    # previous study
    # P50_T3500: 9.1
    # ax1.plot(50, 9.1, **marker_opts__others, color='blue', label="Previous Study: P50_T3500")
    ax2.plot(50, 9.1, **marker_opts__others, color='blue', label="Previous Study: P50_T3500")
    # ax3.plot(50, 9.1*1.78, **marker_opts__others, color='blue', label="Previous Study: P50_T3500")
    ax4.plot(50, 9.1*1.78, **marker_opts__others, color='blue', label="Previous Study: P50_T3500")

# x lim ,  y lim
if secondary_species == "He":
    y_min = 1e-4
    y_max = 1e1
elif secondary_species == "H":
    y_min = 1e-1
    y_max = 1e3

# ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)
# ax3.set_ylim(y_min, y_max)
ax4.set_ylim(y_min, y_max)


# 3) Layout & save
plt.tight_layout()
plt.savefig(f"KD_D_wt_vs_P_T.png", dpi=300)




















# Create a plot with 4 panels corresponding to data from all rows that correspond each to the 4 unique P_T_folders, showing KD_sil_to_metal vs Target temperature (K)

# df_plot = df.copy()

# refine df_superset_wo_nan to those with unique pairs of P_T_folders and Target temperature (K)
df_plot = df_superset_wo_nan.copy()
df_plot = df_plot.drop_duplicates(subset=["P_T_folder", "Target temperature (K)"])

unique_P_T_folders = df_plot["P_T_folder"].unique()


fig, axes = plt.subplots(2, 2, figsize=(10, 8))#, sharex=True, sharey=True)


for ith_P_T_folder in unique_P_T_folders:

    # narrow df_plot to the current P_T_folder
    df_temp = df_plot[df_plot["P_T_folder"] == ith_P_T_folder]

    ith_P_T_folder_Pressure = df_temp["Target pressure (GPa)"].iloc[0]

    # get the axes for the current P_T_folder
    ax = axes.flatten()[list(unique_P_T_folders).index(ith_P_T_folder)]

    # plot KD_sil_to_metal vs Target temperature (K)
    ax.plot(df_temp["Target temperature (K)"], df_temp["KD_sil_to_metal"], marker='o', linestyle='', markersize=10, alpha=0.5, label=f"{ith_P_T_folder_Pressure} (GPa)")
    ax.errorbar(df_temp["Target temperature (K)"], df_temp["KD_sil_to_metal"], yerr=df_temp["KD_sil_to_metal_error"], marker='o', linestyle='', markersize=10, alpha=0.5, capsize=3, elinewidth=1)

    # set y scale to log
    ax.set_yscale("log")
    
    # set title and labels
    ax.set_title(f"{ith_P_T_folder}")
    ax.set_xlabel("Target Temperature (K)")
    ax.set_ylabel("KD_sil_to_metal (log scale)")
    
    # remove x axis labels for the top row and y axis labels for the right column
    # if ith_P_T_folder in unique_P_T_folders[:2]:  # top row
    #     ax.tick_params(labelbottom=False)  # remove x-tick labels
    #     ax.set_xlabel("")
    # if ith_P_T_folder in unique_P_T_folders[1::2]:  # right column
    #     ax.tick_params(labelleft=False)  # remove y-tick labels
    #     ax.set_ylabel("")

    # grid
    ax.grid(True)

    # # add a colorbar for Target temperature (K) wrt hotness
    # scatter = ax.scatter(df_temp["Target Temperature (K)"], df_temp["KD_sil_to_metal"], c=df_temp["Target Temperature (K)"], cmap='hot', norm=plt.Normalize(vmin=df_temp["Target Temperature (K)"].min(), vmax=df_temp["Target Temperature (K)"].max()), alpha=0.5)
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label("Target temperature (K)")

# adjust layout
plt.tight_layout()
# save the figure
plt.savefig(f"KD_sil_to_metal_vs_P_T_panels.png", dpi=300)


# plot all mu_{secondary_species} vs X_{secondary_species}, and color by P_T_folder
# plt.figure(figsize=(10,6))
# plt.scatter(
#     df[f"X_{secondary_species}"], df[f"mu_{secondary_species}"],
#     s=100,
#     alpha=0.5
# )
# plt.savefig("test.png")

print(f"Created: dataframe with G_hp_per_atom, G_hp_per_atom_error, X_{secondary_species}, etc. from all systems")
print(f"Files created: all_TI_results_with_X{secondary_species}.csv, X_{secondary_species}_vs_G_hp_per_atom.png")