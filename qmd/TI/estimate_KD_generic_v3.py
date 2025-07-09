#!/usr/bin/env python3
"""
Estimate chemical potentials and partition coefficients/equilibrium constants for {secondary_species} in Fe and MgSiO3 systems from TI data.
Walks through directory structure, parses log.Ghp_analysis files, assembles results into a DataFrame,
computes mixing fractions, fits linear excess chemical potentials, and adds entropy corrections.
Partially based on the discussion in the paper Li, et al. 2022 paper, "Primitive noble gases sampled from ocean island basalts cannot be from the Earth's core".

v2: Evaluates isobar_calc
v3: Does asymmetric error propagation for KD, etc.

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
from mc_error import monte_carlo_error_asymmetric # located at $HELP_SCRIPTS/general/mc_error.py
from scipy.optimize import curve_fit

# Boltzmann constant in eV/K for entropy term
kB = 8.617333262145e-5


SCRIPT_MODE = 1 # >0, only plot; <0, only do analysis; 0: both
PLOT_MODE=7 #-1: plot all; 0: do not plot, 1: plot #1, 2: plot #2, 3: plot #3 ...




# 1) At the top of your script, snapshot the initial settings
INITIAL_RCP = plt.rcParams.copy()


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


print(f"Secondary species: {secondary_species}")



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



if SCRIPT_MODE <= 0:

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


        # Compute mu_excess = a + b (no X_{secondary_species} factor)
        fn_mu_excess = lambda slope, intercept: slope + intercept
        mu_excess, mu_excess_error = monte_carlo_error(fn_mu_excess, [slope, intercept], [slope_error, intercept_error])
        # mu_excess = fn_mu_excess(slope, intercept)  # compute mu_excess
        df.loc[mask, f"mu_excess_{secondary_species}"] = mu_excess
        df.loc[mask, f"mu_excess_{secondary_species}_error"] = mu_excess_error

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
            return kB * T * np.log(X)
        elif row["Phase"] == f"MgSiO3_{secondary_species}":
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
        # df.at[i, f"mu_{secondary_species}_TS_term"] = compute_mu_TS(T, X)
        mu_TS_term, mu_TS_term_error = monte_carlo_error(compute_mu_TS, [T, X], [0.0, 0.0])  # assuming no error in T and X
        df.at[i, f"mu_{secondary_species}_TS_term"] = mu_TS_term
        df.at[i, f"mu_{secondary_species}_TS_term_error"] = mu_TS_term_error


        fn_mu = lambda mu_excess, mu_TS_term: mu_excess + mu_TS_term
        mu, mu_error = monte_carlo_error(fn_mu, [row[f"mu_excess_{secondary_species}"], mu_TS_term], [row[f"mu_excess_{secondary_species}_error"], mu_TS_term_error])
        df.at[i, f"mu_{secondary_species}"] = mu
        df.at[i, f"mu_{secondary_species}_error"] = mu_error






    # partiction coefficient: (1.78/5) * np.exp(-(mu_excess_{secondary_species}_for_Fe - mu_excess_{secondary_species}_for_MgSiO3) / (kB * T)) for the same P_T_folder
    def compute_KD(row,df_evaluate=None):
        if df_evaluate is None:
            raise ValueError("df_evaluate must be provided to compute_KD")
        # 1) Identify this row’s phase & P_T group
        phase = row["Phase"]
        pt    = row["P_T_folder"]

        # 2) Determine the other phase
        other_phase = f"MgSiO3_{secondary_species}" if phase == f"Fe_{secondary_species}" else f"Fe_{secondary_species}"

        
        # 3) Grab that phase’s mu_excess_{secondary_species} for the same P_T_folder and same temperature
        mask = (
                (df_evaluate["Phase"] == other_phase) &
                (df_evaluate["P_T_folder"] == pt) &
                (df_evaluate["Target temperature (K)"] == row["Target temperature (K)"])
                )
        partner__mu_excess = df_evaluate.loc[mask, f"mu_excess_{secondary_species}"]
        partner__mu_excess_error = df_evaluate.loc[mask, f"mu_excess_{secondary_species}_error"]

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
        # KD, KD_error = monte_carlo_error(fn_KD, [mu_excess, other__mu_excess, T, mult_factor], [mu_excess_error, other__mu_excess_error, 0.0, 0.0])
        KD, KD_error, KD_lower, KD_upper = monte_carlo_error_asymmetric(
            fn_KD,
            [mu_excess, other__mu_excess, T, mult_factor],
            [mu_excess-mu_excess_error, other__mu_excess-other__mu_excess_error, T-0.0, mult_factor-0.0],
            [mu_excess+mu_excess_error, other__mu_excess+other__mu_excess_error, T+0.0, mult_factor+0.0],
        )

        return KD, KD_error, KD_lower, KD_upper

    # Apply it
    # df["KD_sil_to_metal"] = df.apply(compute_KD, axis=1)
    df["KD_sil_to_metal"] = np.nan
    df["KD_sil_to_metal_error"] = np.nan
    for i, row in df.iterrows():
        KD, KD_error, KD_lower, KD_upper = compute_KD(row, df)
        df.at[i, "KD_sil_to_metal"] = KD
        df.at[i, "KD_sil_to_metal_error"] = KD_error
        df.at[i, "KD_sil_to_metal_low"] = KD_lower
        df.at[i, "KD_sil_to_metal_high"] = KD_upper
        # print(f"Computed KD_sil_to_metal for row {i}: {KD:.3f} ± {KD_error:.3f} for {row['Phase']} at {row['P_T_folder']}\n")
    df["D_wt"] = df["KD_sil_to_metal"] * (100/56)
    df["D_wt_error"] = df["KD_sil_to_metal_error"] * (100/56)



    # # 11) Print summary of results
    # print(f"\nKD_sil_to_metal for {secondary_species} in Fe and MgSiO3 systems:")
    # # Loop over each unique (P_T_folder, Phase) combination
    # for (folder, phase), sub_df in df.groupby(["P_T_folder", "Phase"]):
    #     # grab all non-NaN KD values
    #     kd_vals = sub_df["KD_sil_to_metal"].dropna()
    #     if not kd_vals.empty:
    #         kd0 = kd_vals.iloc[0]
    #         mu0 = sub_df[f"mu_excess_{secondary_species}"].iloc[0]
    #         print(f"{folder}, Phase={phase}:")
    #         print(f"                                KD_sil_to_metal = {kd0:.4f} ± {sub_df['KD_sil_to_metal_error'].iloc[0]:.4f} (mu_excess = {mu0:.4f} ± {sub_df[f'mu_excess_{secondary_species}_error'].iloc[0]:.4f})")
    #         # print(f"                                slope = {sub_df['slope'].iloc[0]:.4f} ± {sub_df['slope_error'].iloc[0]:.4f}, intercept = {sub_df['intercept'].iloc[0]:.4f} ± {sub_df['intercept_error'].iloc[0]:.4f}")
    #     else:
    #         print(f"{folder}, Phase={phase}: "
    #                 "No valid KD_sil_to_metal values found.")


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



    # sort df -- reverserd order of columns
    # df.sort_values(by=["Phase", "Target pressure (GPa)"], ascending=[True, False], inplace=True)


    counter_GH_analysis = 0
    df_superset= pd.DataFrame()

    counter_err_update=0


    # Group by Phase and P_T_folder to fit separate lines
    for (phase, pt), df_subset in df.groupby(["Phase", "P_T_folder"]):

        # create an empty df_isobar_superset
        df_isobar_superset = pd.DataFrame()

        # PT_dir = parent dir of df_subset["Base directory"][0]
        PT_dir = Path(df_subset["Base directory"].iloc[0]).parent
        print(f"Processing Phase: {phase}, P_T_folder: {pt} in directory: {PT_dir}")


        ## if PT_dirname P250_T6500, skip
        PT_dirname = PT_dir.name

        # check if there are two GH_analysis.csv files in PT_dir: find . -type f -name "GH_analysis.csv" | wc -l
        num_GH_analysis_files = len(list(PT_dir.glob("**/GH_analysis.csv")))
        
        if num_GH_analysis_files < 2:
            print(f"*** Skipping {phase} at {pt} as there are not enough GH_analysis.csv files (found {num_GH_analysis_files}). ***\n\n")
            continue
        # else:
        #     print(f"Found {num_GH_analysis_files} GH_analysis.csv files.\n\n")


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

            print(f"Processing isobar_calc for {CONFIG_dirname} ...\n")
            # print("="*10)


            # Make 5 duplicate rows based on all df_base entries, and save this in df_isobar.
            df_isobar = df_base.copy()
            df_isobar = pd.concat([df_isobar] * 4, ignore_index=True) # 4 new ones/isobar + the original

            # make all values nan
            # df_isobar.loc[:, df_isobar.columns != "Config_folder"] = np.nan

            # 1) grab only “T…” dirs
            temp_dirs = [
                            d for d in isobar_dir.iterdir()
                            if d.is_dir()
                            and d.name.startswith("T")
                            and d.name[1:].isdigit()   # everything after the "T" is numeric
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
            # print(f"GH_analysis_file: {GH_analysis_file}")
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

            # There, go into each directory beginning with "Target temperature (K)". Whatever comes after "Target temperature (K)" is the temperature in K.
            # 3) enumerate and fill your DataFrame
            for idx, temp_dir in enumerate(temp_dirs):
                temp_val = _parse_temp(temp_dir)
                if temp_val == float('inf'):
                    print(f"Skipping {temp_dir!r}: invalid temperature")
                    continue

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



            # sort wrt temperature
            df_isobar.sort_values("Target temperature (K)", inplace=True)
            GH_df.sort_values("T_target", inplace=True)

            # append df_isobar to df_isobar_superset
            df_isobar_superset = pd.concat([df_isobar_superset, df_isobar], ignore_index=True)

            # print("\n"*4)


        for temp, sub in df_isobar_superset.groupby("Target temperature (K)"):
            # sort sub by X_{secondary_species}
            sub.sort_values(f"X_{secondary_species}", inplace=True)


            x = sub[f"X_{secondary_species}"].values
            x_error = x * 0
            y = sub["G_hp_per_atom_w_TS"].values
            y_error = sub["G_hp_per_atom_w_TS_error"].values
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)#, cov=True)  # fit y = a + b*x


                if x[0] == x[1]:
                    # if the first two x values are the same, we cannot compute slope and intercept
                    slope, intercept = np.nan, np.nan
                    slope_error, intercept_error = np.nan, np.nan
                    continue
                
                fn_slope = lambda y1, y0, x1, x0: (y1 - y0) / (x1 - x0)  # calculate slope
                slope, slope_error = monte_carlo_error(fn_slope, [y[1], y[0], x[1], x[0]], [y_error[1], y_error[0], x_error[1], x_error[0]])
                fn_intercept = lambda y, slope, x: y - slope * x  # calculate intercept
                intercept, intercept_error = monte_carlo_error(fn_intercept, [y[0], slope, x[0]], [y_error[0], slope_error, x_error[0]])

                # Assuming fractional errors in slope and intercept for Gibbs-Helmholtz analysis are same as that for TI!
                slope_error_new = slope_error * 0
                intercept_error_new = intercept_error * 0
                # grab slope and intercept errors of calc_type=TI, and same P_T_folder, Phase and X_{secondary_species}
                mask = (df["Phase"] == phase) & (df["P_T_folder"] == pt) & (df["calc_type"] == "TI") & df["Config_folder"].str.contains("_8H")
                if not df.loc[mask, "slope"].empty and not df.loc[mask, "intercept"].empty and not df.loc[mask, "slope_error"].empty and not df.loc[mask, "intercept_error"].empty:
                    slope_TI = df.loc[mask, "slope"].values[0]
                    intercept_TI = df.loc[mask, "intercept"].values[0]
                    slope_error_TI = df.loc[mask, "slope_error"].values[0]
                    intercept_error_TI = df.loc[mask, "intercept_error"].values[0]
                    slope_error = np.abs((slope_error_TI/slope_TI) * slope)
                    intercept_error = np.abs((intercept_error_TI/intercept_TI) * intercept)
                    counter_err_update += 1
                    print(f"counter_err_update: {counter_err_update} for {phase} at {pt}: slope = {slope:.3f} \\pm {slope_error:.3f}, intercept = {intercept:.3f} \\pm {intercept_error:.3f}\n")
                    print(f"Values used: slope_TI = {slope_TI:.3f}, intercept_TI = {intercept_TI:.3f}, slope_error_TI = {slope_error_TI:.3f}, intercept_error_TI = {intercept_error_TI:.3f}\n")
                    # else:
                    #     slope_error, intercept_error = np.nan, np.nan


            else:
                intercept, slope = np.nan, np.nan
                slope_error, intercept_error = np.nan, np.nan
            # Update the df_isobar_superset with a and b
            mask = (df_isobar_superset["Target temperature (K)"] == temp)

            if temp in df_subset["Target temperature (K)"].values:
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


        # print df_isobar_superset
        # sort wrt Config_folder and then temperature
        df_isobar_superset.sort_values(["Config_folder", "Target temperature (K)"], inplace=True)
        # print(f"df_isobar_superset for {phase} at {pt}:\n{df_isobar_superset[['Config_folder', 'G_hp','G_hp_per_atom', 'G_hp_per_atom_w_TS', 'Target temperature (K)', 'Target pressure (GPa)', f'mu_{secondary_species}', f'mu_{secondary_species}_error']]}\n")
        # print(f"df_isobar_superset for {phase} at {pt}:\n{df_isobar_superset[['Config_folder', 'G_hp','G_hp_per_atom', 'G_hp_per_atom_w_TS', f'mu_excess_{secondary_species}', f'mu_excess_{secondary_species}_error']]}\n")


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

        # print("\n"*8)

        # exit(0)


    print(f"\nNOTE: Assuming fractional errors in slope and intercept for Gibbs-Helmholtz analysis are same as that for TI!\n\n")




    # narrow down to Config_folder with *_8H*
    df_superset = df_superset[df_superset["Config_folder"].str.contains("_8H")]


    # run if KD_sil_to_metal and KD_sil_to_metal_error exist in df_superset
    # if "KD_sil_to_metal" in df_superset.columns and "KD_sil_to_metal_error" in df_superset.columns:
        # calculate KD, D_wt
    for i, row in df_superset.iterrows():
        KD, KD_error, KD_lower, KD_upper = compute_KD(row, df_superset)
        # print(f"Computed KD_sil_to_metal for row {i}: {KD:.3f} ± {KD_error:.3f} for {row['Phase']} at {row['P_T_folder']}\n\n")
        df_superset.at[i, "KD_sil_to_metal"] = KD
        df_superset.at[i, "KD_sil_to_metal_error"] = KD_error
        df_superset.at[i, "KD_sil_to_metal_low"] = KD_lower
        df_superset.at[i, "KD_sil_to_metal_high"] = KD_upper
    df_superset["D_wt"] = df_superset["KD_sil_to_metal"] * (100/56)
    df_superset["D_wt_error"] = df_superset["KD_sil_to_metal_error"] * (100/56)
    df_superset["D_wt_low"] = df_superset["KD_sil_to_metal_low"] * (100/56)
    df_superset["D_wt_high"] = df_superset["KD_sil_to_metal_high"] * (100/56)




    # add df to df_superset
    df_superset = pd.concat([df_superset, df], ignore_index=True)

    # narrow down to Config_folder with *_8H*
    df_superset = df_superset[df_superset["Config_folder"].str.contains("_8H")]


    # sort df_superset by Phase, Config_folder and Target temperature (K)
    df_superset.sort_values(["Phase", "Config_folder", "Target temperature (K)"], inplace=True)









    # 11) Print summary of results
    # print(f"\nKD_sil_to_metal for {secondary_species} in Fe and MgSiO3 systems <df_superset>:")
    # # Loop over each unique (P_T_folder, Phase) combination
    # for (folder, phase), sub_df in df_superset.groupby(["P_T_folder", "Phase"]):
    #     # grab all non-NaN KD values
    #     kd_vals = sub_df["KD_sil_to_metal"].dropna()
    #     # loop through all unique temperatures in sub_df
    #     if not kd_vals.empty:
    #         for temp, temp_df in sub_df.groupby("Target temperature (K)"):
    #             kd0 = temp_df["KD_sil_to_metal"].iloc[0]
    #             mu0 = temp_df[f"mu_excess_{secondary_species}"].iloc[0]
    #             print(f"{folder}, Phase={phase}, T={temp} K:")
    #             print(f"                                KD_sil_to_metal = {kd0:.4f} ± {temp_df['KD_sil_to_metal_error'].iloc[0]:.4f} (mu_excess = {mu0:.4f} ± {temp_df[f'mu_excess_{secondary_species}_error'].iloc[0]:.4f})")
    #             # print(f"                                slope = {temp_df['slope'].iloc[0]:.4f} ± {temp_df['slope_error'].iloc[0]:.4f}, intercept = {temp_df['intercept'].iloc[0]:.4f} ± {temp_df['intercept_error'].iloc[0]:.4f}")
    #     else:
    #         print(f"{folder}, Phase={phase}: "
    #                 "No valid KD_sil_to_metal values found.")




    # create df_selection with P_T_folder, Phase, Target temperature (K), mu_excess, mu_excess_{secondary_species}_error, KD_sil_to_metal, KD_sil_to_metal_error, D_wt, D_wt_error
    df_selection = df_superset[["P_T_folder", "Phase", "Target temperature (K)", f"mu_excess_{secondary_species}", f"mu_excess_{secondary_species}_error", "KD_sil_to_metal", "KD_sil_to_metal_error", 'KD_sil_to_metal_low', 'KD_sil_to_metal_high', "Config_folder"]].copy()

    # narrow down to Config_folder with *_8H*
    df_selection = df_selection[df_selection["Config_folder"].str.contains("_8H")]

    # further narrow down to P50_T3500
    df_selection = df_selection[df_selection["P_T_folder"] == "P250_T6500"]

    print("\n\n df_selection:\n")
    # print(f"\n\n{df_selection}\n\n")
    # write
    # df_selection to CSV
    df_selection.to_csv("all_TI_results_selection.csv", index=False)
    print(f"Wrote all_TI_results_selection.csv with {len(df_selection)} rows.\n\n")



    # Save the assembled table to CSV
    pd.set_option("display.width", 200)
    df_superset.to_csv("all_TI_results_superset.csv", index=False)
    print(f"Wrote all_TI_results_superset.csv with {len(df_superset)} rows.")













    print("\n"*10)
    print("="*50)
    print(f"Unique temperatures in df_superset ({len(df_superset['Target temperature (K)'].unique())}): {df_superset['Target temperature (K)'].unique()}\n")
    print("="*50)
    print("\n"*10)


    # print size of df_superset
    print(f"Size of df_superset: {len(df_superset)} rows, {len(df_superset.columns)} columns.\n")
    df_superset_wo_nan = df_superset.dropna(subset=["G_hp_per_atom_w_TS", "mu_excess_" + secondary_species, "mu_excess_" + secondary_species + "_error", "KD_sil_to_metal", "KD_sil_to_metal_error", "D_wt", "D_wt_error"])
    print(f"Size of df_superset_wo_nan: {len(df_superset_wo_nan)} rows, {len(df_superset_wo_nan.columns)} columns.\n")

    # print all columns with nan
    print(f"Columns with NaN values in df_superset:\n{df_superset.isna().sum()[df_superset.isna().sum() > 0]}\n")

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


















if SCRIPT_MODE >= 0: # plot if > 0




    ############################################
    ############################################
    ############################################
    ###### PLOTS
    ############################################
    ############################################
    ############################################







    ############################################
    ############################################
    ############################################
    def plot_studies(
        fn_ax_KD,
        fn_ax_D_wt,
        datasets,
        x_variable,
        z_variable,
        cmap,
        norm,
        markers,
        marker_opts_scatter,
        marker_opts_error
    ):
        """
        Plot experimental study data on two axes: partition coefficient and distribution coefficient.

        Parameters
        ----------
        ax_KD : matplotlib.axes.Axes
            Axis for plotting KD_sil_to_metal.
        ax_D_wt : matplotlib.axes.Axes
            Axis for plotting D_wt.
        datasets : list of dict
            Each dict must contain keys:
            - x_variable
            - z_variable
            - 'KD_sil_to_metal', 'KD_sil_to_metal_low', 'KD_sil_to_metal_high'
            - 'D_wt', 'D_wt_low', 'D_wt_high'
            - 'label'
        markers : list of str
            Matplotlib marker styles for each dataset.
        x_variable : str
            Column name for x-axis values.
        z_variable : str
            Column name for temperature (used for coloring).
        cmap : Colormap
            Colormap instance for mapping temperature to color.
        norm : Normalize
            Normalize instance for colormap scaling.
        marker_opts_scatter : dict
            Keyword args for scatter (e.g., linestyle, s, alpha).
        marker_opts_error : dict
            Keyword args for errorbar (e.g., linewidth, capsize, alpha).
        """

        for data, marker in zip(datasets, markers):
            # Scatter KD_sil_to_metal
            # var  = df[z_variable].values
            x_vals = data[x_variable]
            temps  = np.array(data[z_variable], float)
            colors = cmap(norm(temps))

            fn_ax_KD.scatter(
                x_vals,
                data["KD_sil_to_metal"],
                color=colors,
                **marker_opts_scatter,
                marker=marker,
                label=data["label"]
            )

            # print value of alpha from marker_opts_scatter
            # print(f"Alpha: {alpha} for dataset {data['label']} with marker {marker}")

            # Error bars for KD_sil_to_metal
            for x0, y0, y_low, y_high, c in zip(
                x_vals,
                data["KD_sil_to_metal"],
                data["KD_sil_to_metal_low"],
                data["KD_sil_to_metal_high"],
                colors
            ):
                low = y0 - y_low
                high = y_high - y0
                yerr = [[low], [high]]  # shape (2,1)
                fn_ax_KD.errorbar(
                    x0, y0,
                    yerr=yerr,
                    fmt='none',
                    **marker_opts_error,
                    ecolor=c
                )


            # Scatter D_wt
            fn_ax_D_wt.scatter(
                x_vals,
                data["D_wt"],
                color=colors,
                **marker_opts_scatter,
                marker=marker,
                label=data["label"]
            )


            # Error bars for D_wt
            for x0, y0, y_low, y_high, c in zip(
                x_vals,
                data["D_wt"],
                data["D_wt_low"],
                data["D_wt_high"],
                colors
            ):
                low = y0 - y_low
                high = y_high - y0
                yerr = [[low], [high]]
                fn_ax_D_wt.errorbar(
                    x0, y0,
                    yerr=yerr,
                    fmt='none',
                    **marker_opts_error,
                    ecolor=c
                )

    # Usage example:
    # plot_studies(
    #     ax_KD,
    #     ax_D_wt,
    #     datasets,
    #     marker_other_expt_studies,
    #     x_variable,
    #     z_variable,
    #     cmap,
    #     norm,
    #     marker_opts_scatter__others,
    #     marker_opts_error__others
    # )
    ############################################
    ############################################
    ############################################









    # 2) Common settings
    from matplotlib.colors import LinearSegmentedColormap
    def pastel_cmap(cmap, factor=0.7, N=256):
        """
        Return a pastel version of `cmap` by blending each color toward white.
        
        Parameters
        ----------
        cmap : Colormap
            The original colormap (e.g. plt.get_cmap('hot')).
        factor : float, optional
            How “pastel” it is: 0 → original, 1 → pure white.  Default 0.7.
        N : int, optional
            Number of entries in the colormap.  Default 256.
        """
        # sample the original colormap
        colors = cmap(np.linspace(0, 1, N))
        # blend RGB channels toward 1.0 (white)
        colors[:, :3] = colors[:, :3] + (1.0 - colors[:, :3]) * factor
        # rebuild a new colormap
        return LinearSegmentedColormap.from_list(f'pastel_{cmap.name}', colors)




    ##############################################
    ##############################################
    ##############################################

    # previous computational studies
    if secondary_species == "He":
            
        """
        P in GPa, T in K, D_wt in wt% (weight percent)

        Yunguo Li et al. 2022 data 
        dict: data_He__Li_et_al_2022
        ignore -- # P:20, T:?, D_wt: 0.0007800604871350279, D_wt_low: 0.0003416077357591995, D_wt_high: 0.0017812663470193448
        P:50, T:3500, D_wt: 0.0025085161975393296, D_wt_low: 0.001076638661030702, D_wt_high: 0.0059636233165946545
        P:135, T:4200, D_wt: 0.0008455001765588335, D_wt_low: 0.00027929562959971827, D_wt_high: 0.0025085161975393296
        
        Zhang & Yin 2012 data
        dict: data_He__Zhang_and_Yin_2012
        P:40, T:3200, D_wt: 0.008921488454390353, D_wt_low: 0.0030070047268396694, D_wt_high: 0.015060610109031605

        Xiong et al. 2020 (corrected) data -- no error known
        dict: data_He__Xiong_et_al_2020_corrected
        P:20, T:5000, D_wt: 0.00003439086890726415, D_wt_low: 0.00003439086890726415, D_wt_high: 0.00003439086890726415
        P:60, T:5000, D_wt: 0.00161063482282789, D_wt_low: 0.00161063482282789, D_wt_high: 0.00161063482282789
        P:135, T:5000, D_wt: 0.0024094848405521382, D_wt_low: 0.0024094848405521382, D_wt_high: 0.0024094848405521382

        Wang et al. 2022 (corrected) data -- no error known
        dict: data_He__Wang_et_al_2022_corrected
        P:20, T:2500, D_wt: 0.00003047649470506126, D_wt_low: 0.00003047649470506126, D_wt_high: 0.00003047649470506126
        P:40, T:3200, D_wt: 0.0012905866607492368, D_wt_low: 0.0012905866607492368, D_wt_high: 0.0012905866607492368
        P:60, T:3600, D_wt: 0.0019306977288832535, D_wt_low: 0.0019306977288832535, D_wt_high: 0.0019306977288832535
        P:135, T:5000, D_wt: 0.17224697497149574, D_wt_low: 0.17224697497149574, D_wt_high: 0.17224697497149574

        Yuan & Steinle-Neumann 2020 data
        dict: data_He__Yuan_and_Steinle_Neumann_2020
        P:10, T:3000, D_wt: 10**(-4.73), D_wt_low: 10**(-4.73-0.40), D_wt_high: 10**(-4.73+0.40)
        P:25, T:3500, D_wt: 10**(-3.48), D_wt_low: 10**(-3.48-0.37), D_wt_high: 10**(-3.48+0.37) 
        P:40, T:3800, D_wt: 10**(-3.32), D_wt_low: 10**(-3.32-0.28), D_wt_high: 10**(-3.32+0.28)
        P:50, T:4000, D_wt: 10**(-2.07), D_wt_low: 10**(-2.07-0.29), D_wt_high: 10**(-2.07+0.29)
        P:80, T:4000, D_wt: 10**(-2.59), D_wt_low: 10**(-2.59-0.26), D_wt_high: 10**(-2.59+0.26)
        P:130, T:5000, D_wt: 10**(-1.24), D_wt_low: 10**(-1.24-0.20), D_wt_high: 10**(-1.24+0.20)

        """

        data_He__Li_et_al_2022 = {
            "Target pressure (GPa)": [50, 135],
            "Target temperature (K)": [3500, 4200],
            "D_wt": [0.0025085161975393296, 0.0008455001765588335],
            "D_wt_low": [0.001076638661030702, 0.00027929562959971827],
            "D_wt_high": [0.0059636233165946545, 0.0025085161975393296],
            "label": "Yunguo Li et al. 2022"
        }

        data_He__Zhang_and_Yin_2012 = {
            "Target pressure (GPa)": [40],
            "Target temperature (K)": [3200],
            "D_wt": [0.008921488454390353],
            "D_wt_low": [0.0030070047268396694],
            "D_wt_high": [0.015060610109031605],
            "label": "Zhang & Yin 2012"
        }

        data_He__Xiong_et_al_2020_corrected = {
            "Target pressure (GPa)": [20, 60, 135],
            "Target temperature (K)": [5000, 5000, 5000],
            "D_wt": [0.00003439086890726415, 0.00161063482282789, 0.0024094848405521382],
            "D_wt_low": [0.00003439086890726415, 0.00161063482282789, 0.0024094848405521382],
            "D_wt_high": [0.00003439086890726415, 0.00161063482282789, 0.0024094848405521382],
            "label": "Xiong et al. 2020 (corrected)"
        }

        data_He__Wang_et_al_2022_corrected = {
            "Target pressure (GPa)": [20, 40, 60, 135],
            "Target temperature (K)": [2500, 3200, 3600, 5000],
            "D_wt": [0.00003047649470506126, 0.0012905866607492368, 0.0019306977288832535, 0.17224697497149574],
            "D_wt_low": [0.00003047649470506126, 0.0012905866607492368, 0.0019306977288832535, 0.17224697497149574],
            "D_wt_high": [0.00003047649470506126, 0.0012905866607492368, 0.0019306977288832535, 0.17224697497149574],
            "label": "Wang et al. 2022 (corrected)"
        }

        data_He__Yuan_and_Steinle_Neumann_2020 = {
            "Target pressure (GPa)": [10, 25, 40, 50, 80, 130],
            "Target temperature (K)": [3000, 3500, 3800, 4000, 4000, 5000],
            "D_wt": [10**(-4.73), 10**(-3.48), 10**(-3.32), 10**(-2.07), 10**(-2.59), 10**(-1.24)],
            "D_wt_low": [10**(-4.73-0.40), 10**(-3.48-0.37), 10**(-3.32-0.28), 10**(-2.07-0.29), 10**(-2.59-0.26), 10**(-1.24-0.20)],
            "D_wt_high": [10**(-4.73+0.40), 10**(-3.48+0.37), 10**(-3.32+0.28), 10**(-2.07+0.29), 10**(-2.59+0.26), 10**(-1.24+0.20)],
            "label": "Yuan & Steinle-Neumann 2020"
        }

        datasets_comp = [
                    data_He__Li_et_al_2022,
                    data_He__Zhang_and_Yin_2012,
                    data_He__Xiong_et_al_2020_corrected,
                    data_He__Wang_et_al_2022_corrected,
                    data_He__Yuan_and_Steinle_Neumann_2020
                                                            ]

        # calculate KD_sil_to_metal from D_wt (including error) using the formula:
        # KD_sil_to_metal = D_wt * (56/100)
        for data in datasets_comp:
            data["KD_sil_to_metal"] = [d * (56/100) for d in data["D_wt"]]
            data["KD_sil_to_metal_low"] = [d * (56/100) for d in data["D_wt_low"]]
            data["KD_sil_to_metal_high"] = [d * (56/100) for d in data["D_wt_high"]]


    if secondary_species == "H":
        """
        P in GPa, T in K, D_wt in wt% (weight percent)

        Yunguo Li et al. 2022 data 
        dict: data_H__Li_et_al_2022
        P: 20, T: 2800, D_wt: 8.726904331596701, D_wt_low: 2.245197139773531, D_wt_high: 34.20189577525205
        P: 50, T: 3500, D_wt: 9.01980466823185, D_wt_low: 2.972475471559787, D_wt_high: 27.145119399821766
        P: 90, T: 3900, D_wt: 17.81952229659459, D_wt_low: 6.145121384918334, D_wt_high: 51.6727587608185
        P: 135, T: 4200, D_wt: 16.006673089593978, D_wt_low: 3.072240409180031, D_wt_high: 83.39633273214913

        Yuan & Steinle-Neumann 2020 data
        dict: data_H__Yuan_and_Steinle_Neumann_2020
        P: 20, T: 2500, KD: 10**(0.86), KD_low: 10**(0.86), KD_high: 10**(0.86)
        P: 40, T: 4000, KD: 10**(1.92), KD_low: 10**(1.92), KD_high: 10**(1.92)
        P: 130, T: 4000, KD: 10**(4.95), KD_low: 10**(4.95), KD_high: 10**(4.95)

        """

        data_H__Li_et_al_2022 = {
            "Target pressure (GPa)": [20, 50, 90, 135],
            "Target temperature (K)": [2800, 3500, 3900, 4200],
            "D_wt": [8.726904331596701, 9.01980466823185, 17.81952229659459, 16.006673089593978],
            "D_wt_low": [2.245197139773531, 2.972475471559787, 6.145121384918334, 3.072240409180031],
            "D_wt_high": [34.20189577525205, 27.145119399821766, 51.6727587608185, 83.39633273214913],
            "label": "Yunguo Li et al. 2022"
        }

        #calculate KD_sil_to_metal from D_wt (including error) using the formula:
        # KD_sil_to_metal = D_wt * (56/100)
        data_H__Li_et_al_2022["KD_sil_to_metal"] = [d * (56/100) for d in data_H__Li_et_al_2022["D_wt"]]
        data_H__Li_et_al_2022["KD_sil_to_metal_low"] = [d * (56/100) for d in data_H__Li_et_al_2022["D_wt_low"]]
        data_H__Li_et_al_2022["KD_sil_to_metal_high"] = [d * (56/100) for d in data_H__Li_et_al_2022["D_wt_high"]]


        data_H__Yuan_and_Steinle_Neumann_2020 = {
            "Target pressure (GPa)": [20, 40, 130],
            "Target temperature (K)": [2500, 4000, 4000],
            "KD_sil_to_metal": [10**(0.86), 10**(1.92), 10**(4.95)],
            "KD_sil_to_metal_low": [10**(0.86), 10**(1.92), 10**(4.95)],
            "KD_sil_to_metal_high": [10**(0.86), 10**(1.92), 10**(4.95)],
            "label": "Yuan & Steinle-Neumann 2020"
        }  

        # calculate D_wt from KD_sil_to_metal (including error) using the formula:
        # D_wt = KD_sil_to_metal * (100/56)
        data_H__Yuan_and_Steinle_Neumann_2020["D_wt"] = [kd * (100/56) for kd in data_H__Yuan_and_Steinle_Neumann_2020["KD_sil_to_metal"]]
        data_H__Yuan_and_Steinle_Neumann_2020["D_wt_low"] = [kd * (100/56) for kd in data_H__Yuan_and_Steinle_Neumann_2020["KD_sil_to_metal_low"]]
        data_H__Yuan_and_Steinle_Neumann_2020["D_wt_high"] = [kd * (100/56) for kd in data_H__Yuan_and_Steinle_Neumann_2020["KD_sil_to_metal_high"]]

        datasets_comp = [
                    data_H__Li_et_al_2022,
                    data_H__Yuan_and_Steinle_Neumann_2020
                                                            ]





    # previous experimental studies
    if secondary_species == "He":
            
        """
        P in GPa, T in K, D_wt in wt% (weight percent)

        Bouhifd et al. 2013 data
        dict: data_He__Bouhifd_et_al_2013__CI_FeNi
        P:1.9, T:2400, D_wt: 1.7 * 1e-3, D_wt_low: (1.7-0.8) * 1e-3, D_wt_high: (1.7+0.8) * 1e-3
        P: 4.1, T:2400, D_wt: 6.7 * 1e-4, D_wt_low: (6.7-4.1) * 1e-4, D_wt_high: (6.7+4.1) * 1e-4
        P: 6.1, T:2450, D_wt: 8.0 * 1e-4, D_wt_low: (8.0-4.6) * 1e-4, D_wt_high: (8.0+4.6) * 1e-4
        P: 8.0, T:2500, D_wt: 3.3 * 1e-3, D_wt_low: (3.3-1.6) * 1e-3, D_wt_high: (3.3+1.6) * 1e-3
        P: 13.1, T:2600, D_wt: 1.7 * 1e-2, D_wt_low: (1.7-0.7) * 1e-2, D_wt_high: (1.7+0.7) * 1e-2

        dict: data_He__Bouhifd_et_al_2013__CI_Fe
        P: 6.0, T:2200, D_wt: 6.4 * 1e-4, D_wt_low: (6.4-2.5) * 1e-4, D_wt_high: (6.4+2.5) * 1e-4
        P: 7.8, T:2200, D_wt: 1.7 * 1e-3, D_wt_low: (1.7-0.8) * 1e-3, D_wt_high: (1.7+0.8) * 1e-3
        P:10.0, T:2200, D_wt: 8.8 * 1e-3, D_wt_low: (8.8-4.8) * 1e-3, D_wt_high: (8.8+4.8) * 1e-3
        P:11.3, T:2300, D_wt: 1.0 * 1e-2, D_wt_low: (1.0-0.5) * 1e-2, D_wt_high: (1.0+0.5) * 1e-2
        P:13.8, T:2450, D_wt: 9.4 * 1e-3, D_wt_low: (9.4-4.8) * 1e-3, D_wt_high: (9.4+4.8) * 1e-3

        dict: data_He__Bouhifd_et_al_2013__CI_FeNiCo
        P: 7.5, T:2200, D_wt: 2.3 * 1e-3, D_wt_low: (2.3-0.7) * 1e-3, D_wt_high: (2.3+0.7) * 1e-3
        P:10.0, T:2250, D_wt: 5.3 * 1e-3, D_wt_low: (5.3-2.2) * 1e-3, D_wt_high: (5.3+2.2) * 1e-3
        P:13.5, T:2350, D_wt: 1.3 * 1e-2, D_wt_low: (1.3-0.5) * 1e-2, D_wt_high: (1.3+0.5) * 1e-2
        P:15.6, T:2450, D_wt: 6.0 * 1e-3, D_wt_low: (6.0-5.2) * 1e-3, D_wt_high: (6.0+5.2) * 1e-3
        P:15.8, T:2500, D_wt: 4.7 * 1e-3, D_wt_low: (4.7-4.1) * 1e-3, D_wt_high: (4.7+4.1) * 1e-3


        Matsuda et al. 1993 data
        dict: data_He__Matsuda_et_al_1993 -- no error known
        P: 0.5, T: 1600+273.15, D_wt: 0.04075005786153117, D_wt_low: 0.04075005786153117, D_wt_high: 0.04075005786153117
        P: 2, T: 1600+273.15, D_wt: 0.014470898697931086, D_wt_low: 0.014470898697931086, D_wt_high: 0.014470898697931086
        P: 6, T: 1600+273.15, D_wt: 0.0010752215989857346, D_wt_low: 0.0010752215989857346, D_wt_high: 0.0010752215989857346


    """

        data_He__Bouhifd_et_al_2013__CI_FeNi = {
            "Target pressure (GPa)": [1.9, 4.1, 6.1, 8.0, 13.1],
            "Target temperature (K)": [2400, 2400, 2450, 2500, 2600],
            "D_wt": [1.7 * 1e-3, 6.7 * 1e-4, 8.0 * 1e-4, 3.3 * 1e-3, 1.7 * 1e-2],
            "D_wt_low": [(1.7-0.8) * 1e-3, (6.7-4.1) * 1e-4, (8.0-4.6) * 1e-4, (3.3-1.6) * 1e-3, (1.7-0.7) * 1e-2],
            "D_wt_high": [(1.7+0.8) * 1e-3, (6.7+4.1) * 1e-4, (8.0+4.6) * 1e-4, (3.3+1.6) * 1e-3, (1.7+0.7) * 1e-2],
            "label": "Bouhifd et al. 2013 (CI-FeNi)"
        }

        data_He__Bouhifd_et_al_2013__CI_Fe = {
            "Target pressure (GPa)": [6.0, 7.8, 10.0, 11.3, 13.8],
            "Target temperature (K)": [2200, 2200, 2200, 2300, 2450],
            "D_wt": [6.4 * 1e-4, 1.7 * 1e-3, 8.8 * 1e-3, 1.0 * 1e-2, 9.4 * 1e-3],
            "D_wt_low": [(6.4-2.5) * 1e-4, (1.7-0.8) * 1e-3, (8.8-4.8) * 1e-3, (1.0-0.5) * 1e-2, (9.4-4.8) * 1e-3],
            "D_wt_high": [(6.4+2.5) * 1e-4, (1.7+0.8) * 1e-3, (8.8+4.8) * 1e-3, (1.0+0.5) * 1e-2, (9.4+4.8) * 1e-3],
            "label": "Bouhifd et al. 2013 (CI-Fe)"
        }

        data_He__Bouhifd_et_al_2013__CI_FeNiCo = {
            "Target pressure (GPa)": [7.5, 10.0, 13.5, 15.6, 15.8],
            "Target temperature (K)": [2200, 2250, 2350, 2450, 2500],
            "D_wt": [2.3 * 1e-3, 5.3 * 1e-3, 1.3 * 1e-2, 6.0 * 1e-3, 4.7 * 1e-3],
            "D_wt_low": [(2.3-0.7) * 1e-3, (5.3-2.2) * 1e-3, (1.3-0.5) * 1e-2, (6.0-5.2) * 1e-3, (4.7-4.1) * 1e-3],
            "D_wt_high": [(2.3+0.7) * 1e-3, (5.3+2.2) * 1e-3, (1.3+0.5) * 1e-2, (6.0+5.2) * 1e-3, (4.7+4.1) * 1e-3],
            "label": "Bouhifd et al. 2013 (CI-FeNiCo)"
        }

        data_He__Matsuda_et_al_1993 = {
            "Target pressure (GPa)": [0.5, 2, 6],
            "Target temperature (K)": [1600+273.15, 1600+273.15, 1600+273.15],
            "D_wt": [0.04075005786153117, 0.014470898697931086, 0.0010752215989857346],
            "D_wt_low": [0.04075005786153117, 0.014470898697931086, 0.0010752215989857346],
            "D_wt_high": [0.04075005786153117, 0.014470898697931086, 0.0010752215989857346],
            "label": "Matsuda et al. 1993 (Basalt-Fe)"
        }

        datasets_expt = [
                    data_He__Bouhifd_et_al_2013__CI_FeNi,
                    data_He__Bouhifd_et_al_2013__CI_Fe,
                    data_He__Bouhifd_et_al_2013__CI_FeNiCo,
                    data_He__Matsuda_et_al_1993
                                                            ]

        # calculate KD_sil_to_metal from D_wt (including error) using the formula:
        # KD_sil_to_metal = D_wt * (56/100)
        for data in datasets_expt:
            data["KD_sil_to_metal"] = [d * (56/100) for d in data["D_wt"]]
            data["KD_sil_to_metal_low"] = [d * (56/100) for d in data["D_wt_low"]]
            data["KD_sil_to_metal_high"] = [d * (56/100) for d in data["D_wt_high"]]    


    elif secondary_species == "H":
        """
        P in GPa, T in K, D_wt in wt% (weight percent), KD in dimensionless

        Tagawa et al. 2021 data (HO0.5_{silicate} + 0.5 Fe_{metal} = H_{metal} + 0.5FeO_{silicate})
        dict: data_H__Tagawa_et_al_2021
        P: 46, T: 3920, KD: 10**1.32, KD_low: 10**(1.32-0.06), KD_high: 10**(1.32+0.06)
        P: 48, T: 3450, KD: 10**1.10, KD_low: 10**(1.10-0.06), KD_high: 10**(1.10+0.06)
        P: 57, T: 3860, KD: 10**1.22, KD_low: 10**(1.22-0.05), KD_high: 10**(1.22+0.05)
        P: 60, T: 4560, KD: 10**1.37, KD_low: 10**(1.37-0.05), KD_high: 10**(1.37+0.05)
        P: 47, T: 4230, KD: 10**1.35, KD_low: 10**(1.35-0.06), KD_high: 10**(1.35+0.06)
        P: 30, T: 3080, KD: 10**1.18, KD_low: 10**(1.18-0.06), KD_high: 10**(1.18+0.06)

        Clesi et al. 2018 data (0.5 H2_{silicate} = H_{metal})
        dict: data_H__Clesi_et_al_2018
        P: 5, T: 2245, D_wt: 0.048, D_wt_low: 0.048-0.032, D_wt_high: 0.048+0.032
        P: 5, T: 2125, D_wt: 0.166, D_wt_low: 0.166-0.081, D_wt_high: 0.166+0.081
        P: 5, T: 2020, D_wt: 0.181, D_wt_low: 0.181-0.091, D_wt_high: 0.181+0.091
        P: 10, T: 2375, D_wt: 0.167, D_wt_low: 0.167-0.077, D_wt_high: 0.167+0.077
        P: 10, T: 2125, D_wt: 0.051, D_wt_low: 0.051-0.017, D_wt_high: 0.051+0.017
        P: 21, T: 2700, D_wt: 0.34, D_wt_low: 0.34-0.15, D_wt_high: 0.34+0.15
        P: 5, T: 2240, D_wt: 0.041, D_wt_low: 0.041-0.027, D_wt_high: 0.041+0.027
        P: 20, T: 2775, D_wt: 0.77, D_wt_low: 0.77-0.32, D_wt_high: 0.77+0.32
        P: 10, T: 2325, D_wt: 0.044, D_wt_low: 0.044-0.014, D_wt_high: 0.044+0.014

        Malavergne et al. 2018 data 
        dict: data_H__Malavergne_et_al_2018
        P -- 10,10,6,1,1,1,1
        T -- 1873.15,1873.15,2073.15,1873.15,1873.15,1873.15,1673.15
        D_wt -- 0.50 (10), 0.33 (7), 0.24 (6), 0.22 (6), 0.047 (12), 0.038 (10), 0.045 (12)
        P: 10, T: 1873.15, D_wt: 0.50, D_wt_low: 0.50-0.10, D_wt_high: 0.50+0.10
        P: 10, T: 1873.15, D_wt: 0.33, D_wt_low: 0.33-0.07, D_wt_high: 0.33+0.07
        P: 6, T: 2073.15, D_wt: 0.24, D_wt_low: 0.24-0.06, D_wt_high: 0.24+0.06
        P: 1, T: 1873.15, D_wt: 0.22, D_wt_low: 0.22-0.06, D_wt_high: 0.22+0.06
        P: 1, T: 1873.15, D_wt: 0.047, D_wt_low: 0.047-0.012, D_wt_high: 0.047+0.012
        P: 1, T: 1873.15, D_wt: 0.038, D_wt_low: 0.038-0.010, D_wt_high: 0.038+0.010
        P: 1, T: 1673.15, D_wt: 0.045, D_wt_low: 0.045-0.012, D_wt_high: 0.045+0.012

        Okuchi 1997 data
        dict: data_H__Okuchi_1997
        P: 7.5, T: 1200+273.15, KD: exp**(-3.6), KD_low: exp**(-3.6-0.1), KD_high: exp**(-3.6+0.1)    
        P: 7.5, T: 1200+273.15, KD: exp**(-3.4), KD_low: exp**(-3.4-0.0), KD_high: exp**(-3.4+0.0)
        P: 7.5, T: 1300+273.15, KD: exp**(-2.4), KD_low: exp**(-2.4-0.6), KD_high: exp**(-2.4+0.6)
        P: 7.5, T: 1400+273.15, KD: exp**(-2.0), KD_low: exp**(-2.0-0.0), KD_high: exp**(-2.0+0.0)
        P: 7.5, T: 1500+273.15, KD: exp**(-1.4), KD_low: exp**(-1.4-0.2), KD_high: exp**(-1.4+0.2)
        P: 7.5, T: 1500+273.15, KD: exp**(1.5), KD_low: exp**(1.5-0.3), KD_high: exp**(1.5+0.3)
        P: 7.5, T: 1500+273.15, KD: exp**(-1.5), KD_low: exp**(-1.5-0.2), KD_high: exp**(-1.5+0.2)


        """

        data_H__Tagawa_et_al_2021 = {
            "Target pressure (GPa)": [46, 48, 57, 60, 47, 30],
            "Target temperature (K)": [3920, 3450, 3860, 4560, 4230, 3080],
            "KD_sil_to_metal": [10**1.32, 10**1.10, 10**1.22, 10**1.37, 10**1.35, 10**1.18],
            "KD_sil_to_metal_low": [10**(1.32-0.06), 10**(1.10-0.06), 10**(1.22-0.05), 10**(1.37-0.05), 10**(1.35-0.06), 10**(1.18-0.06)],
            "KD_sil_to_metal_high": [10**(1.32+0.06), 10**(1.10+0.06), 10**(1.22+0.05), 10**(1.37+0.05), 10**(1.35+0.06), 10**(1.18+0.06)],
            "label": "Tagawa et al. 2021"
        }
        # calculate D_wt from KD_sil_to_metal (including error) using the formula:
        # D_wt = KD_sil_to_metal * (100/56)
        data_H__Tagawa_et_al_2021["D_wt"] = [kd * (100/56) for kd in data_H__Tagawa_et_al_2021["KD_sil_to_metal"]]
        data_H__Tagawa_et_al_2021["D_wt_low"] = [kd * (100/56) for kd in data_H__Tagawa_et_al_2021["KD_sil_to_metal_low"]]
        data_H__Tagawa_et_al_2021["D_wt_high"] = [kd * (100/56) for kd in data_H__Tagawa_et_al_2021["KD_sil_to_metal_high"]]

        data_H__Clesi_et_al_2018 = {
            "Target pressure (GPa)": [5, 5, 5, 10, 10, 21, 5, 20, 10],
            "Target temperature (K)": [2245, 2125, 2020, 2375, 2125, 2700, 2240, 2775, 2325],
            "D_wt": [0.048, 0.166, 0.181, 0.167, 0.051, 0.34, 0.041, 0.77, 0.044],
            "D_wt_low": [0.048-0.032, 0.166-0.081, 0.181-0.091, 0.167-0.077, 0.051-0.017, 0.34-0.15, 0.041-0.027, 0.77-0.32, 0.044-0.014],
            "D_wt_high": [0.048+0.032, 0.166+0.081, 0.181+0.091, 0.167+0.077, 0.051+0.017, 0.34+0.15, 0.041+0.027, 0.77+0.32, 0.044+0.014],
            "label": "Clesi et al. 2018"
        }
        # calculate KD_sil_to_metal from D_wt (including error) using the formula:
        # KD_sil_to_metal = D_wt * (56/100)
        data_H__Clesi_et_al_2018["KD_sil_to_metal"] = [d * (56/100) for d in data_H__Clesi_et_al_2018["D_wt"]]
        data_H__Clesi_et_al_2018["KD_sil_to_metal_low"] = [d * (56/100) for d in data_H__Clesi_et_al_2018["D_wt_low"]]
        data_H__Clesi_et_al_2018["KD_sil_to_metal_high"] = [d * (56/100) for d in data_H__Clesi_et_al_2018["D_wt_high"]]

        data_H__Malavergne_et_al_2018 = {
            "Target pressure (GPa)": [10, 10, 6, 1, 1, 1, 1],
            "Target temperature (K)": [1873.15, 1873.15, 2073.15, 1873.15, 1873.15, 1873.15, 1673.15],
            "D_wt": [0.50, 0.33, 0.24, 0.22, 0.047, 0.038, 0.045],
            "D_wt_low": [0.50-0.10, 0.33-0.07, 0.24-0.06, 0.22-0.06, 0.047-0.012, 0.038-0.010, 0.045-0.012],
            "D_wt_high": [0.50+0.10, 0.33+0.07, 0.24+0.06, 0.22+0.06, 0.047+0.012, 0.038+0.010, 0.045+0.012],
            "label": "Malavergne et al. 2018"
        }
        # calculate KD_sil_to_metal from D_wt (including error) using the formula:
        # KD_sil_to_metal = D_wt * (56/100)
        data_H__Malavergne_et_al_2018["KD_sil_to_metal"] = [d * (56/100) for d in data_H__Malavergne_et_al_2018["D_wt"]]
        data_H__Malavergne_et_al_2018["KD_sil_to_metal_low"] = [d * (56/100) for d in data_H__Malavergne_et_al_2018["D_wt_low"]]
        data_H__Malavergne_et_al_2018["KD_sil_to_metal_high"] = [d * (56/100) for d in data_H__Malavergne_et_al_2018["D_wt_high"]]

        data_H__Okuchi_1997 = {
            "Target pressure (GPa)": [1, 1, 1, 1, 1],
            "Target temperature (K)": [1873.15, 1873.15, 1873.15, 1873.15, 1873.15],
            "D_wt": [0.045, 0.038, 0.045, 0.045, 0.045],
            "D_wt_low": [0.045-0.012, 0.038-0.010, 0.045-0.012, 0.045-0.012, 0.045-0.012],
            "D_wt_high": [0.045+0.012, 0.038+0.010, 0.045+0.012, 0.045+0.012, 0.045+0.012],
            "label": "Okuchi et al. 1997"
        }
        # calculate KD_sil_to_metal from D_wt (including error) using the formula:
        # KD_sil_to_metal = D_wt * (56/100)
        data_H__Okuchi_1997["KD_sil_to_metal"] = [d * (56/100) for d in data_H__Okuchi_1997["D_wt"]]
        data_H__Okuchi_1997["KD_sil_to_metal_low"] = [d * (56/100) for d in data_H__Okuchi_1997["D_wt_low"]]
        data_H__Okuchi_1997["KD_sil_to_metal_high"] = [d * (56/100) for d in data_H__Okuchi_1997["D_wt_high"]]

        datasets_expt = [
                    data_H__Tagawa_et_al_2021,
                    data_H__Clesi_et_al_2018,
                    data_H__Malavergne_et_al_2018,
                    data_H__Okuchi_1997
                                                            ]
        print(f"WARNING: Some studies talk about H2, silicate to metal partitioning, and some H, silicate --- we do H also here, how to make sure nothing is getting messed up?")





    # make this pandas DataFrame
    # datasets_comp = [pd.DataFrame(data) for data in datasets_comp]
    # datasets_expt = [pd.DataFrame(data) for data in datasets_expt]


    ##############################################
    ##############################################
    ##############################################






    # plot X_{secondary_species} vs G_hp_per_atom_w_TS, and color by P_T_folder and size by phase
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # 1) Load your assembled TI results (with columns: Phase, P_T_folder, X_{secondary_species}, G_hp_per_atom_w_TS, a, b, etc.)
    df = pd.read_csv("all_TI_results.csv")




    if PLOT_MODE == 1:

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






    if PLOT_MODE == 2:

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


        # 4) Final styling
        ax.set_xlabel(f"X_{secondary_species}")
        ax.set_ylabel(f"mu_{secondary_species} (eV)")
        ax.set_title(f"X_{secondary_species} vs mu_{secondary_species} with Line Fits by Phase+P_T_folder")
        ax.legend()
        # ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small", title="Legend")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"X_{secondary_species}_vs_mu_{secondary_species}.png")



























    ######################################################################
    ######################################################################
    ######################################################################
    df_superset = pd.read_csv("all_TI_results_superset.csv")
    # narrow down to Config_folder with *_8H*
    df_superset = df_superset[df_superset["Config_folder"].str.contains("_8H")]

    df = df_superset.copy()


    # x lim ,  y lim
    if secondary_species == "He":
        # y_min = 1e-5
        # y_max = 1e1
        y_min = 1e-5
        y_max = 1e3
    elif secondary_species == "H":
        # y_min = 1e-3
        # y_max = 1e3
        y_min = 1e-5
        y_max = 1e3
        print("NOTE: 130 GPa Yuan & Steinle-Neumann 2020 data is not included in the plot as it is way off the scale.")



    
    marker_TI="o"  # marker for TI points
    marker_2phase="s"  # marker for two-phase points
    marker_other_comp_studies=["^", "D", "v", "<", ">"] # array of markers for other studies
    marker_other_expt_studies=["p", "x", "h", "*", "H"]  # array of markers for other experimental studies

    marker_opts = dict(marker='o', linestyle='', markersize=10, alpha=1)#,color=base_color)
    marker_opts_scatter = dict(linestyle='', s=200, alpha=1)#,edgecolor='black',
    marker_opts_error = dict(linestyle='', markersize=10, alpha=1, capsize=3, elinewidth=1)#, color='black',ecolor='black')
    marker_opts_scatter__others = dict(linestyle='', s=100, alpha=0.5)#,color=base_color)
    marker_opts_error__others = dict(linestyle='', markersize=10, capsize=3, elinewidth=1,alpha=0.5)#, color='black',ecolor='black')



    if PLOT_MODE == 3 or PLOT_MODE < 0:
        fig, axes_KD_D_wt__P = plt.subplots(2, 1, figsize=(12, 10))#, sharex=True, sharey=True)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        # ax1, ax_KD, ax3, ax_D_wt = axes.flatten()
        ax_KD, ax_D_wt = axes_KD_D_wt__P.flatten()


        x_variable = "Target pressure (GPa)"  # x-axis variable for all plots
        z_variable = "Target temperature (K)"  # z-axis variable for all plots -- color coding



        # create a colormap based on the temperature range
        temp_min = 1000  # minimum temperature in K
        temp_max = 17000  # maximum temperature in K
        cbar_ticks = [3000, 6000, 9000, 12000, 15000]
        norm = plt.Normalize(
            vmin=temp_min,
            vmax=temp_max
        )

        magma = plt.get_cmap("magma")
        pastel_magma = pastel_cmap(magma, factor=0.25)  # tweak factor between 0 and 1
        cmap = pastel_magma  # use pastel magma for the plots




        # --- Panel 3: KD_sil_to_metal (log y) ---
        # ax_KD.plot(df[x_variable], df["KD_sil_to_metal"], **marker_opts)
        # ax_KD.errorbar(df[x_variable], df["KD_sil_to_metal"], yerr=df["KD_sil_to_metal_error"], **marker_opts_error, color=cmap(norm(df[z_variable])))


        # 1) Plot the colored points
        sc = ax_KD.scatter(
            df[x_variable],
            df["KD_sil_to_metal"],
            c=df[z_variable],
            cmap=cmap,
            norm=norm,
            **marker_opts_scatter,
            marker=marker_TI,  # use the TI marker for scatter points
            label="This study (TI)"
        )

        # 2) Draw per-point errorbars with matching colors
        temps  = df[z_variable].values
        colors = cmap(norm(temps))

        for x0, y0, y_low, y_high, c in zip(
            df[x_variable],
            df["KD_sil_to_metal"],
            df["KD_sil_to_metal_low"],
            df["KD_sil_to_metal_high"],
            colors
        ):

            low  = y0 - y_low
            high = y_high - y0

            # shape (2,1) array: [[low], [high]]
            yerr = [[low], [high]]

            ax_KD.errorbar(
                x0, y0,
                yerr=yerr,
                fmt='none',       # no extra marker
                ecolor=c,         # single RGBA tuple
                **marker_opts_error,
                marker=marker_TI  # use the TI marker for errorbars
            )






        # --- Panel 4: D_wt (log y) ---

        # 1) Plot the colored points
        sc = ax_D_wt.scatter(
            df[x_variable],
            df["D_wt"],
            c=df[z_variable],
            cmap=cmap,
            norm=norm,
            **marker_opts_scatter,
            marker=marker_TI,  # use the TI marker for scatter points
            label="This study (TI)"
        )
        # 2) Draw per-point errorbars with matching colors
        for x0, y0, err0, c in zip(
            df[x_variable],
            df["D_wt"],
            df["D_wt_error"],
            colors
        ):
            ax_D_wt.errorbar(
                x0, y0,
                yerr=err0,
                fmt='none',       # no extra marker
                ecolor=c,         # single RGBA tuple
                **marker_opts_error,
                marker=marker_TI  # use the TI marker for errorbars
            )














        # if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
        if secondary_species == "He":
            """ 
            
            Two-phase simulations for He in Fe and MgSiO3:
            dict: data_He__two_phase_simulations
            P:500, T:9000, KD: 0.032, KD_low: 0.032, KD_high: 0.032
            P:1000, T:13000, KD: 0.32, KD_low: 0.32, KD_high: 0.32

            """

            data_He__two_phase_simulations = {
                "Target pressure (GPa)": [500, 1000],
                "Target temperature (K)": [9000, 13000],
                "KD": [0.032, 0.32],
                "KD_low": [0.032, 0.32],
                "KD_high": [0.032, 0.32]
            }
            # D_wt = KD * (100/56)  # assuming Fe as metal, 56 g/mol
            data_He__two_phase_simulations["D_wt"] = [kd * (100/56) for kd in data_He__two_phase_simulations["KD"]]
            data_He__two_phase_simulations["D_wt_low"] = [kd * (100/56) for kd in data_He__two_phase_simulations["KD_low"]]
            data_He__two_phase_simulations["D_wt_high"] = [kd * (100/56) for kd in data_He__two_phase_simulations["KD_high"]]

            ax_KD.scatter(data_He__two_phase_simulations[x_variable], 
                            data_He__two_phase_simulations["KD"],
                            **marker_opts_scatter,
                            marker=marker_2phase,
                            c=data_He__two_phase_simulations[z_variable],
                            cmap=cmap,
                            norm=norm,
                            label="This study (2P)")
            ax_D_wt.scatter(data_He__two_phase_simulations[x_variable], 
                            data_He__two_phase_simulations["D_wt"],
                            **marker_opts_scatter,
                            marker=marker_2phase,
                            c=data_He__two_phase_simulations[z_variable],
                            cmap=cmap,
                            norm=norm,
                            label="This study (2P)")











        # plot the data points
        plot_studies(
        ax_KD,
        ax_D_wt,
        datasets_comp,
        x_variable,
        z_variable,
        cmap,
        norm,
        marker_other_comp_studies,
        marker_opts_scatter__others,
        marker_opts_error__others
        )









        # plot the data points
        plot_studies(
        ax_KD,
        ax_D_wt,
        datasets_expt,
        x_variable,
        z_variable,
        cmap,
        norm,
        marker_other_expt_studies,
        marker_opts_scatter__others,
        marker_opts_error__others
        )
        ###########################





        # show 1 colorbar for both KD and D_wt plots
        # 1) Create a colorbar for the temperature range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # only needed for older matplotlib versions
        # 2) Add the colorbar to the figure -- width == width of 1 plot
        cbar = fig.colorbar(sm, ax=[ax_KD, ax_D_wt],
                            orientation='horizontal',
                            # fraction=1, pad=0.04,
                            pad=0.1,  # space between colorbar and plot
                            ticks=np.linspace(temp_min, temp_max, 5),
                            location='bottom',  # 'top' or 'bottom'
                            shrink=1,      # shrink to 80% of the original size
                            aspect=50,     # thinner bar
                            )
        cbar.set_label("Temperature (K)")#, rotation=270, labelpad=15)
        # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
        cbar.set_ticks(cbar_ticks)
        # cbar.set_ticklabels([f"{int(t)} K" for t in cbar.get_ticks()])





        # # x lim ,  y lim
        # if secondary_species == "He":
        #     y_min = 1e-5 # 1e-5
        #     y_max = 1e1 #1e1
        # elif secondary_species == "H":
        #     y_min = 1e-3 #1e-3
        #     y_max = 1e6
        ax_KD.set_ylim(y_min, y_max)
        ax_D_wt.set_ylim(y_min, y_max)




        ax_KD.set_yscale("log")
        # ax_KD.set_ylabel(r"K$_D^{rxn: silicate → metal}$")
        # ax_KD.set_ylabel(r"K$_D$")
        ax_KD.set_ylabel(r"Equilibrium Constant ($K_{D}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
        ax_KD.grid(True)
        # ax_KD.tick_params(labelbottom=False)
        # ax_KD.set_xlabel("Pressure (GPa)")
        ax_KD.set_xscale("log")
        # ax_KD.set_xlim(left=0,right=200)  # set x-axis limits for better visibility

        ax_D_wt.set_yscale("log")
        # ax_D_wt.set_ylabel(r"D$_{wt}^{rxn: silicate → metal}$")
        ax_D_wt.set_ylabel(r"Partition Coefficient ($D_{wt}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
        ax_D_wt.grid(True)
        ax_D_wt.set_xlabel("Pressure (GPa)")
        # ax_D_wt.set_xscale("log")
        # ax_D_wt.set_xlim(left=10, right=1100)  # set x-axis limits for better visibility



        # Legend
        # 1) Grab handles & labels from one of your axes
        handles, labels = ax_KD.get_legend_handles_labels()

        # 2) Create a single legend on the right side of the figure
        fig.legend(
            handles,
            labels,
            loc='lower center',        # center vertically on the right edge
            fontsize=8,
            # borderaxespad=0.1,
            # bbox_to_anchor=(1.00, 0.5),
            frameon=False,
            ncol=3,  # number of columns in the legend
            # mode='expand',  # 'expand' to fill the space, 'none'
            labelspacing=1.5,    # default is 0.5; larger → more space
            handletextpad=1.0,   # default is 0.8; larger → more space between handle & text
            columnspacing=2.0,   # if you have multiple columns, space between them
        )

        # 3) Shrink the subplots to make room for the legend
        # fig.subplots_adjust(right=0.82)  # push the plot area leftward





        # fig.subplots_adjust(top=0.88)

        fig.suptitle(
            f"Equilibrium Constant ($K_D$) and Partition Coefficient ($D_{{wt}}$) "
            f"for the reaction {secondary_species}$_{{silicates}}$ $\\rightleftharpoons$ {secondary_species}$_{{metal}}$.\n"
            f"Note: Assumption that X$_{{{secondary_species},silicates}}$ $\\ll$ 1",
            fontsize=10#,
            # y=1.03,            # default ≃0.98, smaller → more gap
        )


        # 3) Layout & save
        # plt.tight_layout(rect=[0, 0, 1, 1.0])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust layout to make room for the title
        plt.savefig(f"KD_D_wt_vs_P_T.png", dpi=300)























    if PLOT_MODE == 4 or PLOT_MODE < 0:

        ########################################################
        ########################################################
        ########################################################
        ########################################################
        ## KD, D_wt vs Temperature for the current study (TI)


        # plt.rcParams.update(INITIAL_RCP)
        # plt.rcdefaults()

        fig, axes_KD_D_wt__T = plt.subplots(2, 1, figsize=(12, 10))#, sharex=True, sharey=True)
        ax_KD__T, ax_D_wt__T = axes_KD_D_wt__T.flatten()



        # add log pressure to df
        # df["log(Target pressure (GPa))"] = np.log10(df["Target pressure (GPa)"])
        # for data in datasets_comp:
        #     data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]
        # for data in datasets_expt:
        #     data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]

        x_variable = "Target temperature (K)"  # x-axis variable for all plots
        # z_variable = "log(Target pressure (GPa))"  # z-axis variable for all plots -- color coding
        z_variable = "Target pressure (GPa)"  # z-axis variable for all plots -- color coding



        # create a colormap based on the temperature range
        if z_variable == "log(Target pressure (GPa))":
            log10_pres_min = np.log10(0.3)  # minimum pressure in GPa
            log10_pres_max = np.log10(3000)  # maximum pressure in GPa
            cbar_ticks = np.log10([1, 10, 100, 1000])
            norm = plt.Normalize(
                vmin=log10_pres_min,
                vmax=log10_pres_max
            )
        elif z_variable == "Target pressure (GPa)":
            pres_min = 0.5  # minimum pressure in GPa
            pres_max = 1000  # maximum pressure in GPa
            cbar_ticks = [0, 200, 400, 600, 800, 1000]
            norm = plt.Normalize(
                vmin=pres_min,
                vmax=pres_max
            )


        viridis = plt.get_cmap("viridis")
        pastel_viridis = pastel_cmap(viridis, factor=0.25)  # tweak factor between 0 and 1
        cmap = pastel_viridis  # use pastel viridis for the plots






        # --- Panel 3: KD_sil_to_metal (log y) ---
        # ax_KD.plot(df[x_variable], df["KD_sil_to_metal"], **marker_opts)
        # ax_KD.errorbar(df[x_variable], df["KD_sil_to_metal"], yerr=df["KD_sil_to_metal_error"], **marker_opts_error, color=cmap(norm(df[z_variable])))


        # 1) Plot the colored points
        sc = ax_KD__T.scatter(
            df[x_variable],
            df["KD_sil_to_metal"],
            c=df[z_variable],
            cmap=cmap,
            norm=norm,
            **marker_opts_scatter,
            marker=marker_TI,  # use the TI marker for scatter points
            label="This study (TI)"
        )

        # 2) Draw per-point errorbars with matching colors
        temps  = df[z_variable].values
        colors = cmap(norm(temps))

        for x0, y0, y_low, y_high, c in zip(
            df[x_variable],
            df["KD_sil_to_metal"],
            df["KD_sil_to_metal_low"],
            df["KD_sil_to_metal_high"],
            colors
        ):

            low  = y0 - y_low
            high = y_high - y0

            # shape (2,1) array: [[low], [high]]
            yerr = [[low], [high]]

            ax_KD__T.errorbar(
                x0, y0,
                yerr=yerr,
                fmt='none',       # no extra marker
                ecolor=c,         # single RGBA tuple
                **marker_opts_error,
                marker=marker_TI  # use the TI marker for errorbars
            )


        # --- Panel 4: D_wt (log y) ---

        # 1) Plot the colored points
        sc = ax_D_wt__T.scatter(
            df[x_variable],
            df["D_wt"],
            c=df[z_variable],
            cmap=cmap,
            norm=norm,
            **marker_opts_scatter,
            marker=marker_TI,  # use the TI marker for scatter points
            label="This study (TI)"
        )
        # 2) Draw per-point errorbars with matching colors
        for x0, y0, err0, c in zip(
            df[x_variable],
            df["D_wt"],
            df["D_wt_error"],
            colors
        ):
            ax_D_wt__T.errorbar(
                x0, y0,
                yerr=err0,
                fmt='none',       # no extra marker
                ecolor=c,         # single RGBA tuple
                **marker_opts_error,
                marker=marker_TI  # use the TI marker for errorbars
            )





        # plot the data points -- computational studies
        plot_studies(
        ax_KD__T,
        ax_D_wt__T,
        datasets_comp,
        x_variable,
        z_variable,
        cmap,
        norm,
        marker_other_comp_studies,
        marker_opts_scatter__others,
        marker_opts_error__others
        )

        # plot the data points -- experimental studies
        plot_studies(
        ax_KD__T,
        ax_D_wt__T,
        datasets_expt,
        x_variable,
        z_variable,
        cmap,
        norm,
        marker_other_expt_studies,
        marker_opts_scatter__others,
        marker_opts_error__others
        )


        # plt.savefig(f"KD_D_wt_vs_P_T__T.png", dpi=300)
        # exit(0)  # exit here to avoid plotting the second figure


        # show 1 colorbar for both KD and D_wt plots
        # 1) Create a colorbar for the temperature range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # only needed for older matplotlib versions
        # 2) Add the colorbar to the figure -- width == width of 1 plot
        if z_variable == "log(Target pressure (GPa))":
            cbar = fig.colorbar(sm, ax=[ax_KD__T, ax_D_wt__T],
                                orientation='horizontal',
                                # fraction=1, pad=0.04,
                                pad=0.1,  # space between colorbar and plot
                                # ticks=np.log10([1, 10, 100, 1000]),
                                location='bottom',  # 'top' or 'bottom'
                                shrink=1,      # shrink to 80% of the original size
                                aspect=50,     # thinner bar
                                )
            cbar.set_label("Pressure (GPa)")#, rotation=270, labelpad=15)
            # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(["1", "10", "100", "1000"])  # set tick labels to be in GPa

            ## grab colors used for P=50,250,500,1000 GPa
            colors__wrt_P = [cmap(norm(np.log10(p))) for p in [50, 250, 500, 1000]]

        elif z_variable == "Target pressure (GPa)":
            cbar = fig.colorbar(sm, ax=[ax_KD__T, ax_D_wt__T],
                                orientation='horizontal',
                                # fraction=1, pad=0.04,
                                pad=0.1,  # space between colorbar and plot
                                # ticks=np.log10([1, 10, 100, 1000]),
                                location='bottom',  # 'top' or 'bottom'
                                shrink=1,      # shrink to 80% of the original size
                                aspect=50,     # thinner bar
                                )
            cbar.set_label("Pressure (GPa)")#, rotation=270, labelpad=15)
            # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(["0", "200", "400", "600", "800", "1000"])  # set tick labels to be in GPa

            ## grab colors used for P=50,250,500,1000 GPa
            colors__wrt_P = [cmap(norm(p)) for p in [50, 250, 500, 1000]]





        # x lim ,  y lim
        # if secondary_species == "He":
        #     y_min__T = 1e-5 # 1e-5
        #     y_max__T = 1e3 #1e1
        # elif secondary_species == "H":
        #     y_min__T = 1e-5 #1e-3
        #     y_max__T = 1e3
        ax_KD__T.set_ylim(y_min, y_max)
        ax_D_wt__T.set_ylim(y_min, y_max)




        ax_KD__T.set_yscale("log")
        # ax_KD.set_ylabel(r"K$_D^{rxn: silicate → metal}$")
        # ax_KD.set_ylabel(r"K$_D$")
        ax_KD__T.set_ylabel(r"Equilibrium Constant ($K_{D}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
        ax_KD__T.grid(True)
        # ax_KD.tick_params(labelbottom=False)
        # ax_KD.set_xlabel("Pressure (GPa)")
        ax_KD__T.set_xscale("log")
        # ax_KD.set_xlim(left=0,right=200)  # set x-axis limits for better visibility

        ax_D_wt__T.set_yscale("log")
        # ax_D_wt.set_ylabel(r"D$_{wt}^{rxn: silicate → metal}$")
        ax_D_wt__T.set_ylabel(r"Partition Coefficient ($D_{wt}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
        ax_D_wt__T.grid(True)
        ax_D_wt__T.set_xlabel("Temperature (K)")
        # ax_D_wt.set_xscale("log")
        # ax_D_wt.set_xlim(left=10, right=1100)  # set x-axis limits for better visibility



        # Legend
        # Legend
        # 1) Grab handles & labels from one of your axes
        handles, labels = ax_KD__T.get_legend_handles_labels()

        # 2) Create a single legend on the right side of the figure
        fig.legend(
            handles,
            labels,
            loc='lower center',        # center vertically on the right edge
            fontsize=8,
            # borderaxespad=0.1,
            # bbox_to_anchor=(1.00, 0.5),
            frameon=False,
            ncol=3,  # number of columns in the legend
            # mode='expand',  # 'expand' to fill the space, 'none'
            labelspacing=1.5,    # default is 0.5; larger → more space
            handletextpad=1.0,   # default is 0.8; larger → more space between handle & text
            columnspacing=2.0,   # if you have multiple columns, space between them
        )

        # 3) Shrink the subplots to make room for the legend
        # fig.subplots_adjust(right=0.82)  # push the plot area leftward





        # fig.subplots_adjust(top=0.88)

        fig.suptitle(
            f"Equilibrium Constant ($K_D$) and Partition Coefficient ($D_{{wt}}$) "
            f"for the reaction {secondary_species}$_{{silicates}}$ $\\rightleftharpoons$ {secondary_species}$_{{metal}}$.\n"
            f"Note: Assumption that X$_{{{secondary_species},silicates}}$ $\\ll$ 1",
            fontsize=10#,
            # y=1.03,            # default ≃0.98, smaller → more gap
        )


        # 3) Layout & save
        # plt.tight_layout(rect=[0, 0, 1, 1.0])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust layout to make room for the title
        plt.savefig(f"KD_D_wt_vs_P_T__T.png", dpi=300)






        ########################################################
        ########################################################
        ########################################################































        print("")
        print("="*50)
        print("")

        # Create a plot with 4 panels corresponding to data from all rows that correspond each to the 4 unique P_T_folders, showing KD_sil_to_metal vs Target temperature (K)

        # df_plot = df.copy()

        # refine df_superset_wo_nan to those with unique pairs of P_T_folders and Target temperature (K)
        df_plot = df.copy()
        # df_plot = df_plot.drop_duplicates(subset=["P_T_folder", "Target temperature (K)"])

        # unique_P_folders = df_plot["Target pressure (GPa)"].unique()
        unique_P_T_folders = df_plot["P_T_folder"].unique()
        # unique_P_T_folders = unique_P_folders 

        # plt.style.use('ggplot')
        # plt.style.use('fivethirtyeight')
        # plt.style.use('bmh')
        # use colors__wrt_P = [cmap(norm(np.log10(p))) for p in [50, 250, 500, 1000]] as default_colors to cycle through
        prop_cycle = plt.rcParams['axes.prop_cycle']
        # default_colors = prop_cycle.by_key()['color']  # e.g. ['#1f77b4', '#ff7f0e', ...]
        default_colors = colors__wrt_P  # use the colors corresponding to P=50,250,500,1000 GPa

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))#, sharex=True, sharey=True)

        counter = 0
        # iterate over unique P_T_folders

        for ind, ith_P_T_folder in enumerate(unique_P_T_folders):

            counter += 1
            print(f"Plotting for P_T_folder: {ith_P_T_folder} ({counter}/{len(unique_P_T_folders)})")

            # narrow df_plot to the current P_T_folder
            df_temporary = df_plot[df_plot["P_T_folder"] == ith_P_T_folder]

            # size of df_temporary
            print(f"Size of df_temporary: {df_temporary.shape}")
            if counter == 1:
                print(f"temp@df_temporary:\n{df_temporary["Target temperature (K)"]}")

            ith_P_T_folder_Pressure = df_temporary["Target pressure (GPa)"].iloc[0]

            # get the axes for the current P_T_folder
            ax = axes.flatten()[list(unique_P_T_folders).index(ith_P_T_folder)]

            # plot KD_sil_to_metal vs Target temperature (K)
            yerr = [[df_temporary["KD_sil_to_metal_low"]], [df_temporary["KD_sil_to_metal_high"]]]
            yerr = np.squeeze(yerr, axis=1)   # now yerr.shape == (2,10)
            ax.errorbar(df_temporary["Target temperature (K)"], df_temporary["KD_sil_to_metal"], yerr=yerr, marker='o', linestyle='', markersize=10, alpha=0.5, capsize=3, elinewidth=1,color=default_colors[ind],label=f"{ith_P_T_folder_Pressure} GPa")
            # ax.errorbar(df_temporary["Target temperature (K)"], df_temporary["KD_sil_to_metal"], yerr=yerr,**marker_opts_error,color=default_colors[ind],label=f"{ith_P_T_folder_Pressure} GPa")

            # set y scale to log
            ax.set_yscale("log")
            
            # set title and labels
            # ax.set_title(f"{ith_P_T_folder}")
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel(r"Equilibrium constant ($K_{D}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")

            # no x labels for the top plots
            if ind < 2:
                ax.set_xlabel("")

            # no y labels for the right plots
            if ind % 2 == 1:
                ax.set_ylabel("")

            # background transparent
            ax.set_facecolor('none')

            # grid
            ax.grid(True)

            # legend
            ax.legend(loc='upper left', fontsize=8, frameon=False)


        # x lim ,  y lim
        # if secondary_species == "He":
        #     y_min = 1e-4 # 1e-5
        #     y_max = 1e0
        # elif secondary_species == "H":
        #     y_min = 1e-0
        #     y_max = 1e4
        # for ax in axes.flatten():
        #     ax.set_ylim(y_min, y_max)
            # ax.set_xlim(left=0, right=200)  # set x-axis limits for better visibility
            # ax.set_xscale("log")  # set x-axis to log scale

        plt.suptitle(
            f"Equilibrium Constant ($K_D$) for the reaction {secondary_species}$_{{silicates}}$ $\\rightleftharpoons$ {secondary_species}$_{{metal}}$.\n"
            f"Note: Assumption that X$_{{{secondary_species},silicates}}$ $\\ll$ 1",
            fontsize=10,
            # y=1.03,            # default ≃0.98, smaller → more gap
        )

        # adjust layout
        # plt.tight_layout()
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



































    if PLOT_MODE == 5 or PLOT_MODE < 0:

        y_min = 1e-5
        y_max = 10**(5.2)

        # low pressure regime < 100 GPa + 5000 K
        # x = P
        xlim_low = 0.
        xlim_high = 70

        fig, axes_KD_D_wt__P = plt.subplots(2, 1, figsize=(12, 10))#, sharex=True, sharey=True)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        # ax1, ax_KD, ax3, ax_D_wt = axes.flatten()
        ax_KD, ax_D_wt = axes_KD_D_wt__P.flatten()


        x_variable = "Target pressure (GPa)"  # x-axis variable for all plots
        z_variable = "Target temperature (K)"  # z-axis variable for all plots -- color coding



        # create a colormap based on the temperature range
        temp_min = 1500  # minimum temperature in K
        temp_max = 5500  # maximum temperature in K
        cbar_ticks = [2000, 3000, 4000, 5000]
        norm = plt.Normalize(
            vmin=temp_min,
            vmax=temp_max
        )

        magma = plt.get_cmap("magma")
        pastel_magma = pastel_cmap(magma, factor=0.25)  # tweak factor between 0 and 1
        cmap = pastel_magma  # use pastel magma for the plots




        # --- Panel 3: KD_sil_to_metal (log y) ---
        # ax_KD.plot(df[x_variable], df["KD_sil_to_metal"], **marker_opts)
        # ax_KD.errorbar(df[x_variable], df["KD_sil_to_metal"], yerr=df["KD_sil_to_metal_error"], **marker_opts_error, color=cmap(norm(df[z_variable])))


        # 1) Plot the colored points
        sc = ax_KD.scatter(
            df[x_variable],
            df["KD_sil_to_metal"],
            c=df[z_variable],
            cmap=cmap,
            norm=norm,
            **marker_opts_scatter,
            marker=marker_TI,  # use the TI marker for scatter points
            label="This study (TI)"
        )

        # 2) Draw per-point errorbars with matching colors
        temps  = df[z_variable].values
        colors = cmap(norm(temps))

        for x0, y0, y_low, y_high, c in zip(
            df[x_variable],
            df["KD_sil_to_metal"],
            df["KD_sil_to_metal_low"],
            df["KD_sil_to_metal_high"],
            colors
        ):

            low  = y0 - y_low
            high = y_high - y0

            # shape (2,1) array: [[low], [high]]
            yerr = [[low], [high]]

            ax_KD.errorbar(
                x0, y0,
                yerr=yerr,
                fmt='none',       # no extra marker
                ecolor=c,         # single RGBA tuple
                **marker_opts_error,
                marker=marker_TI  # use the TI marker for errorbars
            )






        # --- Panel 4: D_wt (log y) ---

        # 1) Plot the colored points
        sc = ax_D_wt.scatter(
            df[x_variable],
            df["D_wt"],
            c=df[z_variable],
            cmap=cmap,
            norm=norm,
            **marker_opts_scatter,
            marker=marker_TI,  # use the TI marker for scatter points
            label="This study (TI)"
        )
        # 2) Draw per-point errorbars with matching colors
        for x0, y0, err0, c in zip(
            df[x_variable],
            df["D_wt"],
            df["D_wt_error"],
            colors
        ):
            ax_D_wt.errorbar(
                x0, y0,
                yerr=err0,
                fmt='none',       # no extra marker
                ecolor=c,         # single RGBA tuple
                **marker_opts_error,
                marker=marker_TI  # use the TI marker for errorbars
            )














        # if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
        if secondary_species == "He":
            """ 
            
            Two-phase simulations for He in Fe and MgSiO3:
            dict: data_He__two_phase_simulations
            P:500, T:9000, KD: 0.032, KD_low: 0.032, KD_high: 0.032
            P:1000, T:13000, KD: 0.32, KD_low: 0.32, KD_high: 0.32

            """

            data_He__two_phase_simulations = {
                "Target pressure (GPa)": [500, 1000],
                "Target temperature (K)": [9000, 13000],
                "KD": [0.032, 0.32],
                "KD_low": [0.032, 0.32],
                "KD_high": [0.032, 0.32]
            }
            # D_wt = KD * (100/56)  # assuming Fe as metal, 56 g/mol
            data_He__two_phase_simulations["D_wt"] = [kd * (100/56) for kd in data_He__two_phase_simulations["KD"]]
            data_He__two_phase_simulations["D_wt_low"] = [kd * (100/56) for kd in data_He__two_phase_simulations["KD_low"]]
            data_He__two_phase_simulations["D_wt_high"] = [kd * (100/56) for kd in data_He__two_phase_simulations["KD_high"]]

            ax_KD.scatter(data_He__two_phase_simulations[x_variable], 
                            data_He__two_phase_simulations["KD"],
                            **marker_opts_scatter,
                            marker=marker_2phase,
                            c=data_He__two_phase_simulations[z_variable],
                            cmap=cmap,
                            norm=norm,
                            label="This study (2P)")
            ax_D_wt.scatter(data_He__two_phase_simulations[x_variable], 
                            data_He__two_phase_simulations["D_wt"],
                            **marker_opts_scatter,
                            marker=marker_2phase,
                            c=data_He__two_phase_simulations[z_variable],
                            cmap=cmap,
                            norm=norm,
                            label="This study (2P)")











        # plot the data points
        plot_studies(
        ax_KD,
        ax_D_wt,
        datasets_comp,
        x_variable,
        z_variable,
        cmap,
        norm,
        marker_other_comp_studies,
        marker_opts_scatter__others,
        marker_opts_error__others
        )









        # plot the data points
        plot_studies(
        ax_KD,
        ax_D_wt,
        datasets_expt,
        x_variable,
        z_variable,
        cmap,
        norm,
        marker_other_expt_studies,
        marker_opts_scatter__others,
        marker_opts_error__others
        )
        ###########################





        # show 1 colorbar for both KD and D_wt plots
        # 1) Create a colorbar for the temperature range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # only needed for older matplotlib versions
        # 2) Add the colorbar to the figure -- width == width of 1 plot
        cbar = fig.colorbar(sm, ax=[ax_KD, ax_D_wt],
                            orientation='horizontal',
                            # fraction=1, pad=0.04,
                            pad=0.1,  # space between colorbar and plot
                            ticks=np.linspace(temp_min, temp_max, 5),
                            location='bottom',  # 'top' or 'bottom'
                            shrink=1,      # shrink to 80% of the original size
                            aspect=50,     # thinner bar
                            )
        cbar.set_label("Temperature (K)")#, rotation=270, labelpad=15)
        # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
        cbar.set_ticks(cbar_ticks)
        # cbar.set_ticklabels([f"{int(t)} K" for t in cbar.get_ticks()])





        # # x lim ,  y lim
        # if secondary_species == "He":
        #     y_min = 1e-5 # 1e-5
        #     y_max = 1e1 #1e1
        # elif secondary_species == "H":
        #     y_min = 1e-3 #1e-3
        #     y_max = 1e6
        ax_KD.set_ylim(y_min, y_max)
        ax_D_wt.set_ylim(y_min, y_max)

        ax_KD.set_xlim(xlim_low, xlim_high)
        ax_D_wt.set_xlim(xlim_low, xlim_high)



        ax_KD.set_yscale("log")
        # ax_KD.set_ylabel(r"K$_D^{rxn: silicate → metal}$")
        # ax_KD.set_ylabel(r"K$_D$")
        ax_KD.set_ylabel(r"Equilibrium Constant ($K_{D}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
        ax_KD.grid(True)
        # ax_KD.tick_params(labelbottom=False)
        # ax_KD.set_xlabel("Pressure (GPa)")
        # ax_KD.set_xscale("log")
        # ax_KD.set_xlim(left=0,right=200)  # set x-axis limits for better visibility

        ax_D_wt.set_yscale("log")
        # ax_D_wt.set_ylabel(r"D$_{wt}^{rxn: silicate → metal}$")
        ax_D_wt.set_ylabel(r"Partition Coefficient ($D_{wt}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
        ax_D_wt.grid(True)
        ax_D_wt.set_xlabel("Pressure (GPa)")
        # ax_D_wt.set_xscale("log")
        # ax_D_wt.set_xlim(left=10, right=1100)  # set x-axis limits for better visibility



        # Legend
        # 1) Grab handles & labels from one of your axes
        handles, labels = ax_KD.get_legend_handles_labels()

        # 2) Create a single legend on the right side of the figure
        fig.legend(
            handles,
            labels,
            loc='lower center',        # center vertically on the right edge
            fontsize=8,
            # borderaxespad=0.1,
            # bbox_to_anchor=(1.00, 0.5),
            frameon=False,
            ncol=3,  # number of columns in the legend
            # mode='expand',  # 'expand' to fill the space, 'none'
            labelspacing=1.5,    # default is 0.5; larger → more space
            handletextpad=1.0,   # default is 0.8; larger → more space between handle & text
            columnspacing=2.0,   # if you have multiple columns, space between them
        )

        # 3) Shrink the subplots to make room for the legend
        # fig.subplots_adjust(right=0.82)  # push the plot area leftward





        # fig.subplots_adjust(top=0.88)

        fig.suptitle(
            f"Equilibrium Constant ($K_D$) and Partition Coefficient ($D_{{wt}}$) "
            f"for the reaction {secondary_species}$_{{silicates}}$ $\\rightleftharpoons$ {secondary_species}$_{{metal}}$.\n"
            f"Note: Assumption that X$_{{{secondary_species},silicates}}$ $\\ll$ 1",
            fontsize=10#,
            # y=1.03,            # default ≃0.98, smaller → more gap
        )


        # 3) Layout & save
        # plt.tight_layout(rect=[0, 0, 1, 1.0])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust layout to make room for the title
        plt.savefig(f"KD_D_wt_vs_P_T__lowPT.png", dpi=300)











    if PLOT_MODE == 6 or PLOT_MODE < 0:

        ########################################################
        ########################################################
        ########################################################
        ########################################################
        ## KD, D_wt vs Temperature for the current study (TI) --- LOW PRESSURE REGIME < 100 GPa + 5000 K

        # x = T
        xlim_min = 1500
        xlim_max = 5500

        # plt.rcParams.update(INITIAL_RCP)
        # plt.rcdefaults()

        fig, axes_KD_D_wt__T = plt.subplots(2, 1, figsize=(12, 10))#, sharex=True, sharey=True)
        ax_KD__T, ax_D_wt__T = axes_KD_D_wt__T.flatten()



        # add log pressure to df
        # df["log(Target pressure (GPa))"] = np.log10(df["Target pressure (GPa)"])
        # for data in datasets_comp:
        #     data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]
        # for data in datasets_expt:
        #     data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]

        x_variable = "Target temperature (K)"  # x-axis variable for all plots
        # z_variable = "log(Target pressure (GPa))"  # z-axis variable for all plots -- color coding
        z_variable = "Target pressure (GPa)"  # z-axis variable for all plots -- color coding



        # create a colormap based on the temperature range
        if z_variable == "log(Target pressure (GPa))":
            log10_pres_min = np.log10(0.3)  # minimum pressure in GPa
            log10_pres_max = np.log10(70)  # maximum pressure in GPa
            cbar_ticks = np.log10([1, 10, 100, 1000])
            norm = plt.Normalize(
                vmin=log10_pres_min,
                vmax=log10_pres_max
            )
        elif z_variable == "Target pressure (GPa)":
            pres_min = 0.1  # minimum pressure in GPa
            pres_max = 70  # maximum pressure in GPa
            cbar_ticks = [0, 10, 20, 30, 40, 50, 60, 70]
            norm = plt.Normalize(
                vmin=pres_min,
                vmax=pres_max
            )


        viridis = plt.get_cmap("viridis")
        pastel_viridis = pastel_cmap(viridis, factor=0.25)  # tweak factor between 0 and 1
        cmap = pastel_viridis  # use pastel viridis for the plots






        # --- Panel 3: KD_sil_to_metal (log y) ---
        # ax_KD.plot(df[x_variable], df["KD_sil_to_metal"], **marker_opts)
        # ax_KD.errorbar(df[x_variable], df["KD_sil_to_metal"], yerr=df["KD_sil_to_metal_error"], **marker_opts_error, color=cmap(norm(df[z_variable])))


        # 1) Plot the colored points
        sc = ax_KD__T.scatter(
            df[x_variable],
            df["KD_sil_to_metal"],
            c=df[z_variable],
            cmap=cmap,
            norm=norm,
            **marker_opts_scatter,
            marker=marker_TI,  # use the TI marker for scatter points
            label="This study (TI)"
        )

        # 2) Draw per-point errorbars with matching colors
        temps  = df[z_variable].values
        colors = cmap(norm(temps))

        for x0, y0, y_low, y_high, c in zip(
            df[x_variable],
            df["KD_sil_to_metal"],
            df["KD_sil_to_metal_low"],
            df["KD_sil_to_metal_high"],
            colors
        ):

            low  = y0 - y_low
            high = y_high - y0

            # shape (2,1) array: [[low], [high]]
            yerr = [[low], [high]]

            ax_KD__T.errorbar(
                x0, y0,
                yerr=yerr,
                fmt='none',       # no extra marker
                ecolor=c,         # single RGBA tuple
                **marker_opts_error,
                marker=marker_TI  # use the TI marker for errorbars
            )


        # --- Panel 4: D_wt (log y) ---

        # 1) Plot the colored points
        sc = ax_D_wt__T.scatter(
            df[x_variable],
            df["D_wt"],
            c=df[z_variable],
            cmap=cmap,
            norm=norm,
            **marker_opts_scatter,
            marker=marker_TI,  # use the TI marker for scatter points
            label="This study (TI)"
        )
        # 2) Draw per-point errorbars with matching colors
        for x0, y0, err0, c in zip(
            df[x_variable],
            df["D_wt"],
            df["D_wt_error"],
            colors
        ):
            ax_D_wt__T.errorbar(
                x0, y0,
                yerr=err0,
                fmt='none',       # no extra marker
                ecolor=c,         # single RGBA tuple
                **marker_opts_error,
                marker=marker_TI  # use the TI marker for errorbars
            )





        # plot the data points -- computational studies
        plot_studies(
        ax_KD__T,
        ax_D_wt__T,
        datasets_comp,
        x_variable,
        z_variable,
        cmap,
        norm,
        marker_other_comp_studies,
        marker_opts_scatter__others,
        marker_opts_error__others
        )

        # plot the data points -- experimental studies
        plot_studies(
        ax_KD__T,
        ax_D_wt__T,
        datasets_expt,
        x_variable,
        z_variable,
        cmap,
        norm,
        marker_other_expt_studies,
        marker_opts_scatter__others,
        marker_opts_error__others
        )


        # plt.savefig(f"KD_D_wt_vs_P_T__T.png", dpi=300)
        # exit(0)  # exit here to avoid plotting the second figure


        # show 1 colorbar for both KD and D_wt plots
        # 1) Create a colorbar for the temperature range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # only needed for older matplotlib versions
        # 2) Add the colorbar to the figure -- width == width of 1 plot
        if z_variable == "log(Target pressure (GPa))":
            cbar = fig.colorbar(sm, ax=[ax_KD__T, ax_D_wt__T],
                                orientation='horizontal',
                                # fraction=1, pad=0.04,
                                pad=0.1,  # space between colorbar and plot
                                # ticks=np.log10([1, 10, 100, 1000]),
                                location='bottom',  # 'top' or 'bottom'
                                shrink=1,      # shrink to 80% of the original size
                                aspect=50,     # thinner bar
                                )
            cbar.set_label("Pressure (GPa)")#, rotation=270, labelpad=15)
            # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(["1", "10", "100", "1000"])  # set tick labels to be in GPa

            ## grab colors used for P=50,250,500,1000 GPa
            # colors__wrt_P = [cmap(norm(np.log10(p))) for p in [50, 250, 500, 1000]]

        elif z_variable == "Target pressure (GPa)":
            cbar = fig.colorbar(sm, ax=[ax_KD__T, ax_D_wt__T],
                                orientation='horizontal',
                                # fraction=1, pad=0.04,
                                pad=0.1,  # space between colorbar and plot
                                # ticks=np.log10([1, 10, 100, 1000]),
                                location='bottom',  # 'top' or 'bottom'
                                shrink=1,      # shrink to 80% of the original size
                                aspect=50,     # thinner bar
                                )
            cbar.set_label("Pressure (GPa)")#, rotation=270, labelpad=15)
            # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
            cbar.set_ticks(cbar_ticks)
            # cbar.set_ticklabels(["0", "200", "400", "600", "800", "1000"])  # set tick labels to be in GPa

            ## grab colors used for P=50,250,500,1000 GPa
            # colors__wrt_P = [cmap(norm(p)) for p in [50, 250, 500, 1000]]





        # x lim ,  y lim
        # if secondary_species == "He":
        #     y_min__T = 1e-5 # 1e-5
        #     y_max__T = 1e3 #1e1
        # elif secondary_species == "H":
        #     y_min__T = 1e-5 #1e-3
        #     y_max__T = 1e3
        ax_KD__T.set_ylim(y_min, y_max)
        ax_D_wt__T.set_ylim(y_min, y_max)

        ax_KD__T.set_xlim(xlim_min, xlim_max)
        ax_D_wt__T.set_xlim(xlim_min, xlim_max)


        ax_KD__T.set_yscale("log")
        # ax_KD.set_ylabel(r"K$_D^{rxn: silicate → metal}$")
        # ax_KD.set_ylabel(r"K$_D$")
        ax_KD__T.set_ylabel(r"Equilibrium Constant ($K_{D}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
        ax_KD__T.grid(True)
        # ax_KD.tick_params(labelbottom=False)
        # ax_KD.set_xlabel("Pressure (GPa)")
        # ax_KD__T.set_xscale("log")
        # ax_KD.set_xlim(left=0,right=200)  # set x-axis limits for better visibility

        ax_D_wt__T.set_yscale("log")
        # ax_D_wt.set_ylabel(r"D$_{wt}^{rxn: silicate → metal}$")
        ax_D_wt__T.set_ylabel(r"Partition Coefficient ($D_{wt}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
        ax_D_wt__T.grid(True)
        ax_D_wt__T.set_xlabel("Temperature (K)")
        # ax_D_wt.set_xscale("log")
        # ax_D_wt.set_xlim(left=10, right=1100)  # set x-axis limits for better visibility



        # Legend
        # Legend
        # 1) Grab handles & labels from one of your axes
        handles, labels = ax_KD__T.get_legend_handles_labels()

        # 2) Create a single legend on the right side of the figure
        fig.legend(
            handles,
            labels,
            loc='lower center',        # center vertically on the right edge
            fontsize=8,
            # borderaxespad=0.1,
            # bbox_to_anchor=(1.00, 0.5),
            frameon=False,
            ncol=3,  # number of columns in the legend
            # mode='expand',  # 'expand' to fill the space, 'none'
            labelspacing=1.5,    # default is 0.5; larger → more space
            handletextpad=1.0,   # default is 0.8; larger → more space between handle & text
            columnspacing=2.0,   # if you have multiple columns, space between them
        )

        # 3) Shrink the subplots to make room for the legend
        # fig.subplots_adjust(right=0.82)  # push the plot area leftward





        # fig.subplots_adjust(top=0.88)

        fig.suptitle(
            f"Equilibrium Constant ($K_D$) and Partition Coefficient ($D_{{wt}}$) "
            f"for the reaction {secondary_species}$_{{silicates}}$ $\\rightleftharpoons$ {secondary_species}$_{{metal}}$.\n"
            f"Note: Assumption that X$_{{{secondary_species},silicates}}$ $\\ll$ 1",
            fontsize=10#,
            # y=1.03,            # default ≃0.98, smaller → more gap
        )


        # 3) Layout & save
        # plt.tight_layout(rect=[0, 0, 1, 1.0])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust layout to make room for the title
        plt.savefig(f"KD_D_wt_vs_P_T__T__lowPT.png", dpi=300)






        ########################################################
        ########################################################
        ########################################################



























    if PLOT_MODE == 7 or PLOT_MODE < 0:

        ylim_low = 1500
        ylim_high = 18000

        # low pressure regime < 100 GPa + 5000 K
        # x = P
        xlim_low = 0.3
        xlim_high = 1600

        fig, axes_PT = plt.subplots(1, 1, figsize=(10, 10))


        x_variable = "Target pressure (GPa)"  # x-axis variable for all plots
        y_variable = "Target temperature (K)"  # z-axis variable for all plots -- color coding
        z_variable = "KD_sil_to_metal"  # z-axis variable for all plots -- color coding
        z_variable = "log10(KD_sil_to_metal)"  # z-axis variable for all plots -- color coding



        # create a colormap based on the temperature range
        if secondary_species == "H":
            z_min = 1e-2  # minimum temperature in K
            z_max = 1e3  # maximum temperature in K
        elif secondary_species == "He":
            z_min = 1e-5  # minimum temperature in K
            z_max = 1e0  # maximum temperature in
        log10_z_min = np.log10(z_min)
        log10_z_max = np.log10(z_max)
        # cbar_ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
        # cbar_ticks = [2000, 3000, 4000, 5000]
        # norm = plt.Normalize(
        #     vmin=z_min,
        #     vmax=z_max
        # )
        from matplotlib.colors import LogNorm
        norm = LogNorm(
            vmin=z_min,
            vmax=z_max
        )
        # norm = LogNorm(vmin=1e-6, vmax=1e5)

        magma = plt.get_cmap("magma")
        pastel_magma = pastel_cmap(magma, factor=0.25)  # tweak factor between 0 and 1
        cmap = pastel_magma  # use pastel magma for the plots



        df_plot = df.copy()
        sc = axes_PT.scatter(
            df_plot["Target pressure (GPa)"],
            df_plot["Target temperature (K)"],
            c = df_plot["KD_sil_to_metal"],
            norm = norm,
            cmap = cmap,
            **marker_opts_scatter,
            marker = marker_TI,
            label  = "This study"
        )

        # 4) Overlay each computational‐study dict
        for data, marker in zip(datasets_comp, marker_other_comp_studies):
            xs = [float(v) for v in data["Target pressure (GPa)"]]
            ys = [float(v) for v in data["Target temperature (K)"]]
            zs = [float(v) for v in data["KD_sil_to_metal"]]
            axes_PT.scatter(
                xs, ys,
                c = zs,
                norm = norm,
                cmap = cmap,
                **marker_opts_scatter__others,
                marker = marker,
                label  = data.get("label", "Comp. study")
            )

        # 5) Overlay experimental‐study dicts similarly
        for data, marker in zip(datasets_expt, marker_other_expt_studies):
            xs = [float(v) for v in data["Target pressure (GPa)"]]
            ys = [float(v) for v in data["Target temperature (K)"]]
            zs = [float(v) for v in data["KD_sil_to_metal"]]
            axes_PT.scatter(
                xs, ys,
                c = zs,
                norm = norm,
                cmap = cmap,
                **marker_opts_scatter__others,
                marker = marker,
                label  = data.get("label", "Expt. study")
            )


        cbar = fig.colorbar(
            sc,
            ax=axes_PT,
            orientation="vertical",
            pad=0.1,
            shrink=1,
            aspect=50,
            ticks=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1e3,1e4,1e5]
        )
        cbar.set_label(r"$K_D$") 




        # # x lim ,  y lim
        # if secondary_species == "He":
        #     y_min = 1e-5 # 1e-5
        #     y_max = 1e1 #1e1
        # elif secondary_species == "H":
        #     y_min = 1e-3 #1e-3
        #     y_max = 1e6
        # axes_PT.set_ylim(y_min, y_max)
        

        # axes_PT.set_xlim(xlim_low, xlim_high)



        axes_PT.set_yscale("log")
        axes_PT.set_ylabel(r"Temperature (K)")
        axes_PT.grid(True)
        # y axis ticks
        axes_PT.set_yticks([2000, 4000, 8000, 16000])
        axes_PT.set_yticklabels([r"2000", r"4000", r"8000", r"16000"])
        # axes_PT.tick_params(labelbottom=False)
        axes_PT.set_xlabel("Pressure (GPa)")
        axes_PT.set_xscale("log")
        axes_PT.set_xticks([1, 10, 100, 1000])
        axes_PT.set_xticklabels([r"1", r"10", r"100", r"1000"])
        # axes_PT.set_xlim(left=0,right=200)  # set x-axis limits for better visibility

        axes_PT.set_xlim(xlim_low, xlim_high)
        axes_PT.set_ylim(ylim_low, ylim_high)

        # Legend
        # 1) Grab handles & labels from one of your axes
        handles, labels = axes_PT.get_legend_handles_labels()

        # 2) Create a single legend on the right side of the figure
        # fig.legend(
        #     handles,
        #     labels,
        #     loc='lower center',        # center vertically on the right edge
        #     fontsize=8,
        #     # borderaxespad=0.1,
        #     # bbox_to_anchor=(1.00, 0.5),
        #     frameon=False,
        #     ncol=3,  # number of columns in the legend
        #     # mode='expand',  # 'expand' to fill the space, 'none'
        #     labelspacing=1.5,    # default is 0.5; larger → more space
        #     handletextpad=1.0,   # default is 0.8; larger → more space between handle & text
        #     columnspacing=2.0,   # if you have multiple columns, space between them
        # )

        axes_PT.grid(True, which="both", ls="--", alpha=0.3)


        axes_PT.legend(frameon=False, fontsize=8,
                        labelspacing=1.0,    # default is 0.5; larger → more space
                        handletextpad=1.0)   # default is 0.8; larger → more space between handle & text)


        # 3) Shrink the subplots to make room for the legend
        # fig.subplots_adjust(right=0.82)  # push the plot area leftward





        # fig.subplots_adjust(top=0.88)

        fig.suptitle(
            f"Partitioning of {secondary_species} between silicate and metal as a function of pressure and temperature.\n",
            fontsize=10#,
            # y=1.03,            # default ≃0.98, smaller → more gap
        )


        # 3) Layout & save
        # plt.tight_layout(rect=[0, 0, 1, 1.0])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust layout to make room for the title
        plt.savefig(f"PT_coverage.png", dpi=300)



