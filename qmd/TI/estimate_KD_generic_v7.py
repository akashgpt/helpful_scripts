#!/usr/bin/env python3
"""
Estimate chemical potentials and partition coefficients/equilibrium constants for {secondary_species} in Fe and MgSiO3 systems from TI data.
Walks through directory structure, parses log.Ghp_analysis files, assembles results into a DataFrame,
computes mixing fractions, fits linear excess chemical potentials, and adds entropy corrections.
Partially based on the discussion in the paper Li, et al. 2022 paper, "Primitive noble gases sampled from ocean island basalts cannot be from the Earth's core".

v2: Evaluates isobar_calc
v3: Does asymmetric error propagation for KD, etc.

Usage: python $HELP_SCRIPTS_TI/estimate_KD_generic_v7.py -s He > log.estimate_KD_generic_v7 2>&1

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

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from mc_error import \
    monte_carlo_error  # located at $HELP_SCRIPTS/general/mc_error.py
from mc_error import \
    monte_carlo_error_asymmetric  # located at $HELP_SCRIPTS/general/mc_error.py
from mc_error import \
    monte_carlo_error_asymmetric_w_bounds  # located at $HELP_SCRIPTS/general/mc_error.py
from mc_error import \
    monte_carlo_error_asymmetric_w_io_bounds  # located at $HELP_SCRIPTS/general/mc_error.py
from mc_error import \
    monte_carlo_error_asymmetric_w_io_bounds_vectorized_outputs  # located at $HELP_SCRIPTS/general/mc_error.py
from scipy.optimize import curve_fit

# Boltzmann constant in eV/K for entropy term
kB = 8.617333262145e-5
mu_MgSiO3 = 100.0  # atomic mass of MgSiO3 in g/mol; Mg: 24.305, Si: 28.0855, O: 15.999
mu_Fe = 55.845  # atomic mass of Fe in g/mol
mu_H = 1.00784  # atomic mass of H in g/mol
mu_H2 = 2 * mu_H  # atomic mass of H2 in g/mol
mu_He = 4.002602  # atomic mass of He in g/mol
epsilon = 1e-10  # small value to avoid division by zero
num_array_X=100  # number of points in the log-spaced array for X_{secondary_species}
max_array_X=0.2 #100/109  # maximum value of X_{secondary_species} (corresponds to a wt fraction of 0.1)

SCRIPT_MODE             = 1 # >0, only plot; <0, only do analysis; 0: both
PLOT_MODE               = 31 # 21 #-1: plot all; 0: do not plot, 1: plot #1, 2: plot #2, 3: plot #3 ...
TEMPERATURES_TO_REMOVE  = [10400, 7200, 5200]  # temperatures to remove from the final DataFrame
FIT_MODE                = 0 # 1: plot fit to data; 0: do not plot fit
H_STOICH_MODE           = 1 # 2: calculate for H2, 1: calculate for H



# 0) 


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





    # df -- create df["array_X_{secondary_species}"] -- log varying X from 1e-10 to 100/109 (corresponds to a wt fraction of 0.1)
    # 1) build your “X” array once
    arr_X = np.logspace(np.log10(1e-10), np.log10(max_array_X), num=num_array_X)

    arr_inf = np.nan * np.ones_like(arr_X)  # create an array of NaN with the same shape as arr_X

    arr_PT = np.nan * np.ones_like(arr_X)  # create an array of NaN with the same shape as arr_X

    # 2) define which prefixes get the full `arr` and which get `np.nan`
    array_cols = {
        # these four get the log‐spaced array
        **{f"array_X_{secondary_species}{suffix}": arr_X
        for suffix in ["", "_error", "_lower", "_upper"]},
        # these four groups get initialized to NaN
        **{f"{prefix}_{secondary_species}{suffix}": arr_inf
        for prefix in ["array_Xw", "array_KD", "array_D_wt", "array_KD_prime"]
        for suffix in ["", "_error", "_lower", "_upper"]},
        
        **{f"array_{prefix}_{secondary_species}": arr_PT
        for prefix in ["T", "P"]},

        **{f"array_X_{secondary_species}{suffix}": arr_X
        for suffix in ["_in_Fe", "_in_Fe_error", "_in_Fe_lower", "_in_Fe_upper"]},
    }
    # array_Xw_{secondary_species} -- Weight fraction corresponding to array_X_{secondary_species} -- exact
    # array_KD: corresponding KD -- exact
    # array_D_wt: corresponding D_wt -- exact
    # KD_prime: array_KD_prime_{secondary_species} where KD_prime represents the KD for <<H2 (silicate) = H (metal)>> reaction (rather than the default <<H (silicate) = H (metal)>> reaction) -- exact
    
    # 3) loop once to add them all, broadcasting each fill to every row
    for col, fill in array_cols.items():
        # [fill] * len(df) creates a list of length df with the same object/value
        # object-dtype is inferred automatically when the fill is an array
        df[col] = pd.Series([fill] * len(df), index=df.index, dtype="object")



    # print(f"df[array_KD_secondary_species]:\n{df[f'array_X_{secondary_species}']}\n\n")
    # print(f"df[array_KD_secondary_species]:\n{df[f'array_KD_{secondary_species}']}\n\n")
    # print(f"Shape of df[array_KD_secondary_species]: {df.at[0,f'array_X_{secondary_species}'].shape}\n\n")

    # exit(0)  # exit after parsing and creating the DataFrame



    # sort all columns wrt "Phase" and "Target pressure (GPa)"
    df.sort_values(by=["Phase", "Target pressure (GPa)"], inplace=True)
    # df.sort_values(by=["Target pressure (GPa)"], inplace=True)









    # 7) Compute {secondary_species} mole fraction X_{secondary_species} = n_{secondary_species} / total Fe or Mg for each row depending on phase

    # def frac_secondary_species(row):
    #     counts = row["Atom counts"]
    #     species = row["Unique species"]
    #     mapping = dict(zip(species, counts))
    #     total = sum(counts)
    #     return mapping.get(secondary_species, 0) / total if total else 0.0

    # df[f"X_{secondary_species}"] = df.apply(frac_secondary_species, axis=1)

    def frac_secondary_species(row):
        species = row["Unique species"]       # e.g. ["Fe","O","Si","X"]
        counts  = row["Atom counts"]          # e.g. [  4, 12,  4,  1]
        mapping = dict(zip(species, counts))  # {"Fe":4, "O":12, "Si":4, "X":1}

        # how many secondary atoms?
        n_sec = mapping.get(secondary_species, 0)

        # pick the right primary
        phase = row["Phase"]
        if phase == f"Fe_{secondary_species}":
            n_prim = mapping.get("Fe", 0)
        elif phase == f"MgSiO3_{secondary_species}":
            n_prim = mapping.get("Mg", 0)
        else:
            # if you have other phases, either return 0 or np.nan
            return 0.0

        total = n_prim + n_sec
        return (n_sec / total) if (total > 0) else 0.0

    df[f"X_{secondary_species}"] = df.apply(frac_secondary_species, axis=1)
    print(f"*** WARNING: Is X_{secondary_species} computed correctly? Check the equations below with Haiyang!!! ***\n")







    # initialize all species columns
    df["Total # of species"] = np.nan
    df["G_hp_per_species"] = np.nan
    df["G_hp_per_species_error"] = np.nan
    df["G_hp_per_species_w_TS"] = np.nan
    df["G_hp_per_species_w_TS_error"] = np.nan

    # for each row, compute the total number of species
    for i, row in df.iterrows():
        species = row["Unique species"]       # e.g. ["Fe","O","Si","X"]
        counts  = row["Atom counts"]          # e.g. [  4, 12,  4,  1]
        mapping = dict(zip(species, counts))  # {"Fe":4, "O":12, "Si":4, "X":1}

        # how many secondary atoms?
        n_sec = mapping.get(secondary_species, 0)

        # pick the right primary
        phase = row["Phase"]
        if phase == f"Fe_{secondary_species}":
            n_prim = mapping.get("Fe", 0)
        elif phase == f"MgSiO3_{secondary_species}":
            n_prim = mapping.get("Mg", 0)

        total = n_prim + n_sec
        df.at[i, "Total # of species"] = total

    print(f"Total # of species:\n{df['Total # of species']}\n\n")

    # print phase, directory name and X_{secondary_species} for each row
    # for i, row in df.iterrows():
    #     print(f"Phase: {row['Phase']}, P_T_folder: {row['Directory name']}, X_{secondary_species}: {row[f'X_{secondary_species}']:.6f}")

    # exit(0)  # exit after computing X_{secondary_species} for each row

    # 8) Drop any columns starting with WARNING
    warn_cols = [c for c in df.columns if c.startswith("WARNING")]
    if warn_cols:
        df.drop(columns=warn_cols, inplace=True)





    # TS_per_atom = df["TS"] / df["Total # of atoms"]
    df["TS_per_atom"] = df["TS"] / df["Total # of atoms"]
    df["TS_per_species"] = df["TS"] / df["Total # of species"]
    # print(f"TS_per_species:\n{df['TS_per_species']}\n\n")

    # initialize G_hp_per_species, G_hp_per_species_w_TS to df
    df["G_hp_per_species"] = np.nan
    df["G_hp_per_species_w_TS"] = np.nan


    print(f"\nWARNING: Check the equations below with Haiyang!!!\n")
    # G_hp_per_atom_w_TS = df["G_hp_per_atom"] + df["TS_per_atom"]
    # for all cases except those with n_{secondary_species} = 0
    for i, row in df.iterrows():
        if row[f"X_{secondary_species}"] > 0:
            df.at[i, "G_hp_per_atom_w_TS"] = row["G_hp_per_atom"] + row["TS_per_atom"]
            df.at[i, "G_hp_per_species"] = row["G_hp_per_atom"] * row["Total # of atoms"] / row["Total # of species"]
            df.at[i, "G_hp_per_species_w_TS"] = row["TS_per_species"] + (row["G_hp_per_atom"] * row["Total # of atoms"] / row["Total # of species"])
            # print(f"G_hp_per_species_w_TS: {df.at[i, 'G_hp_per_species_w_TS']}")
            # print(f"TS_per_species: {row['TS_per_species']}")
            # print(f"G_hp_per_species: {row['G_hp_per_species']}")
            # print(f"row['G_hp_per_atom'] * row['Total # of atoms']: {row['G_hp_per_atom'] * row['Total # of atoms']}")
            # print(f"row['G_hp_per_atom'] * row['Total # of atoms'] / row['Total # of species']: {row['G_hp_per_atom'] * row['Total # of atoms'] / row['Total # of species']}")

        else:
            df.at[i, "G_hp_per_atom_w_TS"] = row["G_hp_per_atom"]
            df.at[i, "G_hp_per_species"] = row["G_hp_per_atom"] * row["Total # of atoms"] / row["Total # of species"]
            df.at[i, "G_hp_per_species_w_TS"] = (row["G_hp_per_atom"] * row["Total # of atoms"] / row["Total # of species"])  # no TS term if X_{secondary_species} = 0
            # df.at[i, "G_hp_per_atom"] = row["G_hp_per_atom"] - row["TS_per_atom"]
            
    # df["G_hp_per_atom_w_TS"] = df["G_hp_per_atom"] + df["TS_per_atom"]
    # print G_hp_per_atom_w_TS vs G_hp_per_species_w_TS
    # print(f"G_hp_per_atom_w_TS:\n{df['G_hp_per_atom_w_TS']}\n\n")
    # print(f"G_hp_per_species_w_TS:\n{df['G_hp_per_species_w_TS']}\n\n")

    print(f"WARNING: the line above was not commented earlier but I just did it now as I can't figure out why it was added in the first place. It seems to be a mistake though doesn't affect results previous or current in any way.\n\n")


    df["G_hp_per_atom_w_TS_error"] = df["G_hp_per_atom_error"]  # assuming no error in TS term
    df["G_hp_per_species_error"] = df["G_hp_per_atom_error"] * df["Total # of atoms"] / df["Total # of species"]
    df["G_hp_per_species_w_TS_error"] = df["G_hp_per_species_error"]  # assuming no error in TS term


    # mu = mu_excess + mu_TS_term

    # 9) Fit linear excess chemical potential mu_excess = a + b
    # Initialize columns
    df["intercept"] = np.nan
    df["slope"] = np.nan
    df["intercept_error"] = np.nan
    df["slope_error"] = np.nan

    # Group by Phase and P_T_folder to fit separate lines
    for (phase, pt), sub in df.groupby(["Phase", "P_T_folder"]):
        x = sub[f"X_{secondary_species}"].values
        # y = sub["G_hp_per_atom_w_TS"].values
        y = sub["G_hp_per_species_w_TS"].values  # use G_hp_per_species_w_TS for fitting
        # y_error = sub["G_hp_per_atom_w_TS_error"].values # 1 sigma error in G_hp_per_atom_w_TS
        y_error = sub["G_hp_per_species_w_TS_error"].values  # 1 sigma error in G_hp_per_species_w_TS
        # weights = 1/σ_y
        # w = 1.0 / y_error
        # print(f"Fitting mu_excess for Phase: {phase}, P_T_folder: {pt} with {len(x)} points")
        # print(f"  X_{secondary_species} values: {x}")
        # print(f"  G_hp_per_species_w_TS values: {y}")
        # print(f"  G_hp_per_species_w_TS_error values: {y_error}")
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
        X = max(X, epsilon**2)  # avoid log(0) or negative values

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


    # partiction coefficient: (1.78/5) * np.exp(-(mu_excess_{secondary_species}_for_Fe - mu_excess_{secondary_species}_for_MgSiO3) / (kB * T)) for the same P_T_folder
    def compute_KD__exact(row,df_evaluate=None):
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
        fn_KD__part = lambda mu_excess, other__mu_excess, T, mult_factor: np.exp(-mult_factor * (mu_excess - other__mu_excess) / (kB * T))
        KD__part, KD__part_error, KD__part_lower, KD__part_upper = monte_carlo_error_asymmetric(
            fn_KD__part,
            [mu_excess, other__mu_excess, T, mult_factor],
            [mu_excess-mu_excess_error, other__mu_excess-other__mu_excess_error, T-0.0, mult_factor-0.0],
            [mu_excess+mu_excess_error, other__mu_excess+other__mu_excess_error, T+0.0, mult_factor+0.0],
        )



        # fn_estimate__X_in_Fe # i.e. estimating array_X_{secondary_species} (Fe)
        if phase == f"MgSiO3_{secondary_species}":
            array_X_in_MgSiO3 = row[f"array_X_{secondary_species}"]
        else:
            array_X_in_MgSiO3 = df_evaluate.loc[mask, f"array_X_{secondary_species}"].iloc[0]

        fn_estimate__X_in_Fe = lambda array_X_in_MgSiO3, KD__part: (array_X_in_MgSiO3/(5 - 4*array_X_in_MgSiO3)) * KD__part
        array_X_in_Fe = fn_estimate__X_in_Fe(
            array_X_in_MgSiO3, KD__part
        )




        if phase == f"Fe_{secondary_species}":
            df_evaluate.loc[mask, f"array_X_{secondary_species}"] = array_X_in_MgSiO3
            row__array_X_in_MgSiO3 = np.nan * np.ones_like(array_X_in_Fe)  # fill with NaN -- i.e., if Fe_{secondary_species} phase, then we do not have X_in_MgSiO3
        else:
            row__array_X_in_MgSiO3 = array_X_in_MgSiO3

        fn_KD = lambda KD__part, X_in_MgSiO3: KD__part / (5 - 4*X_in_MgSiO3)
        # KD, KD_error = monte_carlo_error(fn_KD, [mu_excess, other__mu_excess, T, mult_factor], [mu_excess_error, other__mu_excess_error, 0.0, 0.0])
        KD, KD_error, KD_lower, KD_upper = monte_carlo_error_asymmetric(
            fn_KD,
            [KD__part, array_X_in_MgSiO3],
            [KD__part_lower, array_X_in_MgSiO3 - 0.0],
            [KD__part_upper, array_X_in_MgSiO3 + 0.0],
        )

        return KD, KD_error, KD_lower, KD_upper, row__array_X_in_MgSiO3, df_evaluate

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






















    # exit(0)

    print(f"="*50)
    print(f"Note: It is assumed here that y or X_{secondary_species} in silicates is << 1 for KD, ... . array_KD, etc. are the exact ones. !")
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


            GH_analysis_file = isobar_dir / "GH_analysis.csv"


            # 1) grab only “T…” dirs
            temp_dirs = [
                            d for d in isobar_dir.iterdir()
                            if d.is_dir()
                            and d.name.startswith("T")
                            and d.name[1:].isdigit()   # everything after the "T" is numeric
                        ]

            # remove all entries with T == TEMPERATURES_TO_REMOVE, e.g., 10400, 7200, 5200
            temp_dirs = [d for d in temp_dirs if d.name[1:] not in [str(t) for t in TEMPERATURES_TO_REMOVE]]

            # Make 5 duplicate rows based on all df_base entries, and save this in df_isobar.
            df_isobar = df_base.copy()
            df_isobar = pd.concat([df_isobar] * len(temp_dirs), ignore_index=True) # 4 new ones/isobar + the original

            # make all values nan
            # df_isobar.loc[:, df_isobar.columns != "Config_folder"] = np.nan

            # 2) sort by the numeric part after “T”
            def _parse_temp(d: Path) -> float:
                try:
                    return float(d.name[1:])
                except ValueError:
                    return float('inf')  # push bad names to the end

            temp_dirs.sort(key=_parse_temp)
            # print(f"Found {len(temp_dirs)} temperature directories in {isobar_dir}.")

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
                df_isobar.loc[idx, "TS_per_species"] = (df_base["TS_per_species"].iloc[0] / df_base["Target temperature (K)"].iloc[0]) * temp_val
                # read GH_analysis.csv as GH_df and grab the GFE and GFE_error based on corresponding GH_df["T_target"]==df_isobar["Target temperature (K)"] and replace "G_hp", "G_hp_error" in df_isobar
                df_isobar.loc[idx, "G_hp"] = GH_df.loc[GH_df["T_target"] == temp_val, "GFE"].values[0]
                df_isobar.loc[idx, "G_hp_error"] = GH_df.loc[GH_df["T_target"] == temp_val, "GFE_err"].values[0]

                
                # Calculate G_hp_per_atom = G_hp / Total # of atoms and G_hp_per_atom_error = G_hp_per_atom / Total # of atoms, and update df_isobar with these values.
                total_atoms = df_base["Total # of atoms"].iloc[0]
                total_species = df_base["Total # of species"].iloc[0]
                df_isobar.loc[idx, "G_hp_per_atom"] = df_isobar.loc[idx, "G_hp"] / total_atoms
                df_isobar.loc[idx, "G_hp_per_species"] = df_isobar.loc[idx, "G_hp"] / total_species
                df_isobar.loc[idx, "G_hp_per_atom_error"] = df_isobar.loc[idx, "G_hp_error"] / total_atoms
                df_isobar.loc[idx, "G_hp_per_species_error"] = df_isobar.loc[idx, "G_hp_error"] / total_species

                # Then update "G_hp_per_atom_w_TS" in df_isobar as well, using the same formula as above in this script, i.e., G_hp_per_atom_w_TS = G_hp_per_atom + TS_per_atom if X_{secondary_species} > 0, else just G_hp_per_atom.
                if df_isobar.loc[idx, f"X_{secondary_species}"] > 0:
                    df_isobar.loc[idx, "G_hp_per_atom_w_TS"] = df_isobar.loc[idx, "G_hp_per_atom"] + df_isobar.loc[idx, "TS_per_atom"]
                    df_isobar.loc[idx, "G_hp_per_species_w_TS"] = df_isobar.loc[idx, "G_hp_per_species"] + df_isobar.loc[idx, "TS_per_species"]
                else:
                    df_isobar.loc[idx, "G_hp_per_atom_w_TS"] = df_isobar.loc[idx, "G_hp_per_atom"]
                    df_isobar.loc[idx, "G_hp_per_species_w_TS"] = df_isobar.loc[idx, "G_hp_per_species"]  # no TS term if X_{secondary_species} = 0
                df_isobar.loc[idx, "G_hp_per_atom_w_TS_error"] = df_isobar.loc[idx, "G_hp_per_atom_error"]  # assuming no error in TS term
                df_isobar.loc[idx, "G_hp_per_species_w_TS_error"] = df_isobar.loc[idx, "G_hp_per_species_error"]  # assuming no error in TS term



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
            # y = sub["G_hp_per_atom_w_TS"].values
            # y_error = sub["G_hp_per_atom_w_TS_error"].values
            y = sub["G_hp_per_species_w_TS"].values  # use G_hp_per_species_w_TS for fitting
            y_error = sub["G_hp_per_species_w_TS_error"].values  # 1 sigma error in G_hp_per_species_w_TS
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
                    # print(f"counter_err_update: {counter_err_update} for {phase} at {pt}: slope = {slope:.3f} \\pm {slope_error:.3f}, intercept = {intercept:.3f} \\pm {intercept_error:.3f}\n")
                    # print(f"Values used: slope_TI = {slope_TI:.3f}, intercept_TI = {intercept_TI:.3f}, slope_error_TI = {slope_error_TI:.3f}, intercept_error_TI = {intercept_error_TI:.3f}\n")
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






    #############################################################
    #############################################################
    # add df to df_superset
    df_superset = pd.concat([df_superset, df], ignore_index=True)

    # narrow down to Config_folder with *_8H*
    df_superset = df_superset[df_superset["Config_folder"].str.contains("_8H")]


    # sort df_superset by Phase, Config_folder and Target temperature (K)
    df_superset.sort_values(["Phase", "Config_folder", "Target temperature (K)"], inplace=True)
    #############################################################
    #############################################################





    # df_superset -- create df_superset["array_X_{secondary_species}"] -- log varying X from 1e-10 to 100/109 (corresponds to a wt fraction of 0.1)
    # 1) build your “X” array once
    # arr_X = np.logspace(np.log10(1e-10), np.log10(100/109), num=num_array_X)
    # arr_inf = np.nan * np.ones_like(arr_X)  # create an array of NaN with the same shape as arr_X

    # 2) define which prefixes get the full `arr` and which get `np.nan`
    # array_cols = {
    #     # these four get the log‐spaced array
    #     **{f"array_X_{secondary_species}{suffix}": arr_X
    #     for suffix in ["", "_error", "_lower", "_upper"]},
    #     # these four groups get initialized to NaN
    #     **{f"{prefix}_{secondary_species}{suffix}": arr_inf
    #     for prefix in ["array_Xw", "array_KD", "array_D_wt", "array_KD_prime"]
    #     for suffix in ["", "_error", "_lower", "_upper"]}

    #     **{f"array_X_{secondary_species}{suffix}": arr_PT
    #     for suffix in ["_T", "_P", ]}
    # }
    # array_Xw_{secondary_species} -- Weight fraction corresponding to array_X_{secondary_species} -- exact
    # array_KD: corresponding KD -- exact
    # array_D_wt: corresponding D_wt -- exact
    # KD_prime: array_KD_prime_{secondary_species} where KD_prime represents the KD for <<H2 (silicate) = H (metal)>> reaction (rather than the default <<H (silicate) = H (metal)>> reaction) -- exact
    
    # 3) loop once to add them all, broadcasting each fill to every row
    for col, fill in array_cols.items():
        # [fill] * len(df) creates a list of length df with the same object/value
        # object-dtype is inferred automatically when the fill is an array
        df_superset[col] = pd.Series([fill] * len(df_superset), index=df_superset.index, dtype="object")
        print(f"Added column {col} to df_superset with shape {df_superset[col].shape}")

    print("")
    print("")
    print("")





    # KD_exact
    for i, row in df_superset.iterrows():

        phase = row["Phase"]
        pt    = row["P_T_folder"]

        # 2) Determine the other phase
        other_phase = f"MgSiO3_{secondary_species}" if phase == f"Fe_{secondary_species}" else f"Fe_{secondary_species}"

        
        # 3) Grab that phase’s mu_excess_{secondary_species} for the same P_T_folder and same temperature
        mask = (
                (df_superset["Phase"] == other_phase) &
                (df_superset["P_T_folder"] == pt) &
                (df_superset["Target temperature (K)"] == row["Target temperature (K)"])
                )
        mask_idx = df_superset.index[mask][0]

        # check if mask exists, if not, raise error
        if not mask.any():
            raise ValueError(f"No matching phase found for {other_phase} at {pt} and temperature {row['Target temperature (K)']} K.")


        # run if this phase is "MgSiO3_{secondary_species}", skip otherwise
        if phase != f"MgSiO3_{secondary_species}":
            print(f"Skipping {phase} at {pt} as it is not MgSiO3_{secondary_species}.")
            continue

        partner__mu_excess = df_superset.loc[mask, f"mu_excess_{secondary_species}"]
        partner__mu_excess_error = df_superset.loc[mask, f"mu_excess_{secondary_species}_error"]

        # Determine the multiplier for the exponent based on phase
        if phase == f"Fe_{secondary_species}":
            mult_factor = 1 # to ensure that KD is always for {secondary_species}_{silicate} -> {secondary_species}_{metal}
        else:
            mult_factor = -1

        if partner__mu_excess.empty or partner__mu_excess_error.empty:
            raise ValueError(f"No mu_excess_{secondary_species} found for {other_phase} at {pt} and temperature {row['Target temperature (K)']} K.")
        other__mu_excess = partner__mu_excess.iloc[0]
        other__mu_excess_error = partner__mu_excess_error.iloc[0]
        mu_excess = row[f"mu_excess_{secondary_species}"]
        mu_excess_error = row[f"mu_excess_{secondary_species}_error"]
        
        ############
        ############
        slope_MgSiO3 = row["slope"]
        slope_MgSiO3_error = row["slope_error"]
        intercept_MgSiO3 = row["intercept"]
        intercept_MgSiO3_error = row["intercept_error"]
        slope_Fe = df_superset.at[mask_idx, "slope"]
        slope_Fe_error = df_superset.at[mask_idx, "slope_error"]
        intercept_Fe = df_superset.at[mask_idx, "intercept"]
        intercept_Fe_error = df_superset.at[mask_idx, "intercept_error"]
        ############
        ############
        
        if mu_excess is None or np.isnan(mu_excess) or other__mu_excess is None or np.isnan(other__mu_excess):
            raise ValueError(f"mu_excess_{secondary_species} or other mu_excess_{secondary_species} is NaN for {phase} at {pt} and temperature {row['Target temperature (K)']} K.")

        # 4) Get the temperature (in K)
        T = row["Target temperature (K)"]
        if np.isnan(T) or T <= 0:
            raise ValueError(f"Invalid temperature {T} K for {phase} at {pt}. Must be positive and non-zero.")

        # 5) Compute KD
        # solve for KD = (x/y) such that (x/y) = (1/(5-4*y)) * np.exp(-mult_factor*(row[f"mu_excess_{secondary_species}"] - other__mu_excess) / (kB * T))
        # return (1/5.0) * np.exp(-mult_factor*(row[f"mu_excess_{secondary_species}"] - other__mu_excess) / (kB * T)) # assuming y is ~ 0
        fn_KD__part = lambda mu_excess, other__mu_excess, T, mult_factor: np.exp(-mult_factor * (mu_excess - other__mu_excess) / (kB * T))
        KD__part, KD__part_error, KD__part_lower, KD__part_upper = monte_carlo_error_asymmetric(
            fn_KD__part,
            [mu_excess, other__mu_excess, T, mult_factor],
            [mu_excess-mu_excess_error, other__mu_excess-other__mu_excess_error, T-0.0, mult_factor-0.0],
            [mu_excess+mu_excess_error, other__mu_excess+other__mu_excess_error, T+0.0, mult_factor+0.0],
        )
        
                # fn_estimate__X_in_Fe # i.e. estimating array_X_{secondary_species} (Fe)
        if phase == f"MgSiO3_{secondary_species}":
            array_X_in_MgSiO3 = row[f"array_X_{secondary_species}"]
        else:
            array_X_in_MgSiO3 = df_superset.loc[mask, f"array_X_{secondary_species}"].iloc[0]

        # fn_estimate__X_in_Fe = lambda KD__part, X_in_MgSiO3: (X_in_MgSiO3/(5 - 4*X_in_MgSiO3)) * KD__part
        # fn_estimate__X_in_Fe = lambda mu_excess, other__mu_excess, T, mult_factor, X_in_MgSiO3: (X_in_MgSiO3/(5 - 4*X_in_MgSiO3)) * np.exp(-mult_factor * (mu_excess - other__mu_excess) / (kB * T))
        # array_X_in_Fe = (array_X_in_MgSiO3/(5 - 4*array_X_in_MgSiO3)) * KD__part

        # if phase == f"MgSiO3_{secondary_species}":
        #     df_superset.loc[mask, f"array_X_{secondary_species}"] = array_X_in_Fe
        #     # row__array_X_in_MgSiO3 = np.nan * np.ones_like(array_X_in_Fe)  # fill with NaN -- i.e., if Fe_{secondary_species} phase, then we do not have X_in_MgSiO3
        # else:
        #     # row__array_X_in_MgSiO3 = array_X_in_MgSiO3
        #     df_superset.at[i, f"array_X_{secondary_species}"] = array_X_in_Fe

        fn_KD = lambda mu_excess, other__mu_excess, T, mult_factor, X_in_MgSiO3: ( np.exp(-mult_factor * (mu_excess - other__mu_excess) / (kB * T)) ) / (5 - 4*X_in_MgSiO3)
        array_KD = np.nan * np.ones_like(array_X_in_MgSiO3)  # initialize with NaN
        array_KD_error = np.nan * np.ones_like(array_X_in_MgSiO3)  # initialize with NaN
        array_KD_lower = np.nan * np.ones_like(array_X_in_MgSiO3)
        array_KD_upper = np.nan * np.ones_like(array_X_in_MgSiO3)
        array_X_in_Fe = np.nan * np.ones_like(array_X_in_MgSiO3)  # initialize with NaN
        array_X_in_Fe_error = np.nan * np.ones_like(array_X_in_MgSiO3)  # initialize with NaN
        array_X_in_Fe_lower = np.nan * np.ones_like(array_X_in_MgSiO3)  # initialize with NaN
        array_X_in_Fe_upper = np.nan * np.ones_like(array_X_in_MgSiO3)
        array_Xw_in_Fe = np.nan * np.ones_like(array_X_in_MgSiO3)  # initialize with NaN
        array_Xw_in_Fe_error = np.nan * np.ones_like(array_X_in_MgSiO3)  # initialize with NaN
        array_Xw_in_Fe_lower = np.nan * np.ones_like(array_X_in_MgSiO3)  # initialize with NaN
        array_Xw_in_Fe_upper = np.nan * np.ones_like(array_X_in_MgSiO3)

        ################
        # Create arrays for T and P to match the shape of array_X_in_MgSiO3
        array_T = np.full_like(array_X_in_MgSiO3, T)  # create an array of the same shape as array_X_in_MgSiO3 filled with T
        array_P = np.full_like(array_X_in_MgSiO3, row["Target pressure (GPa)"])  # create an array of the same shape as array_X_in_M
        ################

        # # KD, KD_error = monte_carlo_error(fn_KD, [mu_excess, other__mu_excess, T, mult_factor], [mu_excess_error, other__mu_excess_error, 0.0, 0.0])
        # for j, X_in_MgSiO3 in enumerate(array_X_in_MgSiO3):
            
        #     KD, KD_error, KD_lower, KD_upper = monte_carlo_error_asymmetric_w_io_bounds(
        #     # KD, KD_error, KD_lower, KD_upper = monte_carlo_error_asymmetric_w_bounds(
        #     # KD, KD_error, KD_lower, KD_upper = monte_carlo_error_asymmetric(
        #         fn_KD,
        #         [mu_excess, other__mu_excess, T, mult_factor, X_in_MgSiO3],
        #         [mu_excess-mu_excess_error, other__mu_excess-other__mu_excess_error, T-0.0, mult_factor-0.0, X_in_MgSiO3],
        #         [mu_excess+mu_excess_error, other__mu_excess+other__mu_excess_error, T+0.0, mult_factor+0.0, X_in_MgSiO3],
        #         # input_bounds=[(0, 1e40), (0, 1)],  # X_in_MgSiO3 should be between 0 and 1
        #         # output_bounds=[(0, 1e40)]  # KD should be between 0 and 1e40
        #     )
        #     array_KD[j] = KD
        #     array_KD_error[j] = KD_error
        #     array_KD_lower[j] = KD_lower
        #     array_KD_upper[j] = KD_upper

        #     # X_in_Fe, X_in_Fe_error, X_in_Fe_lower, X_in_Fe_upper = monte_carlo_error_asymmetric_w_bounds(
        #     #     fn_estimate__X_in_Fe,
        #     #     [KD__part, X_in_MgSiO3],
        #     #     [KD__part_lower, X_in_MgSiO3*(1-epsilon)],
        #     #     [KD__part_upper, X_in_MgSiO3*(1+epsilon)],
        #     #     # output_bounds=(0, 1)
        #     # )
        #     # array_X_in_Fe[j] = X_in_Fe
        #     # array_X_in_Fe_error[j] = X_in_Fe_error
        #     # array_X_in_Fe_lower[j] = X_in_Fe_lower
        #     # array_X_in_Fe_upper[j] = X_in_Fe_upper

        # # KD same for both phases, so we can just assign it to the df_superset
        # df_superset.at[i, f"array_KD_{secondary_species}"] = array_KD
        # df_superset.at[i, f"array_KD_{secondary_species}_error"] = array_KD_error
        # df_superset.at[i, f"array_KD_{secondary_species}_lower"] = array_KD_lower
        # df_superset.at[i, f"array_KD_{secondary_species}_upper"] = array_KD_upper
        # # print df_superset.loc[mask, f"array_KD_{secondary_species}"]
        # # print(f"Before assigning, df_superset.loc[mask, array_KD_secondary_species]:\n{df_superset.loc[mask, f'array_KD_{secondary_species}']}\n")
        # # print(f"Shape: {df_superset.loc[mask, f'array_KD_{secondary_species}'].iloc[0].shape}\n")
        # # print(f"Shape: {df_superset.at[i, f'array_KD_{secondary_species}'].shape}\n")
        # df_superset.at[mask_idx, f"array_KD_{secondary_species}"] = array_KD
        # df_superset.at[mask_idx, f"array_KD_{secondary_species}_error"] = array_KD_error
        # df_superset.at[mask_idx, f"array_KD_{secondary_species}_lower"] = array_KD_lower
        # df_superset.at[mask_idx, f"array_KD_{secondary_species}_upper"] = array_KD_upper

        # if phase == f"MgSiO3_{secondary_species}":
        #     df_superset.at[mask_idx, f"array_X_{secondary_species}"] = array_X_in_Fe
        #     df_superset.at[mask_idx, f"array_X_{secondary_species}_error"] = array_X_in_Fe_error
        #     df_superset.at[mask_idx, f"array_X_{secondary_species}_lower"] = array_X_in_Fe_lower
        #     df_superset.at[mask_idx, f"array_X_{secondary_species}_upper"] = array_X_in_Fe_upper
        # else:
        #     df_superset.at[i, f"array_X_{secondary_species}"] = array_X_in_Fe
        #     df_superset.at[i, f"array_X_{secondary_species}_error"] = array_X_in_Fe_error
        #     df_superset.at[i, f"array_X_{secondary_species}_lower"] = array_X_in_Fe_lower
        #     df_superset.at[i, f"array_X_{secondary_species}_upper"] = array_X_in_Fe_upper










        mu_primary = mu_MgSiO3 # "phase" is always MgSiO3_{secondary_species}
        mu_primary_other = mu_Fe

        if secondary_species == "He":
            mu_secondary = mu_He
        else:
            mu_secondary = mu_H

        # Calculate weight fractions Xw + D_wt
        # Note: phase is always MgSiO3_{secondary_species}, so other is always Fe_{secondary_species}
        def fn_X_to_Xw(X):
            # X array or scalar
            r = mu_primary / mu_secondary
            X_w = X / (X + ((mu_primary/mu_secondary)*(1-X))) 
            return X_w

        def fn_Xw_to_D_wt(X_in_MgSiO3, mu_excess, other__mu_excess, T, mult_factor):
            X_in_Fe = X_in_MgSiO3 * ( np.exp(-mult_factor * (mu_excess - other__mu_excess) / (kB * T)) ) / (5 - 4*X_in_MgSiO3)
            Xw_in_MgSiO3 = X_in_MgSiO3 / ( X_in_MgSiO3 + ((mu_MgSiO3/mu_secondary)*(1-X_in_MgSiO3)) )  # Xw = X / (1 + X)
            Xw_in_Fe = X_in_Fe / ( X_in_Fe + ((mu_Fe/mu_secondary)*(1-X_in_Fe)) )  # Xw = X / (1 + X)
            D_wt = Xw_in_Fe / Xw_in_MgSiO3  # D_wt = Xw_in_Fe / Xw_in_MgSiO3
            # print(f"fn_Xw_to_D_wt: X_in_MgSiO3: {X_in_MgSiO3}, KD: {KD}, X_in_Fe: {X_in_Fe}, Xw_in_MgSiO3: {Xw_in_MgSiO3}, Xw_in_Fe: {Xw_in_Fe}, D_wt: {D_wt}")
            return D_wt
        # def fn_Xw_to_D_wt(X_in_MgSiO3, KD):
        #     X_in_Fe = X_in_MgSiO3 * KD
        #     Xw_in_MgSiO3 = X_in_MgSiO3 / ( X_in_MgSiO3 + ((mu_MgSiO3/mu_secondary)*(1-X_in_MgSiO3)) )  # Xw = X / (1 + X)
        #     Xw_in_Fe = X_in_Fe / ( X_in_Fe + ((mu_Fe/mu_secondary)*(1-X_in_Fe)) )  # Xw = X / (1 + X)
        #     D_wt = Xw_in_Fe / Xw_in_MgSiO3  # D_wt = Xw_in_Fe / Xw_in_MgSiO3
        #     # print(f"fn_Xw_to_D_wt: X_in_MgSiO3: {X_in_MgSiO3}, KD: {KD}, X_in_Fe: {X_in_Fe}, Xw_in_MgSiO3: {Xw_in_MgSiO3}, Xw_in_Fe: {Xw_in_Fe}, D_wt: {D_wt}")
        #     return D_wt

        def fn_KD_prime(X_in_MgSiO3, mu_excess, other__mu_excess, T, mult_factor):
            X_in_Fe = X_in_MgSiO3 * ( np.exp(-mult_factor * (mu_excess - other__mu_excess) / (kB * T)) ) / (5 - 4*X_in_MgSiO3)
            Xw_in_MgSiO3 = X_in_MgSiO3 / ( X_in_MgSiO3 + ((mu_MgSiO3/mu_secondary)*(1-X_in_MgSiO3)) )  # Xw = X / (1 + X)
            Xw_in_Fe = X_in_Fe / ( X_in_Fe + ((mu_Fe/mu_secondary)*(1-X_in_Fe)) )  # Xw = X / (1 + X)
            KD_prime = (1 + ( ((1/Xw_in_MgSiO3)-1) * (mu_H2/mu_MgSiO3) ) ) / ((1 + ( ((1/Xw_in_Fe)-1) * (mu_H/mu_Fe) ) )**2)
            # print(f"fn_KD_prime: X_in_MgSiO3: {X_in_MgSiO3}, KD: {KD}, X_in_Fe: {X_in_Fe}, Xw_in_MgSiO3: {Xw_in_MgSiO3}, Xw_in_Fe: {Xw_in_Fe}, KD_prime: {KD_prime}")
            return KD_prime

        # def fn_for_all(X_in_MgSiO3, mu_excess, other__mu_excess, T, mult_factor):
        #     X_in_Fe = X_in_MgSiO3 * ( np.exp(-mult_factor * (mu_excess - other__mu_excess) / (kB * T)) ) / (5 - 4*X_in_MgSiO3)
        #     Xw_in_MgSiO3 = X_in_MgSiO3 / ( X_in_MgSiO3 + ((mu_MgSiO3/mu_secondary)*(1-X_in_MgSiO3)) )  # Xw = X / (1 + X)
        #     Xw_in_Fe = X_in_Fe / ( X_in_Fe + ((mu_Fe/mu_secondary)*(1-X_in_Fe)) )  # Xw = X / (1 + X)
        #     D_wt = Xw_in_Fe / Xw_in_MgSiO3  # D_wt = Xw_in_Fe / Xw_in_MgSiO3
        #     KD_prime = (1 + ( ((1/Xw_in_MgSiO3)-1) * (mu_H2/mu_MgSiO3) ) ) / ((1 + ( ((1/Xw_in_Fe)-1) * (mu_H/mu_Fe) ) )**2)
            
        #     # concatenate the outputs into a single array
        #     stacked_output = np.stack([Xw_in_MgSiO3, D_wt, KD_prime], axis=1)
        #     return stacked_output


        # # if P50_T3500, skip
        # if pt == "P50_T3500":
        #     # print(f"Skipping {phase} at {pt} as it is P50_T3500.")
        #     continue

        def fn_for_all_v2(X_in_MgSiO3, slope_MgSiO3, intercept_MgSiO3, slope_Fe, intercept_Fe, T):
            mu_excess_MgSiO3 = slope_MgSiO3 + intercept_MgSiO3
            mu_excess_Fe = slope_Fe + intercept_Fe
            
            X_in_Fe = X_in_MgSiO3 * ( np.exp((mu_excess_MgSiO3 - mu_excess_Fe) / (kB * T)) ) / (5 - 4*X_in_MgSiO3)
            Xw_in_MgSiO3 = X_in_MgSiO3 / ( X_in_MgSiO3 + ((mu_MgSiO3/mu_secondary)*(1-X_in_MgSiO3)) )  # Xw = X / (1 + X)
            Xw_in_Fe = X_in_Fe / ( X_in_Fe + ((mu_Fe/mu_secondary)*(1-X_in_Fe)) )  # Xw = X / (1 + X)
            
            KD = ( np.exp((mu_excess_MgSiO3 - mu_excess_Fe) / (kB * T)) ) / (5 - 4*X_in_MgSiO3)
            D_wt = Xw_in_Fe / Xw_in_MgSiO3  # D_wt = Xw_in_Fe / Xw_in_MgSiO3
            KD_prime = (1 + ( ((1/Xw_in_MgSiO3)-1) * (mu_H2/mu_MgSiO3) ) ) / ((1 + ( ((1/Xw_in_Fe)-1) * (mu_H/mu_Fe) ) )**2)
            
            # print(f"X_in_MgSiO3: {X_in_MgSiO3}, mu_excess_MgSiO3: {mu_excess_MgSiO3}, mu_excess_Fe: {mu_excess_Fe}")
            # print(f"X_in_Fe: {X_in_Fe}, Xw_in_MgSiO3: {Xw_in_MgSiO3}, Xw_in_Fe: {Xw_in_Fe}")
            # print(f"KD:{KD}, D_wt: {D_wt}, KD_prime: {KD_prime}\n")
            # concatenate the outputs into a single array
            stacked_output = np.stack([Xw_in_MgSiO3, D_wt, KD_prime, KD, X_in_Fe, Xw_in_Fe], axis=1)
            return stacked_output


        array_Xw = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_Xw_error = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_Xw_lower = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_Xw_upper = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_Xw_other = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_Xw_other_error = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_Xw_other_lower = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_Xw_other_upper = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_D_wt = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_D_wt_error = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_D_wt_lower = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_D_wt_upper = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])
        array_KD_prime = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_KD_prime_error = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_KD_prime_lower = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        array_KD_prime_upper = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])

        # print info on sim
        print(f"Phase: {phase}, P_T_folder: {pt}, Temperature: {T} K")
        print(f"mu_excess: {mu_excess}, mu_excess_error: {mu_excess_error}")
        print(f"other__mu_excess: {other__mu_excess}, other__mu_excess_error: {other__mu_excess_error}\n")

        for j, X in enumerate(row[f"array_X_{secondary_species}"]):
            X_lower = row[f"array_X_{secondary_species}_lower"][j] 
            X_upper = row[f"array_X_{secondary_species}_upper"][j]
            KD = array_KD[j]
            KD_lower = array_KD_lower[j]
            KD_upper = array_KD_upper[j]
            # print(f"KD: {KD}, KD_lower: {KD_lower}, KD_upper: {KD_upper}, X: {X}, X_lower: {X_lower}, X_upper: {X_upper}")

            # X_other = df_superset.loc[mask, f"array_X_{secondary_species}"].iloc[0][j]  # X in the other phase
            # X_other_lower = df_superset.loc[mask, f"array_X_{secondary_species}_lower"].iloc[0][j]
            # X_other_upper = df_superset.loc[mask, f"array_X_{secondary_species}_upper"].iloc[0][j]
            
            

            # # output_all, output_all_error, output_all_lower, output_all_upper = monte_carlo_error_asymmetric(
            # # output_all, output_all_error, output_all_lower, output_all_upper = monte_carlo_error_asymmetric_w_io_bounds(
            # output_all, output_all_error, output_all_lower, output_all_upper = monte_carlo_error_asymmetric_w_io_bounds_vectorized_outputs(
            # # output_all, output_all_error, output_all_lower, output_all_upper = monte_carlo_error_asymmetric_w_bounds(
            #     fn_for_all,
            #     [X, mu_excess, other__mu_excess, T, mult_factor],
            #     [X_lower, mu_excess-mu_excess_error, other__mu_excess-other__mu_excess_error, T-0.0, mult_factor-0.0],
            #     [X_upper, mu_excess+mu_excess_error, other__mu_excess+other__mu_excess_error, T+0.0, mult_factor+0.0],
            #     output_bounds=(0, 1e20),
            #     # output_bounds=(0, 1e40),
            # )
            # print("")
            # len_individual_output = 1#len(X)  # 1000 or so
            # output_all = output_all_error = output_all_lower = output_all_upper = np.nan * np.ones(4)
            # output_all, output_all_error, output_all_lower, output_all_upper = monte_carlo_error_asymmetric_w_bounds(
            output_all, output_all_error, output_all_lower, output_all_upper, in_rate, out_rate, accepted_count_by_raw_input_draws = monte_carlo_error_asymmetric_w_io_bounds_vectorized_outputs(
                fn_for_all_v2,
                [X, slope_MgSiO3, intercept_MgSiO3, slope_Fe, intercept_Fe, T],
                [X_lower, slope_MgSiO3-slope_MgSiO3_error, intercept_MgSiO3 - intercept_MgSiO3_error, slope_Fe-slope_Fe_error, intercept_Fe-intercept_Fe_error, T-0.0],
                [X_upper, slope_MgSiO3+slope_MgSiO3_error, intercept_MgSiO3 + intercept_MgSiO3_error, slope_Fe+slope_Fe_error, intercept_Fe+intercept_Fe_error, T+0.0],
                output_bounds=(0, 1e20),
                return_rates=True,
                # output_bounds=(0, 1e40),
            )
            # print(f"Output shape: {output_all.shape}, Output error shape: {output_all_error.shape}, Output lower shape: {output_all_lower.shape}, Output upper shape: {output_all_upper.shape}")
            # print(f"Output all: {output_all}")
            Xw = output_all[0]  # Xw_in_MgSiO3
            D_wt = output_all[1]  # D_wt
            KD_prime = output_all[2]  # KD_prime
            KD = output_all[3]  # KD
            X_in_Fe = output_all[4]  # X_in_Fe
            Xw_in_Fe = output_all[5]  # Xw_in_Fe
            Xw_error = output_all_error[0]
            D_wt_error = output_all_error[1]
            KD_prime_error = output_all_error[2]
            KD_error = output_all_error[3]
            X_in_Fe_error = output_all_error[4]
            Xw_in_Fe_error = output_all_error[5]
            Xw_lower = output_all_lower[0]
            D_wt_lower = output_all_lower[1]
            KD_prime_lower = output_all_lower[2]
            KD_lower = output_all_lower[3]
            X_in_Fe_lower = output_all_lower[4]
            Xw_in_Fe_lower = output_all_lower[5]
            Xw_upper = output_all_upper[0]
            D_wt_upper = output_all_upper[1]
            KD_prime_upper = output_all_upper[2]
            KD_upper = output_all_upper[3]
            X_in_Fe_upper = output_all_upper[4]
            Xw_in_Fe_upper = output_all_upper[5]

            # print(f"X: {X}, X_lower: {X_lower}, X_upper: {X_upper}")
            # print(f"Xw: {Xw}, Xw_error: {Xw_error}, Xw_lower: {Xw_lower}, Xw_upper: {Xw_upper}")
            # print(f"in_rate: {in_rate}, out_rate: {out_rate}, accepted_count_by_raw_input_draws: {accepted_count_by_raw_input_draws}\n")

            # Xw, Xw_error, Xw_lower, Xw_upper = monte_carlo_error_asymmetric(
            # Xw, Xw_error, Xw_lower, Xw_upper = monte_carlo_error_asymmetric_w_bounds(
            # Xw, Xw_error, Xw_lower, Xw_upper = monte_carlo_error_asymmetric_w_io_bounds(
            #     fn_X_to_Xw,
            #     [X],
            #     [X_lower],
            #     [X_upper],
            #     # bounds=(0,1),
            #     # input_bounds=(0,1),
            #     output_bounds=(0,1),
            # )
            array_Xw[j] = Xw
            array_Xw_error[j] = Xw_error
            array_Xw_lower[j] = Xw_lower
            array_Xw_upper[j] = Xw_upper


            # D_wt = Xw_other / Xw, i.e., Xw_in_Fe / Xw_in_MgSiO3
            # D_wt, D_wt_error, D_wt_lower, D_wt_upper = monte_carlo_error_asymmetric_w_bounds(
            # D_wt, D_wt_error, D_wt_lower, D_wt_upper = monte_carlo_error_asymmetric_w_io_bounds(
            # # D_wt, D_wt_error, D_wt_lower, D_wt_upper = monte_carlo_error_asymmetric(
            #     fn_Xw_to_D_wt,
            #     [X, mu_excess, other__mu_excess, T, mult_factor],
            #     [X_lower, mu_excess-mu_excess_error, other__mu_excess-other__mu_excess_error, T-0.0, mult_factor-0.0],
            #     [X_upper, mu_excess+mu_excess_error, other__mu_excess+other__mu_excess_error, T+0.0, mult_factor+0.0],
            #     # bounds=[(0, 1), (None, None)],
            #     # input_bounds=[(0,1),(0, 1e40)],
            #     output_bounds=[(0, 1e40)],
            # )
            array_D_wt[j] = D_wt
            array_D_wt_error[j] = D_wt_error
            array_D_wt_lower[j] = D_wt_lower
            array_D_wt_upper[j] = D_wt_upper

            # KD_prime
            # KD_prime, KD_prime_error, KD_prime_lower, KD_prime_upper = monte_carlo_error_asymmetric_w_io_bounds(
            # # KD_prime, KD_prime_error, KD_prime_lower, KD_prime_upper = monte_carlo_error_asymmetric(
            #     fn_KD_prime,
            #     [X, mu_excess, other__mu_excess, T, mult_factor],
            #     [X_lower, mu_excess-mu_excess_error, other__mu_excess-other__mu_excess_error, T-0.0, mult_factor-0.0],
            #     [X_upper, mu_excess+mu_excess_error, other__mu_excess+other__mu_excess_error, T+0.0, mult_factor+0.0],
            #     output_bounds=[(0, 1e40)],
            # )
            array_KD_prime[j] = KD_prime
            array_KD_prime_error[j] = KD_prime_error
            array_KD_prime_lower[j] = KD_prime_lower
            array_KD_prime_upper[j] = KD_prime_upper

            # print(f"Row {i}, Phase: {phase}, P_T_folder: {pt}, Target temperature (K): {row['Target temperature (K)']}")
            # print(f"Main phase: X: {X}, X_lower: {X_lower}, X_upper: {X_upper}")
            # print(f"Main phase:  Xw: {Xw}, Xw_lower: {Xw_lower}, Xw_upper: {Xw_upper}")
            # print(f"D_wt: {D_wt}, D_wt_error: {D_wt_error}, D_wt_lower: {D_wt_lower}, D_wt_upper: {D_wt_upper}")
            # print(f"KD_prime: {KD_prime}, KD_prime_error: {KD_prime_error}, KD_prime_lower: {KD_prime_lower}, KD_prime_upper: {KD_prime_upper}\n\n")

            array_KD[j] = KD
            array_KD_error[j] = KD_error
            array_KD_lower[j] = KD_lower
            array_KD_upper[j] = KD_upper


            array_X_in_Fe[j] = X_in_Fe
            array_X_in_Fe_error[j] = X_in_Fe_error
            array_X_in_Fe_lower[j] = X_in_Fe_lower
            array_X_in_Fe_upper[j] = X_in_Fe_upper
            array_Xw_in_Fe[j] = Xw_in_Fe
            array_Xw_in_Fe_error[j] = Xw_in_Fe_error
            array_Xw_in_Fe_lower[j] = Xw_in_Fe_lower
            array_Xw_in_Fe_upper[j] = Xw_in_Fe_upper

            # exit(0)
        
        df_superset.at[i, f"array_Xw_{secondary_species}"] = array_Xw
        df_superset.at[i, f"array_Xw_{secondary_species}_error"] = array_Xw_error
        df_superset.at[i, f"array_Xw_{secondary_species}_lower"] = array_Xw_lower
        df_superset.at[i, f"array_Xw_{secondary_species}_upper"] = array_Xw_upper
        
        df_superset.at[mask_idx, f"array_X_{secondary_species}"] = array_X_in_Fe
        df_superset.at[mask_idx, f"array_X_{secondary_species}_error"] = array_X_in_Fe_error
        df_superset.at[mask_idx, f"array_X_{secondary_species}_lower"] = array_X_in_Fe_lower
        df_superset.at[mask_idx, f"array_X_{secondary_species}_upper"] = array_X_in_Fe_upper
        df_superset.at[mask_idx, f"array_Xw_{secondary_species}"] = array_Xw_in_Fe
        df_superset.at[mask_idx, f"array_Xw_{secondary_species}_error"] = array_Xw_in_Fe_error
        df_superset.at[mask_idx, f"array_Xw_{secondary_species}_lower"] = array_Xw_in_Fe_lower
        df_superset.at[mask_idx, f"array_Xw_{secondary_species}_upper"] = array_Xw_in_Fe_upper

        df_superset.at[i, f"array_X_{secondary_species}_in_Fe"] = array_X_in_Fe
        df_superset.at[i, f"array_X_{secondary_species}_in_Fe_error"] = array_X_in_Fe_error
        df_superset.at[i, f"array_X_{secondary_species}_in_Fe_lower"] = array_X_in_Fe_lower
        df_superset.at[i, f"array_X_{secondary_species}_in_Fe_upper"] = array_X_in_Fe_upper

        # # other goes to the other phase
        # df_superset.at[mask_idx, f"array_Xw_{secondary_species}"] = array_Xw_other
        # df_superset.at[mask_idx, f"array_Xw_{secondary_species}_error"] = array_Xw_other_error
        # df_superset.at[mask_idx, f"array_Xw_{secondary_species}_lower"] = array_Xw_other_lower
        # df_superset.at[mask_idx, f"array_Xw_{secondary_species}_upper"] = array_Xw_other_upper

        # D_wt same for both phases
        df_superset.at[i, f"array_D_wt_{secondary_species}"] = array_D_wt
        df_superset.at[i, f"array_D_wt_{secondary_species}_error"] = array_D_wt_error
        df_superset.at[i, f"array_D_wt_{secondary_species}_lower"] = array_D_wt_lower
        df_superset.at[i, f"array_D_wt_{secondary_species}_upper"] = array_D_wt_upper
        df_superset.at[mask_idx, f"array_D_wt_{secondary_species}"] = array_D_wt
        df_superset.at[mask_idx, f"array_D_wt_{secondary_species}_error"] = array_D_wt_error
        df_superset.at[mask_idx, f"array_D_wt_{secondary_species}_lower"] = array_D_wt_lower
        df_superset.at[mask_idx, f"array_D_wt_{secondary_species}_upper"] = array_D_wt_upper

        # KD_prime same for both phases
        df_superset.at[i, f"array_KD_prime_{secondary_species}"] = array_KD_prime
        df_superset.at[i, f"array_KD_prime_{secondary_species}_error"] = array_KD_prime_error
        df_superset.at[i, f"array_KD_prime_{secondary_species}_lower"] = array_KD_prime_lower
        df_superset.at[i, f"array_KD_prime_{secondary_species}_upper"] = array_KD_prime_upper
        df_superset.at[mask_idx, f"array_KD_prime_{secondary_species}"] = array_KD_prime
        df_superset.at[mask_idx, f"array_KD_prime_{secondary_species}_error"] = array_KD_prime_error
        df_superset.at[mask_idx, f"array_KD_prime_{secondary_species}_lower"] = array_KD_prime_lower
        df_superset.at[mask_idx, f"array_KD_prime_{secondary_species}_upper"] = array_KD_prime_upper

        df_superset.at[i, f"array_KD_{secondary_species}"] = array_KD
        df_superset.at[i, f"array_KD_{secondary_species}_error"] = array_KD_error
        df_superset.at[i, f"array_KD_{secondary_species}_lower"] = array_KD_lower
        df_superset.at[i, f"array_KD_{secondary_species}_upper"] = array_KD_upper
        df_superset.at[mask_idx, f"array_KD_{secondary_species}"] = array_KD
        df_superset.at[mask_idx, f"array_KD_{secondary_species}_error"] = array_KD_error
        df_superset.at[mask_idx, f"array_KD_{secondary_species}_lower"] = array_KD_lower
        df_superset.at[mask_idx, f"array_KD_{secondary_species}_upper"] = array_KD_upper


        # for P, T
        df_superset.at[i, f"array_P_{secondary_species}"] = array_P
        df_superset.at[i, f"array_T_{secondary_species}"] = array_T
        df_superset.at[mask_idx, f"array_P_{secondary_species}"] = array_P
        df_superset.at[mask_idx, f"array_T_{secondary_species}"] = array_T










    
    # for i, row in df_superset.iterrows():
    #     phase = row["Phase"]
    #     pt    = row["P_T_folder"]


        # # Calculate weight fractions Xw
        # fn_X_to_Xw = lambda X, mu_primary, mu_secondary: X / (X + (mu_primary/mu_secondary)*(1-X))  # Xw = X / (1 + X)
        # array_Xw = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        # array_Xw_error = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        # array_Xw_lower = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN
        # array_Xw_upper = np.nan * np.ones_like(df_superset[f"array_X_{secondary_species}"].iloc[0])  # initialize with NaN

        # for j, X in enumerate(row[f"array_X_{secondary_species}"]):
        #     X_lower = row[f"array_X_{secondary_species}_lower"][j] 
        #     X_upper = row[f"array_X_{secondary_species}_upper"][j]
            
        #     if phase == f"Fe_{secondary_species}":
        #         mu_primary = mu_Fe
        #     else:
        #         mu_primary = mu_MgSiO3

        #     if secondary_species == "He":
        #         mu_secondary = mu_He
        #     else:
        #         mu_secondary = mu_H
            
        #     Xw, Xw_error, Xw_lower, Xw_upper = monte_carlo_error_asymmetric(
        #         fn_X_to_Xw,
        #         [X, mu_primary, mu_secondary],
        #         [X_lower, mu_primary-0.0, mu_secondary-0.0],
        #         [X_upper, mu_primary+0.0, mu_secondary+0.0],
        #     )

        #     array_Xw[j] = Xw
        #     array_Xw_error[j] = Xw_error
        #     array_Xw_lower[j] = Xw_lower
        #     array_Xw_upper[j] = Xw_upper
        
        # df_superset.at[i, f"array_Xw_{secondary_species}"] = array_Xw
        # df_superset.at[i, f"array_Xw_{secondary_species}_error"] = array_Xw_error
        # df_superset.at[i, f"array_Xw_{secondary_species}_lower"] = array_Xw_lower
        # df_superset.at[i, f"array_Xw_{secondary_species}_upper"] = array_Xw_upper











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





    # df_superset -- remove all with temperatures TEMPERATURES_TO_REMOVE, e.g. 10400, 7200, 5200
    df_superset = df_superset[~df_superset["Target temperature (K)"].isin(TEMPERATURES_TO_REMOVE)].reset_index(drop=True)
    print(f"\n\nNOTE: Removed entries with T_target == {TEMPERATURES_TO_REMOVE}. Likely crystalline Fe at these T/Ps.\n\n")

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
        marker_opts_error,
        x_low=None,
        x_high=None
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
        x_low : float, optional
            Lower limit for x-axis based data
        x_high : float, optional
            Upper limit for x-axis based data
        """
        # print(f"\n\nMaster: x_min: {x_low}, x_max: {x_high}")
        for data, marker in zip(datasets, markers):
            # Scatter KD_sil_to_metal
            # var  = df[z_variable].values
            # print(f"x_min: {x_low}, x_max: {x_high}")
            x_vals = np.asarray(data[x_variable])
            temps  = np.array(data[z_variable], float)  
            KD = np.array(data["KD_sil_to_metal"], float)
            KD_low = np.array(data["KD_sil_to_metal_low"], float)
            KD_high = np.array(data["KD_sil_to_metal_high"], float)
            D_wt = np.array(data["D_wt"], float)
            D_wt_low = np.array(data["D_wt_low"], float)
            D_wt_high = np.array(data["D_wt_high"], float)

            if x_low is not None:
                mask = np.asarray(x_vals) > x_low
            if x_high is not None:
                mask = np.asarray(x_vals) < x_high

            if 'mask' in locals():
                x_vals = x_vals[mask]
                temps = temps[mask]
                KD = KD[mask]
                KD_low = KD_low[mask]
                KD_high = KD_high[mask]
                D_wt = D_wt[mask]
                D_wt_low = D_wt_low[mask]
                D_wt_high = D_wt_high[mask]

            colors = cmap(norm(temps))


            if x_vals.size > 0:

                fn_ax_KD.scatter(
                    x_vals,
                    KD,
                    color=colors,
                    **marker_opts_scatter,
                    marker=marker,
                    label=data["label"],
                    # rasterized=True
                )

                # print value of alpha from marker_opts_scatter
                # print(f"Alpha: {alpha} for dataset {data['label']} with marker {marker}")

                # Error bars for KD_sil_to_metal
                for x0, y0, y_low, y_high, color in zip(
                    x_vals,
                    KD,
                    KD_low,
                    KD_high,
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
                        ecolor=color
                    )


                # Scatter D_wt
                fn_ax_D_wt.scatter(
                    x_vals,
                    D_wt,
                    color=colors,
                    **marker_opts_scatter,
                    marker=marker,
                    # label=data["label"],
                    # rasterized=True
                )


                # Error bars for D_wt
                for x0, y0, y_low, y_high, color in zip(
                    x_vals,
                    D_wt,
                    D_wt_low,
                    D_wt_high,
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
                        ecolor=color
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



    def _extract_col(src, col):
        """Return 1-D float array for the given column from a DataFrame/Series/dict; None if missing."""
        try:
            if isinstance(src, pd.DataFrame):
                arr = src[col].to_numpy()
            elif isinstance(src, pd.Series):
                val = src[col]
                arr = np.array([val]) if np.isscalar(val) else np.asarray(val)
            else:  # dict-like / numpy structured / etc.
                val = src[col]
                arr = np.array([val]) if np.isscalar(val) else np.asarray(val)
        except (KeyError, AttributeError):
            raise KeyError(col)
        return np.asarray(arr, dtype=float).ravel()

    def _concat_cols(sources, col):
        parts = []
        for s in sources:
            a = _extract_col(s, col)
            if a is not None and a.size > 0:
                parts.append(a)
        return (np.concatenate(parts) if parts else np.array([], dtype=float))





    ############################################
    ############################################
    ############################################










    from ast import literal_eval

    _num_token_re = re.compile(
        r"""
        [+-]? (?:                                   # sign
            (?:\d+\.\d*|\.\d+|\d+)                  # number parts
            (?:[eE][+-]?\d+)?                       # optional exponent
            |                                       # OR
            inf(?:inity)?                           # inf / infinity
            | nan
        )
        """,
        re.VERBOSE | re.IGNORECASE
    )

    def parse_numpy_repr_list(s):
        """
        Parse strings like:
            "[1.0 2.5e-3  nan 4.]" (possibly multi-line)
        into a float64 NumPy array.

        Returns original object if it's already an array or not a bracketed string.
        """
        if isinstance(s, np.ndarray):
            return s
        if not isinstance(s, str):
            return s
        txt = s.strip()
        if len(txt) < 2 or txt[0] != '[' or txt[-1] != ']':
            return s  # not our pattern

        inner = txt[1:-1].strip()
        if not inner:
            return np.array([], dtype=float)

        # Replace any commas with spaces (just in case), collapse whitespace
        inner = inner.replace(',', ' ')
        # Find all numeric tokens
        tokens = _num_token_re.findall(inner)
        if not tokens:
            # fallback: try splitting on whitespace
            parts = inner.split()
        else:
            parts = tokens

        out = []
        for t in parts:
            tl = t.lower()
            if tl in ('nan',):
                out.append(np.nan)
            elif tl in ('inf', '+inf', 'infinity', '+infinity'):
                out.append(np.inf)
            elif tl in ('-inf', '-infinity'):
                out.append(-np.inf)
            else:
                try:
                    out.append(float(t))
                except ValueError:
                    # unexpected token → treat as NaN
                    out.append(np.nan)
        return np.array(out, dtype=float)













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

        Xiong et al. 2020 data -- original
        dict: data_He__Xiong_et_al_2020
        P:20, T:5000, D_wt: 5.8E-3, D_wt_low: (5.8-1.7)E-3, D_wt_high: (5.8+1.7)E-3
        P:60, T:5000, D_wt: 2.3E-2, D_wt_low: (2.3-1.1)E-2, D_wt_high: (2.3+1.1)E-2
        P:135, T:5000, D_wt: 1.4E-2, D_wt_low: (1.4-0.4)E-2, D_wt_high: (1.4+0.4)E-2

        Wang et al. 2022 (corrected) data -- no error known
        dict: data_He__Wang_et_al_2022_corrected
        P:20, T:2500, D_wt: 0.00003047649470506126, D_wt_low: 0.00003047649470506126, D_wt_high: 0.00003047649470506126
        P:40, T:3200, D_wt: 0.0012905866607492368, D_wt_low: 0.0012905866607492368, D_wt_high: 0.0012905866607492368
        P:60, T:3600, D_wt: 0.0019306977288832535, D_wt_low: 0.0019306977288832535, D_wt_high: 0.0019306977288832535
        P:135, T:5000, D_wt: 0.17224697497149574, D_wt_low: 0.17224697497149574, D_wt_high: 0.17224697497149574

        Wang et al. 2022 data -- original
        dict: data_He__Wang_et_al_2022
        P:10, T:2300, D_wt: 8.40E-4, D_wt_low: (8.40-3.90)E-4, D_wt_high: (8.40+3.90)E-4
        P:20, T:2500, D_wt: 9.11E-4, D_wt_low: (9.11-3.54)E-4, D_wt_high: (9.11+3.54)E-4
        P:20, T:4000, D_wt: 1.89E-2, D_wt_low: (1.89-0.63)E-2, D_wt_high: (1.89+0.63)E-2
        P:20, T:5000, D_wt: 4.92E-2, D_wt_low: (4.92-1.57)E-2, D_wt_high: (4.92+1.57)E-2
        P:40, T:3200, D_wt: 6.04E-3, D_wt_low: (6.04-2.33)E-3, D_wt_high: (6.04+2.33)E-3
        P:60, T:3600, D_wt: 7.26E-3, D_wt_low: (7.26-2.68)E-3, D_wt_high: (7.26+2.68)E-3
        P:80, T:4000, D_wt: 1.72E-2, D_wt_low: (1.72-0.70)E-2, D_wt_high: (1.72+0.70)E-2
        P:110, T:4500, D_wt: 2.69E-2, D_wt_low: (2.69-1.08)E-2, D_wt_high: (2.69+1.08)E-2
        P:135, T:5000, D_wt: 6.87E-2, D_wt_low: (6.87-2.42)E-2, D_wt_high: (6.87+2.42)E-2

        Yuan & Steinle-Neumann 2021 data
        dict: data_He__Yuan_and_Steinle_Neumann_2021
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
            "label": "Li et al. 2022"
        }

        data_He__Zhang_and_Yin_2012 = {
            "Target pressure (GPa)": [40],
            "Target temperature (K)": [3200],
            "D_wt": [0.008921488454390353],
            "D_wt_low": [0.0030070047268396694],
            "D_wt_high": [0.015060610109031605],
            "label": r"Zhang \& Yin 2012"
        }

        data_He__Xiong_et_al_2020_corrected = {
            "Target pressure (GPa)": [20, 60, 135],
            "Target temperature (K)": [5000, 5000, 5000],
            "D_wt": [0.00003439086890726415, 0.00161063482282789, 0.0024094848405521382],
            "D_wt_low": [0.00003439086890726415, 0.00161063482282789, 0.0024094848405521382],
            "D_wt_high": [0.00003439086890726415, 0.00161063482282789, 0.0024094848405521382],
            "label": "Xiong et al. 2020 (corrected)"
        }

        data_He__Xiong_et_al_2020 = {
            "Target pressure (GPa)": [20, 60, 135],
            "Target temperature (K)": [5000, 5000, 5000],
            "D_wt": [5.8E-3, 2.3E-2, 1.4E-2],
            "D_wt_low": [(5.8-1.7)*1E-3, (2.3-1.1)*1E-2, (1.4-0.4)*1E-2],
            "D_wt_high": [(5.8+1.7)*1E-3, (2.3+1.1)*1E-2, (1.4+0.4)*1E-2],
            "label": "Xiong et al. 2020"
        }

        data_He__Wang_et_al_2022_corrected = {
            "Target pressure (GPa)": [20, 40, 60, 135],
            "Target temperature (K)": [2500, 3200, 3600, 5000],
            "D_wt": [0.00003047649470506126, 0.0012905866607492368, 0.0019306977288832535, 0.17224697497149574],
            "D_wt_low": [0.00003047649470506126, 0.0012905866607492368, 0.0019306977288832535, 0.17224697497149574],
            "D_wt_high": [0.00003047649470506126, 0.0012905866607492368, 0.0019306977288832535, 0.17224697497149574],
            "label": "Wang et al. 2022 (corrected)"
        }

        data_He__Wang_et_al_2022 = {
            "Target pressure (GPa)": [10,   20,   20,   20,   40,   60,   80,   110,   135],
            "Target temperature (K)": [2300, 2500, 4000, 5000, 3200, 3600, 4000, 4500, 5000],
            "D_wt":        [8.40e-4,        9.11e-4,        1.89e-2,        4.92e-2,        6.04e-3,        7.26e-3,        1.72e-2,        2.69e-2,        6.87e-2],
            "D_wt_low":    [(8.40-3.90)*1e-4, (9.11-3.54)*1e-4, (1.89-0.63)*1e-2, (4.92-1.57)*1e-2, (6.04-2.33)*1e-3, (7.26-2.68)*1e-3, (1.72-0.70)*1e-2, (2.69-1.08)*1e-2, (6.87-2.42)*1e-2],
            "D_wt_high":   [(8.40+3.90)*1e-4, (9.11+3.54)*1e-4, (1.89+0.63)*1e-2, (4.92+1.57)*1e-2, (6.04+2.33)*1e-3, (7.26+2.68)*1e-3, (1.72+0.70)*1e-2, (2.69+1.08)*1e-2, (6.87+2.42)*1e-2],
            "label": "Wang et al. 2022"
        }

        data_He__Yuan_and_Steinle_Neumann_2021 = {
            "Target pressure (GPa)": [10, 25, 40, 50, 80, 130],
            "Target temperature (K)": [3000, 3500, 3800, 4000, 4000, 5000],
            "D_wt": [10**(-4.73), 10**(-3.48), 10**(-3.32), 10**(-2.07), 10**(-2.59), 10**(-1.24)],
            "D_wt_low": [10**(-4.73-0.40), 10**(-3.48-0.37), 10**(-3.32-0.28), 10**(-2.07-0.29), 10**(-2.59-0.26), 10**(-1.24-0.20)],
            "D_wt_high": [10**(-4.73+0.40), 10**(-3.48+0.37), 10**(-3.32+0.28), 10**(-2.07+0.29), 10**(-2.59+0.26), 10**(-1.24+0.20)],
            "label": r"Yuan \& Steinle-Neumann 2021"
        }

        datasets_comp = [
                    data_He__Li_et_al_2022,
                    data_He__Zhang_and_Yin_2012,
                    # data_He__Xiong_et_al_2020_corrected,
                    data_He__Xiong_et_al_2020,
                    # data_He__Wang_et_al_2022_corrected,
                    data_He__Wang_et_al_2022,
                    # data_He__Yuan_and_Steinle_Neumann_2021
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

        Yuan & Steinle-Neumann 2020 data -- NOTE: use water reaction!
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
            "label": "Li et al. 2022"
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
                    # data_H__Yuan_and_Steinle_Neumann_2020
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
        P: 46, T: 3920, KD: 10**1.32, KD_low: 10**(1.32-0.06), KD_high: 10**(1.32+0.06), D_wt: 47, D_wt_low: 47-4, D_wt_high: 47+4
        P: 48, T: 3450, KD: 10**1.10, KD_low: 10**(1.10-0.06), KD_high: 10**(1.10+0.06), D_wt: 29, D_wt_low: 29-2, D_wt_high: 29+2
        P: 57, T: 3860, KD: 10**1.22, KD_low: 10**(1.22-0.05), KD_high: 10**(1.22+0.05), D_wt: 40, D_wt_low: 40-3, D_wt_high: 40+3
        P: 60, T: 4560, KD: 10**1.37, KD_low: 10**(1.37-0.05), KD_high: 10**(1.37+0.05), D_wt: 56, D_wt_low: 56-4, D_wt_high: 56+4
        P: 47, T: 4230, KD: 10**1.35, KD_low: 10**(1.35-0.06), KD_high: 10**(1.35+0.06), D_wt: 57, D_wt_low: 57-6, D_wt_high: 57+6
        P: 30, T: 3080, KD: 10**1.18, KD_low: 10**(1.18-0.06), KD_high: 10**(1.18+0.06), D_wt: 37, D_wt_low: 37-3, D_wt_high: 37+3

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

        Okuchi 1997 data (D estimated from plot in Tagawa et al. 1997)
        dict: data_H__Okuchi_1997
        P: 7.5, T: 1200+273.15, KD: exp(-3.6), KD_low: exp(-3.6-0.1), KD_high: exp(-3.6+0.1), 
        P: 7.5, T: 1200+273.15, KD: exp(-3.4), KD_low: exp(-3.4-0.0), KD_high: exp(-3.4+0.0)
        P: 7.5, T: 1300+273.15, KD: exp(-2.4), KD_low: exp(-2.4-0.6), KD_high: exp(-2.4+0.6)
        P: 7.5, T: 1400+273.15, KD: exp(-2.0), KD_low: exp(-2.0-0.0), KD_high: exp(-2.0+0.0)
        P: 7.5, T: 1500+273.15, KD: exp(-1.4), KD_low: exp(-1.4-0.2), KD_high: exp(-1.4+0.2)
        P: 7.5, T: 1500+273.15, KD: exp(-1.5), KD_low: exp(-1.5-0.3), KD_high: exp(-1.5+0.3)
        P: 7.5, T: 1500+273.15, KD: exp(-1.5), KD_low: exp(-1.5-0.2), KD_high: exp(-1.5+0.2)


        """

        data_H__Tagawa_et_al_2021 = {
            "Target pressure (GPa)": [46, 48, 57, 60, 47, 30],
            "Target temperature (K)": [3920, 3450, 3860, 4560, 4230, 3080],
            "KD_sil_to_metal": [10**1.32, 10**1.10, 10**1.22, 10**1.37, 10**1.35, 10**1.18],
            "KD_sil_to_metal_low": [10**(1.32-0.06), 10**(1.10-0.06), 10**(1.22-0.05), 10**(1.37-0.05), 10**(1.35-0.06), 10**(1.18-0.06)],
            "KD_sil_to_metal_high": [10**(1.32+0.06), 10**(1.10+0.06), 10**(1.22+0.05), 10**(1.37+0.05), 10**(1.35+0.06), 10**(1.18+0.06)],
            "D_wt": [47, 29, 40, 56, 57, 37],
            "D_wt_low": [47-4, 29-2, 40-3, 56-4, 57-6, 37-3],
            "D_wt_high": [47+4, 29+2, 40+3, 56+4, 57+6, 37+3],
            "label": "Tagawa et al. 2021"
        }


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
            "Target pressure (GPa)": [7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5],
            "Target temperature (K)": [1473.15, 1473.15, 1573.15, 1673.15, 1773.15, 1773.15, 1773.15],
            "KD_sil_to_metal": [np.exp(-3.6), np.exp(-3.4), np.exp(-2.4), np.exp(-2.0), np.exp(-1.4), np.exp(-1.5), np.exp(-1.5)],
            "KD_sil_to_metal_low": [np.exp(-3.6-0.1), np.exp(-3.4-0.0), np.exp(-2.4-0.6), np.exp(-2.0-0.0), np.exp(-1.4-0.2), np.exp(-1.5-0.3), np.exp(-1.5-0.2)],
            "KD_sil_to_metal_high": [np.exp(-3.6+0.1), np.exp(-3.4+0.0), np.exp(-2.4+0.6), np.exp(-2.0+0.0), np.exp(-1.4+0.2), np.exp(-1.5+0.3), np.exp(-1.5+0.2)],
            "label": "Okuchi 1997"
        }
        # calculate KD_sil_to_metal from D_wt (including error) using the formula:
        # KD_sil_to_metal = D_wt * (56/100)
        data_H__Okuchi_1997["D_wt"] = [d * (100/56) for d in data_H__Okuchi_1997["KD_sil_to_metal"]]
        data_H__Okuchi_1997["D_wt_low"] = [d * (100/56) for d in data_H__Okuchi_1997["KD_sil_to_metal_low"]]
        data_H__Okuchi_1997["D_wt_high"] = [d * (100/56) for d in data_H__Okuchi_1997["KD_sil_to_metal_high"]]

        datasets_expt = [
                    data_H__Tagawa_et_al_2021,
                    data_H__Clesi_et_al_2018,
                    data_H__Malavergne_et_al_2018,
                    data_H__Okuchi_1997
                                                            ]
        # print(f"WARNING: Some studies talk about H2, silicate to metal partitioning, and some H, silicate --- we do H also here, how to make sure nothing is getting messed up?")





    # make this pandas DataFrame
    # datasets_comp = [pd.DataFrame(data) for data in datasets_comp]
    # datasets_expt = [pd.DataFrame(data) for data in datasets_expt]





    # ********************************************************************************************************
    # ********************************************************************************************************
    # Computational data but from 2-phase simulations
    # if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
    if secondary_species == "He":
        """ 
        
        Two-phase simulations for He in Fe and MgSiO3:
        dict: data_He__two_phase_simulations__old
        P:500, T:9000, KD: 0.032, KD_low: 0.032, KD_high: 0.032
        P:1000, T:13000, KD: 0.32, KD_low: 0.32, KD_high: 0.32

        dict: data_He__two_phase_simulations__new
        P:250, T:6500, KD: 0.173, KD_low: 0.145, KD_high: 0.202, D_wt: 0.179, D_wt_low: 0.147, D_wt_high: 0.213

        """

        data_He__two_phase_simulations__old = {
            "Target pressure (GPa)": [500, 1000],
            "Target temperature (K)": [9000, 13000],
            "KD_sil_to_metal": [0.032, 0.32],
            "KD_sil_to_metal_low": [0.032, 0.32],
            "KD_sil_to_metal_high": [0.032, 0.32],
            "label": "This study (2P-AIMD)"
        }
        # D_wt = KD * (100/56)  # assuming Fe as metal, 56 g/mol
        data_He__two_phase_simulations__old["D_wt"] = [kd * (100/56) for kd in data_He__two_phase_simulations__old["KD_sil_to_metal"]]
        data_He__two_phase_simulations__old["D_wt_low"] = [kd * (100/56) for kd in data_He__two_phase_simulations__old["KD_sil_to_metal_low"]]
        data_He__two_phase_simulations__old["D_wt_high"] = [kd * (100/56) for kd in data_He__two_phase_simulations__old["KD_sil_to_metal_high"]]

        data_He__two_phase_simulations__new = {
            "Target pressure (GPa)": [250],
            "Target temperature (K)": [6500],
            "KD_sil_to_metal": [0.173],
            "KD_sil_to_metal_low": [0.145],
            "KD_sil_to_metal_high": [0.202],
            "D_wt": [0.179],
            "D_wt_low": [0.147],
            "D_wt_high": [0.213],
            "label": "This study (2P-AIMD)"
        }


        datasets_2phases = [data_He__two_phase_simulations__new]

    if secondary_species == "H":
        """ 
        
        Two-phase simulations for H in Fe and MgSiO3:
        dict: data_H__two_phase_simulations
        P:250, T:6500, KD = 1.434, KD_low = 1.239, KD_high = 1.674, D_wt = 3.729, D_wt_low = 2.731, D_wt_high = 5.124
                        X_H_in_MgSiO3 = 0.42190, X_H_in_MgSiO3_lower = 0.37334, X_H_in_MgSiO3_upper = 0.46500
                        X_H_in_Fe = 0.60387, X_H_in_Fe_lower = 0.54287, X_H_in_Fe_upper = 0.66338
        """

        data_H__two_phase_simulations = {
            "Target pressure (GPa)": [250],
            "Target temperature (K)": [6500],
            "KD_sil_to_metal": [1.434],
            "KD_sil_to_metal_low": [1.239],
            "KD_sil_to_metal_high": [1.674],
            "D_wt": [3.729],
            "D_wt_low": [2.731],
            "D_wt_high": [5.124],
            "label": "This study (2P-AIMD)"
        }


        datasets_2phases = [data_H__two_phase_simulations]

        ##########################












    # ********************************************************************************************************
    # ********************************************************************************************************
    # ********************************************************************************************************
    def best_fn(x0, x1, x2):# x0: log(X or X2), x1: P, x2: T

        if secondary_species == "He":
            fn = np.exp(x0) - (x2 * (0.07335293 / x1))
            fn = ((x2 * -0.06782242) + -44.47209) / x1 # loss: 0.12429151 -- one of the simplest w/o X dependence
            fn = ((x2 * -0.0678) + -44.5) / x1 # loss: 0.12429151 -- one of the simplest w/o X dependence -- simplified

        elif secondary_species == "H":
            if H_STOICH_MODE == 2:
                fn = (((x1 * -0.0048796246) + 6.5670815) / np.exp(np.exp(x0))) + x0 # loss: 1.2789929; one of the simplest w no T dependence
                fn = (((x1 * -0.00488) + 6.567) / np.exp(np.exp(x0))) + x0 # loss: 1.2789929; one of the simplest w no T dependence -- simplified
                # fn =  #loss: ??? ; one of the simplest w T dependence
                fn = ((6.7519736 - (0.0051717 * x1)) + x0) / ((x1 ** (1.6505091 + (50.947327 / x0))) + ((x2 ** 0.1455538) ** (np.log(x1) ** ((x0 + (0.6689549 ** x0)) * x0)))) #loss: 1.1131124; the most complex and lowest loss soln
            elif H_STOICH_MODE == 1:
                fn = ((x1 * -0.0033594905) + 3.5382779) ** ((0.3938717 ** (np.exp(x0) ** np.exp(((-43.5826 / (-1.6072097 - x0)) + (np.exp((12387.732 / x2) + x0) + 4.5243583)) / x1))) - -0.0033594905) # loss: 0.12907255

        return np.exp(fn) # {return KD for He} or {KD_prime for H}
    


    def best_fn_v2(x0, x1, x2, H_STOICH_MODE=2):# x0: log(X or X2), x1: P, x2: T

        if secondary_species == "He":
            fn = np.exp(x0) - (x2 * (0.07335293 / x1))
            fn = ((x2 * -0.06782242) + -44.47209) / x1 # loss: 0.12429151 -- one of the simplest w/o X dependence
            fn = ((x2 * -0.0678) + -44.5) / x1 # loss: 0.12429151 -- one of the simplest w/o X dependence -- simplified

        elif secondary_species == "H":
            if H_STOICH_MODE == 2:
                fn = (((x1 * -0.0048796246) + 6.5670815) / np.exp(np.exp(x0))) + x0 # loss: 1.2789929; one of the simplest w no T dependence
                fn = (((x1 * -0.00488) + 6.567) / np.exp(np.exp(x0))) + x0 # loss: 1.2789929; one of the simplest w no T dependence -- simplified
                # fn =  #loss: ??? ; one of the simplest w T dependence
                fn = ((6.7519736 - (0.0051717 * x1)) + x0) / ((x1 ** (1.6505091 + (50.947327 / x0))) + ((x2 ** 0.1455538) ** (np.log(x1) ** ((x0 + (0.6689549 ** x0)) * x0)))) #loss: 1.1131124; the most complex and lowest loss soln
            elif H_STOICH_MODE == 1:
                fn = ((x1 * -0.0033594905) + 3.5382779) ** ((0.3938717 ** (np.exp(x0) ** np.exp(((-43.5826 / (-1.6072097 - x0)) + (np.exp((12387.732 / x2) + x0) + 4.5243583)) / x1))) - -0.0033594905) # loss: 0.12907255

        return np.exp(fn) # {return KD for He} or {KD_prime for H}
    # ********************************************************************************************************
    # ********************************************************************************************************
    # ********************************************************************************************************












    ##############################################
    ##############################################
    ##############################################






    # plot X_{secondary_species} vs G_hp_per_atom_w_TS, and color by P_T_folder and size by phase
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import BoundaryNorm, ListedColormap

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
                # sub["G_hp_per_atom_w_TS"],
                sub["G_hp_per_species_w_TS"],
                color=point_colors,
                s=size_map[phase],
                alpha=alpha_map[phase],
                label=phase
            )

            # now loop to draw each errorbar with its own color
            for x0, y0, err0, c in zip(
                sub[f"X_{secondary_species}"],
                sub["G_hp_per_species_w_TS"],
                sub["G_hp_per_species_w_TS_error"],
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
        # ax.set_ylabel("G_hp_per_atom_w_TS (eV)")
        ax.set_ylabel("G_hp_per_species_w_TS (eV)")
        ax.set_title(f"X_{secondary_species} vs G_hp_per_species_w_TS\nColored by P_T_folder, Sized/Alpha by Phase")
        ax.legend(title="Phase")
        ax.grid(True)
        plt.tight_layout()

        # 10) Save and/or show
        plt.savefig(f"X_{secondary_species}_vs_G_hp_per_atom_w_TS__{secondary_species}.png", dpi=300)
        print(f"Plot saved as X_{secondary_species}_vs_G_hp_per_atom_w_TS__{secondary_species}.png")
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
        plt.savefig(f"X_{secondary_species}_vs_mu_{secondary_species}__{secondary_species}.png")


























    ######################################################################
    ######################################################################
    ######################################################################
    df_superset = pd.read_csv("all_TI_results_superset.csv")
    # narrow down to Config_folder with *_8H*
    df_superset = df_superset[df_superset["Config_folder"].str.contains("_8H")]

    # print df_superset[f'array_X_{secondary_species}'][0]
    # print(df_superset[f'array_X_{secondary_species}'][0])

    # convert all columns beginning with "array_" to pd.Series. These have been read as str: [1.0, 2.0, 3.0]
    array_cols = [c for c in df_superset.columns if c.startswith("array_")]
    for col in array_cols:
        df_superset[col] = df_superset[col].apply(parse_numpy_repr_list)

    # print shape of array_X_{secondary_species}[0]
    # print(f"Shape of array_X_{secondary_species}[0]: {df_superset[f'array_X_{secondary_species}'][0].shape}")



    # # X_array
    # # create array of size (len(df_superset),num_array_X)
    # X_array = np.zeros((len(df_superset), num_array_X))
    # T_array = np.zeros((len(df_superset), num_array_X))
    # P_array = np.zeros((len(df_superset), num_array_X))
    # KD_array = np.zeros((len(df_superset), num_array_X))
    # KD_prime_array = np.zeros((len(df_superset), num_array_X))
    # D_wt_array = np.zeros((len(df_superset), num_array_X))
    # print(f"X_array shape: {X_array.shape}, T_array shape: {T_array.shape}, P_array shape: {P_array.shape}")
    # for i, row in df_superset.iterrows():
    #     X_array[i,:] = row[f"array_X_{secondary_species}"]
    #     T_array[i,:] = row["Target temperature (K)"]
    #     P_array[i,:] = row["Target pressure (GPa)"]
    #     KD_array[i,:] = row[f"array_KD_{secondary_species}"]
    #     KD_prime_array[i,:] = row[f"array_KD_prime_{secondary_species}"]
    #     D_wt_array[i,:] = row[f"array_D_wt_{secondary_species}"]

    # # write these arrays to three as csv files
    # np.savetxt(f"X_array_{secondary_species}.csv", X_array, delimiter=",")
    # np.savetxt(f"T_array_{secondary_species}.csv", T_array, delimiter=",")
    # np.savetxt(f"P_array_{secondary_species}.csv", P_array, delimiter=",")






    # exit()

    df = df_superset.copy()


    # y_min__H = 10**(-3.5)
    # y_max__H = 10**(2.5)
    # y_min__He = 10**(-3.5)
    # y_max__He = 10**(2.5)
    y_min__H = 10**(-2.0)
    y_max__H = 10**(3.0)
    y_min__He = 10**(-4.5)
    y_max__He = 10**(0.5)


    # x lim ,  y lim
    if secondary_species == "He":
        y_min = y_min__He
        y_max = y_max__He
    elif secondary_species == "H":
        y_min = y_min__H
        y_max = y_max__H
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








    if PLOT_MODE == 3 or PLOT_MODE < 0: # KD_D_wt_vs_P_T.png

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














        # # if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
        # if secondary_species == "He":
        #     """ 
            
        #     Two-phase simulations for He in Fe and MgSiO3:
        #     dict: data_He__two_phase_simulations__old
        #     P:500, T:9000, KD: 0.032, KD_low: 0.032, KD_high: 0.032
        #     P:1000, T:13000, KD: 0.32, KD_low: 0.32, KD_high: 0.32

        #     dict: data_He__two_phase_simulations__new

        #     """

        #     data_He__two_phase_simulations__old = {
        #         "Target pressure (GPa)": [500, 1000],
        #         "Target temperature (K)": [9000, 13000],
        #         "KD": [0.032, 0.32],
        #         "KD_low": [0.032, 0.32],
        #         "KD_high": [0.032, 0.32]
        #     }
        #     # D_wt = KD * (100/56)  # assuming Fe as metal, 56 g/mol
        #     data_He__two_phase_simulations__old["D_wt"] = [kd * (100/56) for kd in data_He__two_phase_simulations["KD"]]
        #     data_He__two_phase_simulations__old["D_wt_low"] = [kd * (100/56) for kd in data_He__two_phase_simulations["KD_low"]]
        #     data_He__two_phase_simulations__old["D_wt_high"] = [kd * (100/56) for kd in data_He__two_phase_simulations["KD_high"]]

        #     data_He__two_phase_simulations__new = {
        #         "Target pressure (GPa)": [250],
        #         "Target temperature (K)": [6500],
        #         "KD": [0.173],
        #         "KD_low": [0.145],
        #         "KD_high": [0.202],
        #         "D_wt": [0.179],
        #         "D_wt_low": [0.147],
        #         "D_wt_high": [0.213]
        #     }
            # ax_KD.scatter(data_He__two_phase_simulations[x_variable], 
            #                 data_He__two_phase_simulations["KD"],
            #                 **marker_opts_scatter,
            #                 marker=marker_2phase,
            #                 c=data_He__two_phase_simulations[z_variable],
            #                 cmap=cmap,
            #                 norm=norm,
            #                 label="This study (2P)")
            # ax_D_wt.scatter(data_He__two_phase_simulations[x_variable], 
            #                 data_He__two_phase_simulations["D_wt"],
            #                 **marker_opts_scatter,
            #                 marker=marker_2phase,
            #                 c=data_He__two_phase_simulations[z_variable],
            #                 cmap=cmap,
            #                 norm=norm,
            #                 label="This study (2P)")











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








        # ***************************
        # ***************************
        if FIT_MODE == 1:
            # combine T and P from datasets_expt, datasets_comp and df dataframe
            sources = list(datasets_expt) + list(datasets_comp) + [df]  # uncomment if needed
            # array_T = row[f"array_T_{secondary_species}"] # z_variable -- T
            # array_P = row[f"array_P_{secondary_species}"] # x_variable -- P
            array_T          = _concat_cols(sources, "Target temperature (K)")
            array_P          = _concat_cols(sources, "Target pressure (GPa)")

            if secondary_species == "He":
                # array_x_axis = array_X
                array_X = (array_T ** 0.) * 1e-3
                secondary_species_label = "He"

            elif secondary_species == "H":
                # array_x_axis = array_X2
                array_X2 = array_X = (array_T ** 0.) * 1e-3
                secondary_species_label = "H2"

            array_x_axis = array_P

            x0 = np.log(array_X)
            x1 = array_P
            x2 = array_T

            if secondary_species == "H":
                # plot the best fit line
                y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
            elif secondary_species == "He":
                y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
            # print(f"Best fit line for KD vs P, T: {y_fit}")
            # print(f"logX: {x0}")
            # print(f"T: {array_T}")
            # print(f"P: {array_P}")
            ax_KD.plot(
                array_x_axis, y_fit,
                linestyle='',
                marker="s",
                label=f"Best fit model",
                color='black', markersize=10,
                alpha=0.15
            )
        # ***************************
        # ***************************





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
        plt.savefig(f"KD_D_wt_vs_P_T__{secondary_species}.png", dpi=300)























    if PLOT_MODE == 4 or PLOT_MODE < 0: # KD_D_wt_vs_P_T__T__{secondary_species}.png; KD_sil_to_metal_vs_P_T_panels

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




        # ***************************
        # ***************************
        if FIT_MODE == 1:
            # combine T and P from datasets_expt, datasets_comp and df dataframe
            sources = list(datasets_expt) + list(datasets_comp) + [df]  # uncomment if needed
            # array_T = row[f"array_T_{secondary_species}"] # z_variable -- T
            # array_P = row[f"array_P_{secondary_species}"] # x_variable -- P
            array_T          = _concat_cols(sources, "Target temperature (K)")
            array_P          = _concat_cols(sources, "Target pressure (GPa)")

            if secondary_species == "He":
                # array_x_axis = array_X
                array_X = (array_T ** 0.) * 1e-3
                secondary_species_label = "He"

            elif secondary_species == "H":
                # array_x_axis = array_X2
                array_X2 = array_X = (array_T ** 0.) * 1e-3
                secondary_species_label = "H2"

            array_x_axis = array_T

            x0 = np.log(array_X)
            x1 = array_P
            x2 = array_T

            if secondary_species == "H":
                # plot the best fit line
                y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
            elif secondary_species == "He":
                y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
            print(f"Best fit line for KD vs P, T: {y_fit}")
            print(f"array_x_axis: {array_x_axis}")
            # print(f"logX: {x0}")
            # print(f"T: {array_T}")
            # print(f"P: {array_P}")
            ax_KD__T.plot(
                array_x_axis, y_fit,
                linestyle='',
                marker="s",
                label=f"Best fit model",
                color='black', markersize=10,
                alpha=0.15
            )
        # ***************************
        # ***************************


















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
        plt.savefig(f"KD_D_wt_vs_P_T__T__{secondary_species}.png", dpi=300)






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
                print(f"temp@df_temporary:\n{df_temporary['Target temperature (K)']}")

            ith_P_T_folder_Pressure = df_temporary["Target pressure (GPa)"].iloc[0]

            # get the axes for the current P_T_folder
            ax = axes.flatten()[list(unique_P_T_folders).index(ith_P_T_folder)]

            # plot KD_sil_to_metal vs Target temperature (K)
            yerr = [[df_temporary["KD_sil_to_metal"] - df_temporary["KD_sil_to_metal_low"]], [df_temporary["KD_sil_to_metal_high"] - df_temporary["KD_sil_to_metal"]]]
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
        plt.savefig(f"KD_sil_to_metal_vs_P_T_panels__{secondary_species}.png", dpi=300)


        # plot all mu_{secondary_species} vs X_{secondary_species}, and color by P_T_folder
        # plt.figure(figsize=(10,6))
        # plt.scatter(
        #     df[f"X_{secondary_species}"], df[f"mu_{secondary_species}"],
        #     s=100,
        #     alpha=0.5
        # )
        # plt.savefig("test__{secondary_species}.png")

        print(f"Created: dataframe with G_hp_per_atom, G_hp_per_atom_error, X_{secondary_species}, etc. from all systems")
        print(f"Files created: all_TI_results_with_X{secondary_species}.csv, X_{secondary_species}_vs_G_hp_per_atom__{secondary_species}.png")



































    if PLOT_MODE == 5 or PLOT_MODE < 0: # KD_D_wt_vs_P_T__lowPT__{secondary_species}.png

        # y_min = 1e-5
        # y_max = 10**(5.2)

        if secondary_species == "He":
            y_min = y_min__He
            y_max = y_max__He
        elif secondary_species == "H":
            y_min = y_min__H
            y_max = y_max__H

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















            # ax_KD.scatter(data_He__two_phase_simulations[x_variable], 
            #                 data_He__two_phase_simulations["KD"],
            #                 **marker_opts_scatter,
            #                 marker=marker_2phase,
            #                 c=data_He__two_phase_simulations[z_variable],
            #                 cmap=cmap,
            #                 norm=norm,
            #                 label="This study (2P)")
            # ax_D_wt.scatter(data_He__two_phase_simulations[x_variable], 
            #                 data_He__two_phase_simulations["D_wt"],
            #                 **marker_opts_scatter,
            #                 marker=marker_2phase,
            #                 c=data_He__two_phase_simulations[z_variable],
            #                 cmap=cmap,
            #                 norm=norm,
            #                 label="This study (2P)")











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




        # ***************************
        # ***************************
        if FIT_MODE == 1:
            # combine T and P from datasets_expt, datasets_comp and df dataframe
            sources = list(datasets_expt) + list(datasets_comp) + [df]  # uncomment if needed
            # array_T = row[f"array_T_{secondary_species}"] # z_variable -- T
            # array_P = row[f"array_P_{secondary_species}"] # x_variable -- P
            array_T          = _concat_cols(sources, "Target temperature (K)")
            array_P          = _concat_cols(sources, "Target pressure (GPa)")

            if secondary_species == "He":
                # array_x_axis = array_X
                array_X = (array_T ** 0.) * 1e-3
                secondary_species_label = "He"

            elif secondary_species == "H":
                # array_x_axis = array_X2
                array_X2 = array_X = (array_T ** 0.) * 1e-3
                secondary_species_label = "H2"

            array_x_axis = array_P

            x0 = np.log(array_X)
            x1 = array_P
            x2 = array_T

            if secondary_species == "H":
                # plot the best fit line
                y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
            elif secondary_species == "He":
                y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
            # print(f"Best fit line for KD vs P, T: {y_fit}")
            # print(f"logX: {x0}")
            # print(f"T: {array_T}")
            # print(f"P: {array_P}")
            ax_KD.plot(
                array_x_axis, y_fit,
                linestyle='',
                marker="s",
                label=f"Best fit model",
                color='black', markersize=10,
                alpha=0.15
            )
        # ***************************
        # ***************************











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
        

        ax_KD.set_xlim(xlim_low, xlim_high)
        ax_D_wt.set_xlim(xlim_low, xlim_high)

        ax_KD.set_ylim(y_min, y_max)
        ax_D_wt.set_ylim(y_min, y_max)



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
        plt.savefig(f"KD_D_wt_vs_P_T__lowPT__{secondary_species}.png", dpi=300)











    if PLOT_MODE == 6 or PLOT_MODE < 0: # KD_D_wt_vs_P_T__T__lowPT__{secondary_species}

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


        # # 1) Plot the colored points
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

        # # 2) Draw per-point errorbars with matching colors
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








        # ***************************
        # ***************************
        if FIT_MODE == 1:
            # combine T and P from datasets_expt, datasets_comp and df dataframe
            sources = list(datasets_expt) + list(datasets_comp) + [df]  # uncomment if needed
            # array_T = row[f"array_T_{secondary_species}"] # z_variable -- T
            # array_P = row[f"array_P_{secondary_species}"] # x_variable -- P
            array_T          = _concat_cols(sources, "Target temperature (K)")
            array_P          = _concat_cols(sources, "Target pressure (GPa)")

            if secondary_species == "He":
                # array_x_axis = array_X
                array_X = (array_T ** 0.) * 1e-3
                secondary_species_label = "He"

            elif secondary_species == "H":
                # array_x_axis = array_X2
                array_X2 = array_X = (array_T ** 0.) * 1e-3
                secondary_species_label = "H2"

            array_x_axis = array_T

            x0 = np.log(array_X)
            x1 = array_P
            x2 = array_T

            if secondary_species == "H":
                # plot the best fit line
                y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
            elif secondary_species == "He":
                y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
            # print(f"Best fit line for KD vs P, T: {y_fit}")
            # print(f"logX: {x0}")
            # print(f"T: {array_T}")
            # print(f"P: {array_P}")
            ax_KD__T.plot(
                array_x_axis, y_fit,
                linestyle='',
                marker="s",
                label=f"Best fit model",
                color='black', markersize=10,
                alpha=0.15
            )
        # ***************************
        # ***************************









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
        plt.savefig(f"KD_D_wt_vs_P_T__T__lowPT__{secondary_species}.png", dpi=300)
        # plt.savefig(f"KD_D_wt_vs_P_T__T__lowPT.pdf")






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
        plt.savefig(f"PT_coverage__{secondary_species}.png", dpi=300)



















    # # # print summary of df_superset cols: f"array_KD_{secondary_species}", f"array_KD_{secondary_species}_prime", f"array_D_wt_{secondary_species}", f"array_X_{secondary_species}" and f"array_Xw_{secondary_species}"
    # print("\nSummary of df_superset:")
    # cols = [
    #     f"array_KD_{secondary_species}",
    #     f"array_KD_prime_{secondary_species}",   # correct placement of 'prime'
    #     f"array_D_wt_{secondary_species}",
    #     f"array_X_{secondary_species}",
    #     f"array_Xw_{secondary_species}",
    # ]
    # cols = [
    #     f"array_D_wt_{secondary_species}",
    #     f"array_D_wt_{secondary_species}_lower",
    #     f"array_D_wt_{secondary_species}_upper",
    # ]
    # # print mean, std, min, max of each column
    # for row in df_superset[cols].iterrows():
    #     idx, data = row
    #     print(f"Index {idx}:")
    #     for col in cols:
    #         if col in data:
    #             print(f"  {col}: mean={data[col].mean():.2e}, std={data[col].std():.2e}, min={data[col].min():.2e}, max={data[col].max():.2e}")
    #         else:
    #             print(f"  {col} not found in row {idx}")




    if PLOT_MODE == 8 or PLOT_MODE < 0:



        # plot array_KD vs array_X only for phase = MgSiO3 -- df_superset
        
        # 4 separate figures for KD vs X, KD_prime vs X, D_wt vs X, and D_wt vs Xw
        # create a 4 x 5 grid of subplots where each row corresponds to a different target "Target pressure (GPa)"
        # and each column corresponds to a different "Target temperature (K)" in ascending order
        fig_1, axes_1 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
        axes_1 = axes_1.flatten()  # flatten the 2D array of axes to 1D for easier iteration
        fig_2, axes_2 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
        axes_2 = axes_2.flatten()  # flatten the 2D array of axes to 1D for easier iteration
        fig_3, axes_3 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
        axes_3 = axes_3.flatten()  # flatten the 2D array of axes to 1D for easier iteration
        fig_4, axes_4 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
        axes_4 = axes_4.flatten()  # flatten the 2D array of axes to 1D for easier iteration
        fig_5, axes_5 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
        axes_5 = axes_5.flatten()  # flatten the 2D array of axes to 1D for easier iteration
        fig_6, axes_6 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
        axes_6 = axes_6.flatten()  # flatten the 2D array of axes to 1D for easier iteration

        marker_opts = dict(marker='o', linestyle='', markersize=10, alpha=1)#,color=base_color)
        marker_opts_scatter = dict(linestyle='', s=5, alpha=0.5)#,edgecolor='black',
        marker_opts_error = dict(linestyle='', markersize=10, alpha=0.25, capsize=3, elinewidth=1)#, color='black',ecolor='black')
        
        magma = plt.get_cmap("magma")
        pastel_magma = pastel_cmap(magma, factor=0.25)  # tweak factor between 0 and 1
        cmap = pastel_magma  # use pastel magma for the plots

        # sort df_superset by "Target pressure (GPa)" and "Target temperature (K)" -- largest to smallest pressure but smallest to largest temperature
        df_superset = df_superset.sort_values(
            by=["Target pressure (GPa)", "Target temperature (K)"],
            ascending=[False, True]  # largest to smallest pressure, smallest to largest temperature
        )


        i_axes = -1  # counter for the axes

        # iterate over all systems in df_superset
        for i, row in df_superset.iterrows():

            phase = row["Phase"]
            pt = row["P_T_folder"]

            if phase != f"MgSiO3_{secondary_species}":
                continue

            i_axes += 1
            # if i_axes == 4, 9 or 14, +1
            if i_axes in [4, 9, 14]:
                i_axes += 1 # to maintain 1 P per row

            # 2) Determine the other phase
            other_phase = f"MgSiO3_{secondary_species}" if phase == f"Fe_{secondary_species}" else f"Fe_{secondary_species}"

            mask = (
                (df_superset["Phase"] == other_phase) &
                (df_superset["P_T_folder"] == pt) &
                (df_superset["Target temperature (K)"] == row["Target temperature (K)"])
                )
            mask_idx = df_superset.index[mask][0]
            

            array_KD = row[f"array_KD_{secondary_species}"]
            array_KD_lower = row[f"array_KD_{secondary_species}_lower"]
            array_KD_upper = row[f"array_KD_{secondary_species}_upper"]

            # print KD shape
            # print(f"array_KD: {array_KD}")
            # print(f"array_KD shape: {array_KD.shape}")
            # exit()

            array_KD_prime = row[f"array_KD_prime_{secondary_species}"]
            array_KD_prime_lower = row[f"array_KD_prime_{secondary_species}_lower"]
            array_KD_prime_upper = row[f"array_KD_prime_{secondary_species}_upper"]

            array_D_wt = row[f"array_D_wt_{secondary_species}"]
            array_D_wt_lower = row[f"array_D_wt_{secondary_species}_lower"]
            array_D_wt_upper = row[f"array_D_wt_{secondary_species}_upper"]

            array_X = row[f"array_X_{secondary_species}"]
            array_X_lower = row[f"array_X_{secondary_species}_lower"]
            array_X_upper = row[f"array_X_{secondary_species}_upper"]

            array_Xw = row[f"array_Xw_{secondary_species}"]
            array_Xw_lower = row[f"array_Xw_{secondary_species}_lower"]
            array_Xw_upper = row[f"array_Xw_{secondary_species}_upper"]

            array_X_in_Fe = row[f"array_X_{secondary_species}_in_Fe"]
            array_X_in_Fe_lower = row[f"array_X_{secondary_species}_in_Fe_lower"]
            array_X_in_Fe_upper = row[f"array_X_{secondary_species}_in_Fe_upper"]


            array_T = row[f"array_T_{secondary_species}"]
            array_P = row[f"array_P_{secondary_species}"]

            if secondary_species == "He":
                KD_chosen = array_KD
                KD_chosen_lower = array_KD_lower
                KD_chosen_upper = array_KD_upper
                
                array_x_axis = array_X
                secondary_species_label = "He"

            elif secondary_species == "H":
                KD_chosen = array_KD_prime
                KD_chosen_lower = array_KD_prime_lower
                KD_chosen_upper = array_KD_prime_upper
                array_X2 = array_X / (2 - array_X)  # X_H2 = X_H / (2 - X_H)
                array_X2_lower = row[f"array_X_{secondary_species}_lower"]
                array_X2_upper = row[f"array_X_{secondary_species}_upper"]

                array_x_axis = array_X2
                secondary_species_label = "H2"

            # prints min, max, mean of the above 5 arrays
            # print(f"array_KD: min={np.min(array_KD)}, max={np.max(array_KD)}, mean={np.mean(array_KD)}")
            # print(f"array_KD_prime: min={np.min(array_KD_prime)}, max={np.max(array_KD_prime)}, mean={np.mean(array_KD_prime)}")
            # print(f"array_D_wt: min={np.min(array_D_wt)}, max={np.max(array_D_wt)}, mean={np.mean(array_D_wt)}")
            # print(f"array_X: min={np.min(array_X)}, max={np.max(array_X)}, mean={np.mean(array_X)}")
            # print(f"array_Xw: min={np.min(array_Xw)}, max={np.max(array_Xw)}, mean={np.mean(array_Xw)}")

            # array_X_other_phase = df_superset.loc[mask_idx, f"array_X_{secondary_species}"]
            # array_X_lower_other_phase = df_superset.loc[mask_idx, f"array_X_{secondary_species}_lower"]
            # array_X_upper_other_phase = df_superset.loc[mask_idx, f"array_X_{secondary_species}_upper"]

            # array_Xw_other_phase = df_superset.loc[mask_idx, f"array_Xw_{secondary_species}"]
            # array_Xw_lower_other_phase = df_superset.loc[mask_idx, f"array_Xw_{secondary_species}_lower"]
            # array_Xw_upper_other_phase = df_superset.loc[mask_idx, f"array_Xw_{secondary_species}_upper"]


            # plot the KD vs X, KD_prime vs X, D_wt vs X, and D_wt vs Xw
            ########################
            axes_1[i_axes].scatter(
                array_X, array_KD,
                label = (
                f"P={row['Target pressure (GPa)']:.0f} GPa, "
                f"T={row['Target temperature (K)']:.0f} K"
                ),                
                **marker_opts_scatter
            )
            # axes_1[i_axes].errorbar(
            #     array_X, array_KD,
            #     yerr=[array_KD-array_KD_lower, array_KD_upper-array_KD],
            #     fmt='none',  # no extra marker
            #     # ecolor=cmap(norm(row["Target temperature (K)"])),  # single RGBA tuple
            #     **marker_opts_error,
            #     marker=marker_TI  # use the TI marker for errorbars
            # )
            axes_1[i_axes].fill_between(
                array_X,
                array_KD_lower,
                array_KD_upper,
                alpha=0.2, label='Error range',
                color=axes_1[i_axes].collections[0].get_edgecolor()  # same as the scatter points
            )
            # horizontal line at y=K_D
            axes_1[i_axes].axhline(
                y=row["KD_sil_to_metal"],
                color='black', linestyle='--', linewidth=1,
                # label=f"K$_D$ = {row['KD_sil_to_metal']:.2e}"
            )
            # error
            axes_1[i_axes].axhline(
                y=row["KD_sil_to_metal_low"],
                color='black', linestyle=':', linewidth=1,
            )
            axes_1[i_axes].axhline(
                y=row["KD_sil_to_metal_high"],
                color='black', linestyle=':', linewidth=1,
            )

            # ***************************
            # ***************************
            if FIT_MODE == 1 and secondary_species == "He":
                # plot the best fit line
                x0 = np.log(array_x_axis)
                x1 = array_P
                x2 = array_T
                # print("array_P:", array_P)
                y_fit = best_fn(x0, x1, x2)
                axes_1[i_axes].plot(
                    array_x_axis, y_fit,
                    linestyle='--',
                    label=f"Best fit model",
                    color='black', linewidth=1.5
                )
            # ***************************
            # ***************************


            ########################

            ########################
            axes_2[i_axes].scatter(
                array_X, array_KD_prime,
                label = (
                f"P={row['Target pressure (GPa)']:.0f} GPa, "
                f"T={row['Target temperature (K)']:.0f} K"
                ),             
                **marker_opts_scatter
            )
            # axes_2[i_axes].errorbar(
            #     array_X, array_KD_prime,
            #     yerr=[array_KD_prime-array_KD_prime_lower, array_KD_prime_upper-array_KD_prime],
            #     fmt='none',  # no extra marker
            #     # ecolor=cmap(norm(row["Target temperature (K)"])),  # single RGBA tuple
            #     **marker_opts_error,
            #     marker=marker_TI  # use the TI marker for errorbars
            # )
            axes_2[i_axes].fill_between(
                array_X,
                array_KD_prime_lower,
                array_KD_prime_upper,
                alpha=0.2, label='Error range',
                color=axes_2[i_axes].collections[0].get_edgecolor()  # same as the scatter points
            )

            # ***************************
            # ***************************
            if FIT_MODE == 1 and secondary_species == "H":
                # plot the best fit line
                x0 = np.log(array_x_axis)
                x1 = array_P
                x2 = array_T
                # print("array_P:", array_P)
                y_fit = best_fn(x0, x1, x2)
                axes_2[i_axes].plot(
                    array_x_axis, y_fit,
                    linestyle='--',
                    label=f"Best fit model",
                    color='black', linewidth=1.5
                )
            # ***************************
            # ***************************


            ########################

            ########################
            axes_3[i_axes].scatter(
                array_X, array_D_wt,
                label = (
                f"P={row['Target pressure (GPa)']:.0f} GPa, "
                f"T={row['Target temperature (K)']:.0f} K"
                ),
                **marker_opts_scatter
            )
            # axes_3[i_axes].errorbar(
            #     array_X, array_D_wt,
            #     yerr=[array_D_wt-array_D_wt_lower, array_D_wt_upper-array_D_wt],
            #     fmt='none',  # no extra marker
            #     # ecolor=cmap(norm(row["Target temperature (K)"])),  # single RGBA tuple
            #     **marker_opts_error,
            #     marker=marker_TI  # use the TI marker for errorbars
            # )
            axes_3[i_axes].fill_between(
                array_X,
                array_D_wt_lower,
                array_D_wt_upper,
                alpha=0.2, label='Error range',
                color=axes_3[i_axes].collections[0].get_edgecolor()  # same as the scatter points
            )


            # horizontal line at y=D_wt
            axes_3[i_axes].axhline(
                y=row["D_wt"],
                color='black', linestyle='--', linewidth=1,
                # label=f"D$_{{wt}}$ = {row['D_wt']:.2e}"
            )
            # error
            axes_3[i_axes].axhline(
                y=row["D_wt_low"],
                color='black', linestyle=':', linewidth=1,
                # label=f"D$_{{wt}}$ error = {row['D_wt_error']:.2e}"
            )
            axes_3[i_axes].axhline(
                y=row["D_wt_high"],
                color='black', linestyle=':', linewidth=1,
            )
            ########################

            ########################
            axes_4[i_axes].scatter(
                array_Xw, array_D_wt,
                label = (
                f"P={row['Target pressure (GPa)']:.0f} GPa, "
                f"T={row['Target temperature (K)']:.0f} K"
                ),
                **marker_opts_scatter
            )
            # axes_4[i_axes].errorbar(
            #     array_Xw, array_D_wt,
            #     yerr=[array_D_wt-array_D_wt_lower, array_D_wt_upper-array_D_wt],
            #     fmt='none',  # no extra marker
            #     # ecolor=cmap(norm(row["Target temperature (K)"])),  # single RGBA tuple
            #     **marker_opts_error,
            #     marker=marker_TI  # use the TI marker for errorbars
            # )
            axes_4[i_axes].fill_between(
                array_Xw,
                array_D_wt_lower,
                array_D_wt_upper,
                alpha=0.2, label='Error range',
                color=axes_4[i_axes].collections[0].get_edgecolor()  # same as the scatter points
            )
            # horizontal line at y=D_wt
            axes_4[i_axes].axhline(
                y=row["D_wt"],
                color='black', linestyle='--', linewidth=1,
                # label=f"D$_{{wt}}$ = {row['D_wt']:.2e}"
            )
            # error
            axes_4[i_axes].axhline(
                y=row["D_wt_low"],
                color='black', linestyle=':', linewidth=1,
                # label=f"D$_{{wt}}$ error = {row['D_wt_error']:.2e}"
            )
            axes_4[i_axes].axhline(
                y=row["D_wt_high"],
                color='black', linestyle=':', linewidth=1,
            )
            ########################
            # plot X vs Xw
            axes_5[i_axes].scatter(
                array_X, array_Xw,
                label = (
                f"P={row['Target pressure (GPa)']:.0f} GPa, "
                f"T={row['Target temperature (K)']:.0f} K"
                ),
                **marker_opts_scatter
            )
            # axes_5[i_axes].errorbar(
            #     array_X, array_Xw,
            #     yerr=[array_Xw-array_Xw_lower, array_Xw_upper-array_Xw],
            #     fmt='none',  # no extra marker
            #     # ecolor=cmap(norm(row["Target temperature (K)"])),  # single RGBA tuple
            #     **marker_opts_error,
            #     marker=marker_TI  # use the TI marker for errorbars
            # )
            axes_5[i_axes].fill_between(
                array_X,
                array_Xw_lower,
                array_Xw_upper,
                alpha=0.2, label='Error range',
                color=axes_5[i_axes].collections[0].get_edgecolor()  # same as the scatter points
            )
                
            # print(f"array_X: {array_X}")
            # print(f"array_Xw: {array_Xw}")
            # print(f"array_Xw-array_Xw_lower: {array_Xw - array_Xw_lower}")
            # print(f"array_Xw-array_Xw_upper: {array_Xw - array_Xw_upper}")
            ########################
            # plot X_in_Fe vs X
            axes_6[i_axes].scatter(
                array_X, array_X_in_Fe,
                label = (
                f"P={row['Target pressure (GPa)']:.0f} GPa, "
                f"T={row['Target temperature (K)']:.0f} K"
                ),
                **marker_opts_scatter
            )
            axes_6[i_axes].fill_between(
                array_X,
                array_X_in_Fe_lower,
                array_X_in_Fe_upper,
                alpha=0.2, label='Error range',
                color=axes_6[i_axes].collections[0].get_edgecolor()  # same as the scatter points
            )
            ########################



        # set x and y limits for all axes
        for axes in [axes_1, axes_2, axes_3, axes_4, axes_5, axes_6]:
            for i_axes, ax in enumerate(axes):
                # if axes is not axes_2:
                #     if axes is not axes_1:
                #         ax.set_xscale("log")
                ax.set_xscale("log")
                # if axes is not axes_2:
                ax.set_yscale("log")
                ax.grid(True, which="both", ls="--", alpha=0.5)

                if i_axes not in (0, 5, 10, 15):
                    ax.tick_params(labelleft=False)
                else:
                    ax.tick_params(labelleft=True)
                    if axes is axes_1:
                        ax.set_ylabel(f"K$_D$")
                    elif axes is axes_2:
                        ax.set_ylabel(f"K$_D^{{H_{{2,sil}}\\rightleftharpoons H_{{Fe}}}}$")
                    elif axes is axes_3:
                        ax.set_ylabel(f"D$_{{wt}}$")
                    elif axes is axes_4:
                        ax.set_ylabel(f"D$_{{wt}}$")
                    elif axes is axes_5:
                        ax.set_ylabel(f"X$_{{w,{secondary_species}}}$")
                    elif axes is axes_6:
                        ax.set_ylabel(fr"$X_{{{secondary_species}}}^{{Fe}}$")

                if i_axes >= 15:
                    ax.tick_params(labelbottom=True)         # force it ON explicitly
                    # If shared x hides it, unhide individual labels:
                    for lbl in ax.get_xticklabels():
                        lbl.set_visible(True)
                    if axes is not axes_4 and axes is not axes_5:
                        ax.set_xlabel(fr"$X_{{{secondary_species}}}^{{MgSiO_3}}$")
                    elif axes is axes_4:
                        ax.set_xlabel(fr"$X_{{w,{secondary_species}}}^{{MgSiO_3}}$")
                    elif axes is axes_5:
                        ax.set_xlabel(fr"$X_{{{secondary_species}}}^{{MgSiO_3}}$")
                    elif axes is axes_6:
                        ax.set_xlabel(fr"$X_{{{secondary_species}}}^{{MgSiO_3}}$")
                else:
                    ax.tick_params(labelbottom=False)

                # delete sub-plots that are not used -- i_axes = 4,9,14
                if i_axes in (4, 9, 14):
                    ax.remove()

                # legend
                # if i_axes < 15:
                #     ax.legend(loc="lower right", fontsize=8)
                # else:
                #     ax.legend(loc="upper right", fontsize=8)
                ax.legend(loc="best", fontsize=8)

                # x axis limits
                # if axes is not axes_3 and axes is not axes_4:
                #     ax.set_xlim(1e-4, 100/109)
                # else:
                #     ax.set_xlim(1e-4, 0.1)  # set x
                # ax.set_xlower(1e-4)
                ax.set_xlim(1e-6, None)

                if axes is axes_6:
                    # set y limits for axes_6
                    ax.set_ylim(1e-6, 2.0)

        # # If *all* subplots share the same legend entries, collect once:
        # handles, labels = axes_1[0].get_legend_handles_labels()
        # # Place a single legend outside (adjust bbox_to_anchor as needed)
        # axes_1[0].figure.legend(handles, labels, loc="upper center",
        #                         bbox_to_anchor=(0.5, 1.02), ncol=len(labels),
        #                         frameon=False, fontsize=8)
            # only keep x-axis labels

        # for ax in axes_2:
        #     # ax.set_xlim(0, 1)
        #     # ax.set_ylim(1e-6, 1e2)
        #     # ax.set_xscale("log")
        #     # ax.set_yscale("log")
        #     ax.grid(True, which="both", ls="--", alpha=0.3)
        #     ax.set_xlabel(f"X$_{{{secondary_species}}}$")
        #     ax.set_ylabel(f"K$_D'$")
        #     ax.legend(loc="upper right", fontsize=8)

        # for ax in axes_3:
        #     # ax.set_xlim(0, 1)
        #     # ax.set_ylim(1e-6, 1e2)
        #     # ax.set_xscale("log")
        #     # ax.set_yscale("log")
        #     ax.grid(True, which="both", ls="--", alpha=0.3)
        #     ax.set_xlabel(f"X$_{{{secondary_species}}}$")
        #     ax.set_ylabel(f"D$_{{wt}}$")
        #     # ax.legend(loc="upper right", fontsize=8)

        # for ax in axes_4:
        #     # # ax.set_xlim(0, 1)
        #     # ax.set_ylim(1e-6, 1e2)
        #     # ax.set_xscale("log")
        #     # ax.set_yscale("log")
        #     ax.grid(True, which="both", ls="--", alpha=0.3)
        #     ax.set_xlabel(f"X$_{{w,{secondary_species}}}$")
        #     ax.set_ylabel(f"D$_{{wt}}$")
        #     # ax.legend(loc="upper right", fontsize=8)

        # set the title for each figure
        fig_1.suptitle(
            f"K$_D$ vs X for {secondary_species}",
            fontsize=10
        )
        fig_2.suptitle(
            f"K$_D^{{H_{{2,sil}}\\rightleftharpoons H_{{Fe}}}}$ vs X for {secondary_species}",
            fontsize=10
        )
        fig_3.suptitle(
            f"D$_{{wt}}$ vs X for {secondary_species}",
            fontsize=10
        )
        fig_4.suptitle(
            f"D$_{{wt}}$ vs Xw for {secondary_species}",
            fontsize=10
        )
        fig_5.suptitle(
            f"X vs Xw for {secondary_species}",
            fontsize=10
        )
        fig_6.suptitle(
            f"X$_{{{secondary_species}}}^{{Fe}}$ vs X$_{{{secondary_species}}}^{{MgSiO_3}}$",
            fontsize=10
        )

        # from matplotlib.collections import PathCollection, LineCollection
        # def sanitize_for_pdf(fig):
        #     for ax in fig.axes:
        #         # Remove empty collections or patch their arrays
        #         for coll in list(ax.collections):
        #             try:
        #                 # 1) Empty PathCollection (e.g., scatter/markers with 0 points)
        #                 if isinstance(coll, PathCollection):
        #                     offs = np.asarray(getattr(coll, "get_offsets", lambda: np.array([]))())
        #                     if offs.size == 0:
        #                         coll.remove()
        #                         continue
        #                     lw = np.asarray(coll.get_linewidths())
        #                     if lw.size == 0:
        #                         coll.set_linewidths((0.0,))   # avoid np.max([]) in PDF backend
        #                     sz = np.asarray(getattr(coll, "get_sizes", lambda: np.array([]))())
        #                     if sz.size == 0:
        #                         coll.set_sizes((0.0,))
        #                 # 2) Empty LineCollection (e.g., errorbar caps with no segments)
        #                 if isinstance(coll, LineCollection):
        #                     if not coll.get_segments():       # empty list
        #                         coll.remove()
        #                         continue
        #             except Exception:
        #                 pass
        #         # Also remove truly empty Line2D lines (rare but safe)
        #         for line in list(ax.lines):
        #             x = np.asarray(line.get_xdata(orig=False))
        #             y = np.asarray(line.get_ydata(orig=False))
        #             if x.size == 0 or y.size == 0:
        #                 try: line.remove()
        #                 except Exception: pass



        # sanitize_for_pdf(fig_1)
        # sanitize_for_pdf(fig_2)
        # sanitize_for_pdf(fig_3)
        # sanitize_for_pdf(fig_4)
        # sanitize_for_pdf(fig_5)
        # sanitize_for_pdf(fig_6)



        # save the figures
        fig_1.savefig(f"array__KD_vs_X__{secondary_species}.png", dpi=300)
        fig_2.savefig(f"array__KD_prime_vs_X__{secondary_species}.png", dpi=300)
        fig_3.savefig(f"array__D_wt_vs_X__{secondary_species}.png", dpi=300)
        fig_4.savefig(f"array__D_wt_vs_Xw__{secondary_species}.png", dpi=300)
        fig_5.savefig(f"array__X_vs_Xw__{secondary_species}.png", dpi=300)
        fig_6.savefig(f"array__X_in_Fe_vs_X_in_MgSiO3__{secondary_species}.png", dpi=300)

        # save as pdf
        # fig_1.savefig("array__KD_vs_X.pdf")
        # fig_2.savefig(f"array__KD_prime_vs_X.pdf")
        # fig_3.savefig(f"array__D_wt_vs_X.pdf")
        # fig_4.savefig(f"array__D_wt_vs_Xw.pdf")
        # fig_5.savefig(f"array__X_vs_Xw.pdf")
        # fig_6.savefig(f"array__X_in_Fe_vs_X_in_MgSiO3.pdf")













































    if PLOT_MODE == 15 or PLOT_MODE < 0: # KD_D_wt_vs_P_T__lowPT__only_past_data__{secondary_species}.png

        # y_min = 1e-5
        # y_max = 10**(5.2)

        if secondary_species == "He":
            y_min = y_min__He
            y_max = y_max__He
        elif secondary_species == "H":
            y_min = y_min__H
            y_max = y_max__H

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




        # 2) Draw per-point errorbars with matching colors
        temps  = df_superset[z_variable].values
        colors = cmap(norm(temps))
























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
        # plt.savefig(f"KD_D_wt_vs_P_T__lowPT__{secondary_species}.png", dpi=300)
        plt.savefig(f"KD_D_wt_vs_P_T__lowPT__only_past_data__{secondary_species}.png", dpi=300)










        





















    if PLOT_MODE == 16 or PLOT_MODE < 0: # KD_D_wt_vs_P_T__T__lowPT__{secondary_species}__past_data_only

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





        # # 2) Draw per-point errorbars with matching colors
        temps  = df_superset[z_variable].values
        colors = cmap(norm(temps))




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


        # plt.savefig(f"KD_D_wt_vs_P_T__T__{secondary_species}.png", dpi=300)
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
        # plt.savefig(f"KD_D_wt_vs_P_T__T__lowPT__{secondary_species}.png", dpi=300)
        plt.savefig(f"KD_D_wt_vs_P_T__T__lowPT__only_past_data__{secondary_species}.png", dpi=300)
        # plt.savefig(f"KD_D_wt_vs_P_T__T__lowPT.pdf")
























    if PLOT_MODE == 21 or PLOT_MODE < 0: # paper__KD_D_wt__vs_P_T_{secondary_species}

        # fig, axes_KD_D_wt__P = plt.subplots(2, 2, figsize=(12, 10))#, sharex=True, sharey=True)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        # ax1, ax_KD, ax3, ax_D_wt = axes.flatten()
        # ax_KD_low, ax_KD_high, ax_D_wt_low, ax_D_wt_high = axes_KD_D_wt__P.flatten()
    
        import matplotlib.gridspec as gridspec

        # sort df_superset by pressure and KD_sil_to_metal
        df_superset = df_superset.sort_values(by=["Target pressure (GPa)", "KD_sil_to_metal"])

        fig = plt.figure(figsize=(12, 12))


        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times"]
        # plt.rcParams.update({
        #     "font.weight": "medium",          # bold text everywhere
        #     "axes.labelweight": "medium",     # x/y labels
        #     "axes.titleweight": "medium",     # titles
        #     # "text.latex.preamble": r"\usepackage{sfmath}",  # bold math
        # })


        # all font sizes
        font_size_title = 14
        font_size_labels = 16
        font_size_ticks = 14
        font_size_legend = 12


        x_min = -8
        x_mid = 70
        x_max = 1071


        # Big grid: 2 rows, 1 col
        outer = gridspec.GridSpec(2, 1, height_ratios=[1, 0.02], hspace=0.2)

        # Top block: looser spacing
        gs_top = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], 
                                                width_ratios=[0.5, 0.5],
                                                wspace=0.01)

        # Bottom block: tighter spacing
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1])


        # # 2 rows × 2 columns, with column widths 40% and 60%
        # gs = gridspec.GridSpec(2, 2, 
        #                     width_ratios=[0.4, 0.6],
        #                     height_ratios=[1, 1, 0.06],
        #                     hspace=0.15, wspace=0.05)

        ax_KD_low = fig.add_subplot(gs_top[0, 0])  # Top-left (40%)
        ax_KD_high = fig.add_subplot(gs_top[0, 1])  # Top-right (60%)
        ax_D_wt_low = fig.add_subplot(gs_top[1, 0])  # Bottom-left (40%)
        ax_D_wt_high = fig.add_subplot(gs_top[1, 1])  # Bottom-right (60%)



        marker_TI="o"  # marker for TI points
        marker_2phase="s"  # marker for two-phase points
        marker_other_comp_studies=["^", "D", "v", "<", ">"] # array of markers for other studies
        marker_other_expt_studies=["p", "H", "h", "*", ""]  # array of markers for other experimental studies

        marker_opts = dict(marker='o', linestyle='', markersize=10, alpha=1)#,color=base_color)
        marker_opts_scatter = dict(linestyle='', s=200, alpha=1,edgecolors='black', linewidths=1)
        marker_opts_error = dict(linestyle='', markersize=10, alpha=1, capsize=5, elinewidth=1)#, color='black',ecolor='black')
        marker_opts_scatter__others = dict(linestyle='', s=200, alpha=0.5)#, edgecolors='k',linewidths=0)#,color=base_color, alpha=0.5
        marker_opts_error__others = dict(linestyle='', markersize=10, capsize=5, elinewidth=1,alpha=0.5)#, color='black',ecolor='black')
        marker_opts_scatter__2phase = dict(linestyle='', s=400, alpha=0.8,edgecolors='black', linewidths=2)
        marker_opts_error__2phase = dict(linestyle='', markersize=10, alpha=0.8, capsize=5, elinewidth=2)
        # marker_opts_scatter__2phase_v2 = dict(linestyle='', s=200, alpha=1,edgecolors='black', linewidths=2)
        # marker_opts_error__2phase_v2 = dict(linestyle='', markersize=10, alpha=1, capsize=5, elinewidth=2)

        # add log temperature to df
        df_superset["log(Target temperature (K))"] = np.log10(df_superset["Target temperature (K)"])
        df_superset["log(Target pressure (GPa))"] = np.log10(df_superset["Target pressure (GPa)"])
        for data in datasets_comp:
            data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
            data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]
        for data in datasets_expt:
            data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
            data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]

        if secondary_species == "He":
            # data_He__two_phase_simulations["log(Target temperature (K))"] = [ np.log10(p) for p in data_He__two_phase_simulations["Target temperature (K)"] ]
            # data_He__two_phase_simulations["log(Target pressure (GPa))"] = [ np.log10(p) for p in data_He__two_phase_simulations["Target pressure (GPa)"] ]
            for data in datasets_2phases:
                data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
                data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]


        ######################################
        x_variable = "Target pressure (GPa)"  # x-axis variable for all plots
        # x_variable = "log(Target pressure (GPa))"  # x-axis variable for all plots
        # z_variable = "Target temperature (K)"  # z-axis variable for all plots -- color coding
        z_variable = "log(Target temperature (K))"
        ######################################
        
        if x_variable == "log(Target pressure (GPa))":
            x_min = np.log10(5)
            x_mid = np.log10(200.0)
            x_max = np.log10(1100)

        # create a colormap based on the temperature range
        if z_variable == "log(Target temperature (K))":
            temp_min = 1000  # minimum temperature in K
            temp_max = 16000  # maximum temperature in K
            log10_temp_min = np.log10(temp_min)  # minimum temperature in K
            log10_temp_max = np.log10(temp_max)  # maximum temperature in K
            cbar_ticks = np.log10([1000, 2000, 4000, 8000, 16000])
            cbar_labels = [1000, 2000, 4000, 8000, 16000]
            norm = plt.Normalize(
                vmin=log10_temp_min,
                vmax=log10_temp_max
            )
        elif z_variable == "Target temperature (K)":
            temp_min = 1000  # minimum temperature in K
            temp_max = 15000  # maximum temperature in K
            cbar_ticks = [1000, 3000, 6000, 9000, 12000, 15000]
            cbar_labels = [f"{int(t)}" for t in cbar_ticks]
            norm = plt.Normalize(
                vmin=temp_min,
                vmax=temp_max
            )


        magma = plt.get_cmap("magma")
        viridis = plt.get_cmap("viridis")
        coolwarm = plt.get_cmap("coolwarm")
        pastel_magma = pastel_cmap(magma, factor=0.05)  # tweak factor between 0 and 1
        pastel_viridis = pastel_cmap(viridis, factor=0.05)  # tweak factor between 0 and 1
        pastel_coolwarm = pastel_cmap(coolwarm, factor=0.05)  # tweak factor between 0 and 1
        cmap = pastel_viridis  # use pastel magma/viridis/... for the plots
        # cmap = pastel_magma  # use pastel magma/viridis/... for the plots
        # cmap = pastel_coolwarm  # use pastel magma/viridis/... for the plots










        for axes_low_or_high in ["low", "high"]:


            # KD = all first elements of each row from df_superset[f"array_KD_{secondary_species}"]
            KD = df_superset[f"array_KD_{secondary_species}"].str[0].to_numpy()
            KD_low = df_superset[f"array_KD_{secondary_species}_lower"].str[0].to_numpy()
            KD_high = df_superset[f"array_KD_{secondary_species}_upper"].str[0].to_numpy()

            # D_wt = all first elements of each row from df_superset[f"array_D_wt_{secondary_species}"]
            D_wt = df_superset[f"array_D_wt_{secondary_species}"].str[0].to_numpy()
            D_wt_low = df_superset[f"array_D_wt_{secondary_species}_lower"].str[0].to_numpy()
            D_wt_high = df_superset[f"array_D_wt_{secondary_species}_upper"].str[0].to_numpy()

            # x and z variables
            df_x_variable = df_superset[x_variable].values
            df_z_variable = df_superset[z_variable].values



            if axes_low_or_high == "low":
                ax_KD = ax_KD_low
                ax_D_wt = ax_D_wt_low

                # limit df_x_variable, df_z_variable and KD, D_wt, ... to df_x_variable < x_mid
                mask = np.asarray(df_x_variable) < x_mid
                df_x_variable = df_x_variable[mask]
                df_z_variable = df_z_variable[mask]
                KD = KD[mask]
                KD_low = KD_low[mask]
                KD_high = KD_high[mask]
                D_wt = D_wt[mask]
                D_wt_low = D_wt_low[mask]
                D_wt_high = D_wt_high[mask]

            elif axes_low_or_high == "high":
                ax_KD = ax_KD_high
                ax_D_wt = ax_D_wt_high

                # limit ... to > x_mid
                mask = np.asarray(df_x_variable) > x_mid
                # print(f"Number of points in high P plot: {len(mask)}, x_mid: {x_mid}, df_x_variable: {df_x_variable}")
                df_x_variable = df_x_variable[mask]
                df_z_variable = df_z_variable[mask]
                KD = KD[mask]
                KD_low = KD_low[mask]
                KD_high = KD_high[mask]
                D_wt = D_wt[mask]
                D_wt_low = D_wt_low[mask]
                D_wt_high = D_wt_high[mask]

            # --- Panel 3: KD_sil_to_metal (log y) ---

            # 1) Plot the colored points
            sc = ax_KD.scatter(
                df_x_variable,
                KD,
                c=df_z_variable,
                cmap=cmap,
                norm=norm,
                **marker_opts_scatter,
                marker=marker_TI,  # use the TI marker for scatter points
                label="This study (TI+AIMD)",
                # rasterized=True
            )

            # 2) Draw per-point errorbars with matching colors
            temps  = df_z_variable
            colors = cmap(norm(temps))

            for x0, y0, y_low, y_high, c in zip(
                df_x_variable,
                KD,
                KD_low,
                KD_high,
                # df_superset["KD_sil_to_metal"],
                # df_superset["KD_sil_to_metal_low"],
                # df_superset["KD_sil_to_metal_high"],
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
                df_x_variable,
                D_wt,
                c=df_z_variable,
                cmap=cmap,
                norm=norm,
                **marker_opts_scatter,
                marker=marker_TI,  # use the TI marker for scatter points
                label="This study (TI-AIMD)",
                # rasterized=True
            )
            # 2) Draw per-point errorbars with matching colors
            for x0, y0, y_low, y_high, c in zip(
                df_x_variable,
                D_wt,
                D_wt_low,
                D_wt_high,
                # df_superset["KD_sil_to_metal"],
                # df_superset["KD_sil_to_metal_low"],
                # df_superset["KD_sil_to_metal_high"],
                colors
            ):

                low  = y0 - y_low
                high = y_high - y0

                # shape (2,1) array: [[low], [high]]
                yerr = [[low], [high]]

                ax_D_wt.errorbar(
                    x0, y0,
                    yerr=yerr,
                    fmt='none',       # no extra marker
                    ecolor=c,         # single RGBA tuple
                    **marker_opts_error,
                    marker=marker_TI  # use the TI marker for errorbars
                )














            # # if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
            if secondary_species == "He":

            #     ax_KD.scatter(data_He__two_phase_simulations[x_variable], 
            #                     data_He__two_phase_simulations["KD"],
            #                     **marker_opts_scatter,
            #                     marker=marker_2phase,
            #                     c=data_He__two_phase_simulations[z_variable],
            #                     cmap=cmap,
            #                     norm=norm,
            #                     label="This study (2P-AIMD)",
            #                     # rasterized=True
            #                     )
            #     ax_D_wt.scatter(data_He__two_phase_simulations[x_variable], 
            #                     data_He__two_phase_simulations["D_wt"],
            #                     **marker_opts_scatter,
            #                     marker=marker_2phase,
            #                     c=data_He__two_phase_simulations[z_variable],
            #                     cmap=cmap,
            #                     norm=norm,
            #                     label="This study (2P-AIMD)",
            #                     # rasterized=True
            #                     )
                plot_studies(
                ax_KD,
                ax_D_wt,
                datasets_2phases,
                x_variable,
                z_variable,
                cmap,
                norm,
                marker_2phase,
                marker_opts_scatter__2phase,
                marker_opts_error__2phase,
                x_low=x_mid if axes_low_or_high=="high" else None,
                x_high=x_mid if axes_low_or_high=="low" else None
                )










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
            marker_opts_error__others,
            x_low=x_mid if axes_low_or_high=="high" else None,
            x_high=x_mid if axes_low_or_high=="low" else None
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
            marker_opts_error__others,
            x_low=x_mid if axes_low_or_high=="high" else None,
            x_high=x_mid if axes_low_or_high=="low" else None
            )
            ###########################








            # ***************************
            # ***************************
            if FIT_MODE == 1:
                # combine T and P from datasets_expt, datasets_comp and df dataframe
                sources = list(datasets_expt) + list(datasets_comp) + [df]  # uncomment if needed
                # array_T = row[f"array_T_{secondary_species}"] # z_variable -- T
                # array_P = row[f"array_P_{secondary_species}"] # x_variable -- P
                array_T          = _concat_cols(sources, "Target temperature (K)")
                array_P          = _concat_cols(sources, "Target pressure (GPa)")

                if secondary_species == "He":
                    # array_x_axis = array_X
                    array_X = (array_T ** 0.) * 1e-3
                    secondary_species_label = "He"

                elif secondary_species == "H":
                    # array_x_axis = array_X2
                    array_X2 = array_X = (array_T ** 0.) * 1e-3
                    secondary_species_label = "H2"

                array_x_axis = array_P

                x0 = np.log(array_X)
                x1 = array_P
                x2 = array_T

                if secondary_species == "H":
                    # plot the best fit line
                    y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
                elif secondary_species == "He":
                    y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
                # print(f"Best fit line for KD vs P, T: {y_fit}")
                # print(f"logX: {x0}")
                # print(f"T: {array_T}")
                # print(f"P: {array_P}")
                ax_KD.plot(
                    array_x_axis, y_fit,
                    linestyle='',
                    marker="s",
                    label=f"Best fit model",
                    color='black', markersize=10,
                    alpha=0.15
                )
            # ***************************
            # ***************************





        # show 1 colorbar for both KD and D_wt plots
        # 1) Create a colorbar for the temperature range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # only needed for older matplotlib versions
        # 2) Add the colorbar to the figure -- width == width of 1 plot
        # cbar = fig.colorbar(sm, ax=[ax_D_wt_low, ax_D_wt_high],
        #                     orientation='horizontal',
        #                     # fraction=1, pad=0.04,
        #                     pad=100,  # space between colorbar and plot
        #                     ticks=np.linspace(temp_min, temp_max, 5),
        #                     location='bottom',  # 'top' or 'bottom'
        #                     shrink=1,      # shrink to 80% of the original size
        #                     aspect=50,     # thinner bar
        #                     )
        cax = fig.add_subplot(gs_bottom[0, :])
        # cax.axis("off")  # hide frame and ticks
        # cax.set_xlabel("Pressure (GPa)", labelpad=10)  # shared label
        cbar = fig.colorbar(
            sm, cax=cax, orientation='horizontal',
            ticks=np.linspace(temp_min, temp_max, 5)
        )
        cbar.set_label("Temperature (K)", fontsize=font_size_labels)#, rotation=270, labelpad=15)
        # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_labels)
        cbar.ax.tick_params(labelsize=font_size_ticks)
        # bold boundary
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")





        # # x lim ,  y lim

        for axes in [ax_KD_low, ax_KD_high, ax_D_wt_low, ax_D_wt_high]:

            # draw horizontal dashed line at y=1
            axes.axhline(y=1, color='gray', linestyle='-', linewidth=7.5, alpha=0.35)
            # annotate on top and immediately below it, on the right end side
            if axes in [ax_KD_low, ax_D_wt_low]:
                axes.annotate("Lithophile", xy=(0, 1), xytext=(-7, 0.6),
                            fontsize=12, ha='left', color='dimgray')
                axes.annotate("Siderophile", xy=(0, 1), xytext=(-7, 1.3),
                            fontsize=12, ha='left', color='dimgray')

            # if secondary_species == "He":
            #     y_min = 1e-5 # 1e-5
            #     y_max = 1e1 #1e1
            # elif secondary_species == "H":
            #     y_min = 1e-3 #1e-3
            #     y_max = 1e6
            axes.set_ylim(y_min, y_max)
            if axes in [ax_KD_low, ax_D_wt_low]:
                axes.set_xlim(x_min, x_mid)
            else:
                axes.set_xlim(x_mid, x_max)
            axes.set_yscale("log")
            axes.grid(True)

            # for bottom axes, set 1 x label
            # if axes in [ax_D_wt_low, ax_D_wt_high]:
            #     axes.set_xlabel("Pressure (GPa)")
            fig.supxlabel("Pressure (GPa)", y=0.15, fontsize=font_size_labels)

            # for left axes, set y labels
            if axes in [ax_KD_low]:
                axes.set_ylabel(r"Equilibrium Constant ($K_{D}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
            elif axes in [ax_D_wt_low]:
                axes.set_ylabel(r"Partition Coefficient ($D_{wt}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")

            # no y tick labels for right panels
            if axes in [ax_KD_high, ax_D_wt_high]:
                axes.set_yticklabels([])

            axes.tick_params(axis='y', which='both', direction='in', pad=15)

            # move ticks to right side on right panels
            if axes in [ax_KD_high, ax_D_wt_high]:
                axes.yaxis.tick_right()                     # move ticks to right side
                # axes.yaxis.set_label_position("right")      # move y-axis label too
                axes.tick_params(axis='y', which='both', direction='in', pad=5)  # adjust tick direction/padding

            # font sizes
            for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                label.set_fontsize(font_size_ticks)
            axes.set_xlabel(axes.get_xlabel(), fontsize=font_size_labels)
            axes.set_ylabel(axes.get_ylabel(), fontsize=font_size_labels)

            # bold boundaries
            for spine in axes.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("black")

        # Legend
        # 1) Grab handles & labels from one of the axes
        # handles, labels = ax_KD_low.get_legend_handles_labels()
        # Collect from both axes
        handles1, labels1 = ax_KD_low.get_legend_handles_labels()
        handles2, labels2 = ax_KD_high.get_legend_handles_labels()

        # Merge
        handles = handles2 + handles1
        labels  = labels2 + labels1

        # Deduplicate while keeping order
        unique = dict(zip(labels, handles))   # keys=labels, values=handles
        handles, labels = unique.values(), unique.keys()

        if secondary_species == "He":
            ncol_legend = 4
        elif secondary_species == "H":
            ncol_legend = 4

        # 2) Create a single legend on the right side of the figure
        fig.legend(
            handles,
            labels,
            loc='lower center',        # center vertically on the right edge
            fontsize=font_size_legend,
            # borderaxespad=0.1,
            bbox_to_anchor=(0.50, 0.9),
            # frameon=False,
            ncol=ncol_legend,  # number of columns in the legend
            # mode='expand',  # 'expand' to fill the space, 'none'
            labelspacing=1.,    # default is 0.5; larger → more space
            handletextpad=1.0,   # default is 0.8; larger → more space between handle & text
            columnspacing=2,   # if you have multiple columns, space between them
            handlelength=1.0,
            markerscale=0.75             # shrink markers to ... % in legend
        )

        # 3) Shrink the subplots to make room for the legend
        # fig.subplots_adjust(right=0.82)  # push the plot area leftward






        # fig.subplots_adjust(top=0.88)

        # fig.suptitle(
        #     f"Equilibrium Constant ($K_D$) and Partition Coefficient ($D_{{wt}}$) "
        #     f"for the reaction {secondary_species}$_{{silicates}}$ $\\rightleftharpoons$ {secondary_species}$_{{metal}}$.\n"
        #     f"Note: Assumption that X$_{{{secondary_species},silicates}}$ $\\ll$ 1",
        #     fontsize=font_size_title,
        #     # y=1.03,            # default ≃0.98, smaller → more gap
        # )

        # plt.subplots_adjust(wspace=0.05, hspace=0.1)
        # plt.subplots_adjust(wspace=0.05)
    
        from collections.abc import Sequence

        from matplotlib.collections import PathCollection

        # --- Axes: remove empty collections & fix linewidths
        for ax in fig.axes:
            for coll in list(ax.collections):
                if isinstance(coll, PathCollection):
                    # remove truly empty offsets
                    if np.size(coll.get_offsets()) == 0:
                        coll.remove()
                        continue

                    lw = coll.get_linewidths()
                    # Normalize: scalar OK; list/array must be non-empty
                    if lw is None:
                        coll.set_linewidth(0.8)
                    elif np.isscalar(lw):
                        pass  # fine
                    elif isinstance(lw, Sequence):
                        if len(lw) == 0:
                            coll.set_linewidth(0.8)
                        # else: keep as-is
                    else:
                        # fallback: treat as array
                        if np.asarray(lw).size == 0:
                            coll.set_linewidth(0.8)

        # Legends too
        for leg in fig.legends:
            for h in leg.legend_handles:
                if isinstance(h, PathCollection):
                    lw = h.get_linewidths()
                    if lw is None or (not np.isscalar(lw) and len(np.atleast_1d(lw)) == 0):
                        h.set_linewidth(0.8)

        # for tick in ax.get_xticklabels() + ax.get_yticklabels():
        #     tick.set_fontweight("bold")

        # 3) Layout & save
        # plt.tight_layout()
        # plt.tight_layout(rect=[0, 0, 1, 1.0])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust layout to make room for the title
        plt.savefig(f"paper__KD_D_wt__vs_P_T_{secondary_species}.png", dpi=300)

        # pdf
        # plt.savefig(f"paper__KD_D_wt__vs_P_T_{secondary_species}.eps")
        plt.savefig(f"paper__KD_D_wt__vs_P_T_{secondary_species}.svg", format="svg", dpi=300)



























    if PLOT_MODE == 22 or PLOT_MODE < 0: # paper__KD_D_wt__vs_P_T__T__{secondary_species}

        # fig, axes_KD_D_wt__P = plt.subplots(2, 2, figsize=(12, 10))#, sharex=True, sharey=True)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        # ax1, ax_KD, ax3, ax_D_wt = axes.flatten()
        # ax_KD_low, ax_KD_high, ax_D_wt_low, ax_D_wt_high = axes_KD_D_wt__P.flatten()
    
        import matplotlib.gridspec as gridspec

        # sort df_superset by pressure and KD_sil_to_metal
        df_superset = df_superset.sort_values(by=["Target pressure (GPa)", "KD_sil_to_metal"])

        fig = plt.figure(figsize=(12, 12))


        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times"]
        # plt.rcParams.update({
        #     "font.weight": "medium",          # bold text everywhere
        #     "axes.labelweight": "medium",     # x/y labels
        #     "axes.titleweight": "medium",     # titles
        #     # "text.latex.preamble": r"\usepackage{sfmath}",  # bold math
        # })


        # all font sizes
        font_size_title = 14
        font_size_labels = 16
        font_size_ticks = 14
        font_size_legend = 12


        x_min = 1800
        x_mid = 17777
        x_max = 18000


        # Big grid: 2 rows, 1 col
        outer = gridspec.GridSpec(2, 1, height_ratios=[1, 0.02], hspace=0.2)

        # Top block: looser spacing
        gs_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0])#, 
                                                # width_ratios=[0.5, 0.5],
                                                # wspace=0.01)

        # Bottom block: tighter spacing
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1])


        # # 2 rows × 2 columns, with column widths 40% and 60%
        # gs = gridspec.GridSpec(2, 2, 
        #                     width_ratios=[0.4, 0.6],
        #                     height_ratios=[1, 1, 0.06],
        #                     hspace=0.15, wspace=0.05)

        # ax_KD_low = fig.add_subplot(gs_top[0, 0])  # Top-left (40%)
        # ax_KD_high = fig.add_subplot(gs_top[0, 1])  # Top-right (60%)
        # ax_D_wt_low = fig.add_subplot(gs_top[1, 0])  # Bottom-left (40%)
        # ax_D_wt_high = fig.add_subplot(gs_top[1, 1])  # Bottom-right (60%)
        ax_KD_low = fig.add_subplot(gs_top[0])  # Top (100%)
        ax_D_wt_low = fig.add_subplot(gs_top[1])  # Bottom (100%)



        marker_TI="o"  # marker for TI points
        marker_2phase="s"  # marker for two-phase points
        marker_other_comp_studies=["^", "D", "v", "<", ">"] # array of markers for other studies
        marker_other_expt_studies=["p", "H", "h", "*", ""]  # array of markers for other experimental studies

        marker_opts = dict(marker='o', linestyle='', markersize=10, alpha=1)#,color=base_color)
        marker_opts_scatter = dict(linestyle='', s=200, alpha=1,edgecolors='black', linewidths=1)
        marker_opts_error = dict(linestyle='', markersize=10, alpha=1, capsize=5, elinewidth=1)#, color='black',ecolor='black')
        marker_opts_scatter__others = dict(linestyle='', s=200, alpha=0.75)#, edgecolors='k',linewidths=0)#,color=base_color, alpha=0.5
        marker_opts_error__others = dict(linestyle='', markersize=10, capsize=5, elinewidth=1,alpha=0.75)#, color='black',ecolor='black')
        marker_opts_scatter__2phase = dict(linestyle='', s=400, alpha=0.9,edgecolors='black', linewidths=2)
        marker_opts_error__2phase = dict(linestyle='', markersize=10, alpha=0.9, capsize=5, elinewidth=2)
        # marker_opts_scatter__2phase_v2 = dict(linestyle='', s=200, alpha=1,edgecolors='black', linewidths=2)
        # marker_opts_error__2phase_v2 = dict(linestyle='', markersize=10, alpha=1, capsize=5, elinewidth=2)

        # add log temperature to df
        df_superset["log(Target temperature (K))"] = np.log10(df_superset["Target temperature (K)"])
        df_superset["log(Target pressure (GPa))"] = np.log10(df_superset["Target pressure (GPa)"])
        for data in datasets_comp:
            data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
            data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]
        for data in datasets_expt:
            data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
            data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]
        
        if secondary_species == "He":
            # data_He__two_phase_simulations["log(Target temperature (K))"] = [ np.log10(p) for p in data_He__two_phase_simulations["Target temperature (K)"] ]
            # data_He__two_phase_simulations["log(Target pressure (GPa))"] = [ np.log10(p) for p in data_He__two_phase_simulations["Target pressure (GPa)"] ]
            for data in datasets_2phases:
                data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
                data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]


        ######################################
        x_variable = "log(Target temperature (K))"
        # x_variable = "Target temperature (K)"  # x-axis variable for all plots
        # x_variable = "Target pressure (GPa)"  # x-axis variable for all plots
        # x_variable = "log(Target pressure (GPa))"  # x-axis variable for all plots
        # z_variable = "Target temperature (K)"  # z-axis variable for all plots -- color coding
        # z_variable = "log(Target temperature (K))"
        # z_variable = "Target pressure (GPa)"  # z-axis variable for all plots -- color coding
        z_variable = "log(Target pressure (GPa))"
        ######################################
        
        if x_variable == "log(Target pressure (GPa))":
            x_min = np.log10(5)
            x_mid = np.log10(200.0)
            x_max = np.log10(1100)

        if x_variable == "log(Target temperature (K))":
            x_min = np.log10(x_min)
            x_mid = np.log10(x_mid)
            x_max = np.log10(x_max)

        # create a colormap based on the temperature range
        if z_variable == "log(Target temperature (K))":
            temp_min = 1000  # minimum temperature in K
            temp_max = 16000  # maximum temperature in K
            log10_temp_min = np.log10(temp_min)  # minimum temperature in K
            log10_temp_max = np.log10(temp_max)  # maximum temperature in K
            cbar_ticks = np.log10([1000, 2000, 4000, 8000, 16000])
            cbar_labels = [1000, 2000, 4000, 8000, 16000]
            norm = plt.Normalize(
                vmin=log10_temp_min,
                vmax=log10_temp_max
            )
        elif z_variable == "Target temperature (K)":
            temp_min = 1000  # minimum temperature in K
            temp_max = 15000  # maximum temperature in K
            cbar_ticks = [1000, 3000, 6000, 9000, 12000, 15000]
            cbar_labels = [f"{int(t)}" for t in cbar_ticks]
            norm = plt.Normalize(
                vmin=temp_min,
                vmax=temp_max
            )

        # create a colormap based on the temperature range
        if z_variable == "log(Target pressure (GPa))":
            pres_min = 0.8  # minimum pressure in GPa
            pres_max = 1200  # maximum pressure in GPa
            log10_pres_min = np.log10(pres_min)  # minimum pressure in GPa
            log10_pres_max = np.log10(pres_max)  # maximum pressure in GPa
            cbar_ticks = np.log10([1, 10, 100, 1000])
            cbar_labels = [1, 10, 100, 1000]
            norm = plt.Normalize(
                vmin=log10_pres_min,
                vmax=log10_pres_max
            )
        elif z_variable == "Target pressure (GPa)":
            pres_min = 0.8  # minimum pressure in GPa
            pres_max = 1200  # maximum pressure in GPa
            cbar_ticks = [0, 200, 400, 600, 800, 1000]
            cbar_labels = [0, 200, 400, 600, 800, 1000]
            norm = plt.Normalize(
                vmin=pres_min,
                vmax=pres_max
            )


        magma = plt.get_cmap("magma")
        viridis = plt.get_cmap("viridis")
        coolwarm = plt.get_cmap("coolwarm")
        pastel_magma = pastel_cmap(magma, factor=0.05)  # tweak factor between 0 and 1
        pastel_viridis = pastel_cmap(viridis, factor=0.05)  # tweak factor between 0 and 1
        pastel_coolwarm = pastel_cmap(coolwarm, factor=0.05)  # tweak factor between 0 and 1
        cmap = pastel_viridis  # use pastel magma/viridis/... for the plots
        # cmap = pastel_magma  # use pastel magma/viridis/... for the plots
        cmap = pastel_coolwarm  # use pastel magma/viridis/... for the plots










        # for axes_low_or_high in ["low", "high"]:
        for axes_low_or_high in ["low"]:


            # KD = all first elements of each row from df_superset[f"array_KD_{secondary_species}"]
            KD = df_superset[f"array_KD_{secondary_species}"].str[0].to_numpy()
            KD_low = df_superset[f"array_KD_{secondary_species}_lower"].str[0].to_numpy()
            KD_high = df_superset[f"array_KD_{secondary_species}_upper"].str[0].to_numpy()

            # D_wt = all first elements of each row from df_superset[f"array_D_wt_{secondary_species}"]
            D_wt = df_superset[f"array_D_wt_{secondary_species}"].str[0].to_numpy()
            D_wt_low = df_superset[f"array_D_wt_{secondary_species}_lower"].str[0].to_numpy()
            D_wt_high = df_superset[f"array_D_wt_{secondary_species}_upper"].str[0].to_numpy()

            # x and z variables
            df_x_variable = df_superset[x_variable].values
            df_z_variable = df_superset[z_variable].values



            if axes_low_or_high == "low":
                ax_KD = ax_KD_low
                ax_D_wt = ax_D_wt_low

                # limit df_x_variable, df_z_variable and KD, D_wt, ... to df_x_variable < x_mid
                mask = np.asarray(df_x_variable) < x_mid
                df_x_variable = df_x_variable[mask]
                df_z_variable = df_z_variable[mask]
                KD = KD[mask]
                KD_low = KD_low[mask]
                KD_high = KD_high[mask]
                D_wt = D_wt[mask]
                D_wt_low = D_wt_low[mask]
                D_wt_high = D_wt_high[mask]

            elif axes_low_or_high == "high":
                ax_KD = ax_KD_high
                ax_D_wt = ax_D_wt_high

                # limit ... to > x_mid
                mask = np.asarray(df_x_variable) > x_mid
                # print(f"Number of points in high P plot: {len(mask)}, x_mid: {x_mid}, df_x_variable: {df_x_variable}")
                df_x_variable = df_x_variable[mask]
                df_z_variable = df_z_variable[mask]
                KD = KD[mask]
                KD_low = KD_low[mask]
                KD_high = KD_high[mask]
                D_wt = D_wt[mask]
                D_wt_low = D_wt_low[mask]
                D_wt_high = D_wt_high[mask]

            # --- Panel 3: KD_sil_to_metal (log y) ---

            # 1) Plot the colored points
            sc = ax_KD.scatter(
                df_x_variable,
                KD,
                c=df_z_variable,
                cmap=cmap,
                norm=norm,
                **marker_opts_scatter,
                marker=marker_TI,  # use the TI marker for scatter points
                label="This study (TI+AIMD)",
                # rasterized=True
            )

            # 2) Draw per-point errorbars with matching colors
            temps  = df_z_variable
            colors = cmap(norm(temps))

            for x0, y0, y_low, y_high, c in zip(
                df_x_variable,
                KD,
                KD_low,
                KD_high,
                # df_superset["KD_sil_to_metal"],
                # df_superset["KD_sil_to_metal_low"],
                # df_superset["KD_sil_to_metal_high"],
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
                df_x_variable,
                D_wt,
                c=df_z_variable,
                cmap=cmap,
                norm=norm,
                **marker_opts_scatter,
                marker=marker_TI,  # use the TI marker for scatter points
                label="This study (TI-AIMD)",
                # rasterized=True
            )
            # 2) Draw per-point errorbars with matching colors
            for x0, y0, y_low, y_high, c in zip(
                df_x_variable,
                D_wt,
                D_wt_low,
                D_wt_high,
                # df_superset["KD_sil_to_metal"],
                # df_superset["KD_sil_to_metal_low"],
                # df_superset["KD_sil_to_metal_high"],
                colors
            ):

                low  = y0 - y_low
                high = y_high - y0

                # shape (2,1) array: [[low], [high]]
                yerr = [[low], [high]]

                ax_D_wt.errorbar(
                    x0, y0,
                    yerr=yerr,
                    fmt='none',       # no extra marker
                    ecolor=c,         # single RGBA tuple
                    **marker_opts_error,
                    marker=marker_TI  # use the TI marker for errorbars
                )














            # # if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
            if secondary_species == "He":

            #     ax_KD.scatter(data_He__two_phase_simulations[x_variable], 
            #                     data_He__two_phase_simulations["KD"],
            #                     **marker_opts_scatter,
            #                     marker=marker_2phase,
            #                     c=data_He__two_phase_simulations[z_variable],
            #                     cmap=cmap,
            #                     norm=norm,
            #                     label="This study (2P-AIMD)",
            #                     # rasterized=True
            #                     )
            #     ax_D_wt.scatter(data_He__two_phase_simulations[x_variable], 
            #                     data_He__two_phase_simulations["D_wt"],
            #                     **marker_opts_scatter,
            #                     marker=marker_2phase,
            #                     c=data_He__two_phase_simulations[z_variable],
            #                     cmap=cmap,
            #                     norm=norm,
            #                     label="This study (2P-AIMD)",
            #                     # rasterized=True
            #                     )
                plot_studies(
                ax_KD,
                ax_D_wt,
                datasets_2phases,
                x_variable,
                z_variable,
                cmap,
                norm,
                marker_2phase,
                marker_opts_scatter__2phase,
                marker_opts_error__2phase,
                x_low=x_mid if axes_low_or_high=="high" else None,
                x_high=x_mid if axes_low_or_high=="low" else None
                )










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
            marker_opts_error__others,
            x_low=x_mid if axes_low_or_high=="high" else None,
            x_high=x_mid if axes_low_or_high=="low" else None
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
            marker_opts_error__others,
            x_low=x_mid if axes_low_or_high=="high" else None,
            x_high=x_mid if axes_low_or_high=="low" else None
            )
            ###########################








            # ***************************
            # ***************************
            if FIT_MODE == 1:
                # combine T and P from datasets_expt, datasets_comp and df dataframe
                sources = list(datasets_expt) + list(datasets_comp) + [df]  # uncomment if needed
                # array_T = row[f"array_T_{secondary_species}"] # z_variable -- T
                # array_P = row[f"array_P_{secondary_species}"] # x_variable -- P
                array_T          = _concat_cols(sources, "Target temperature (K)")
                array_P          = _concat_cols(sources, "Target pressure (GPa)")

                if secondary_species == "He":
                    # array_x_axis = array_X
                    array_X = (array_T ** 0.) * 1e-3
                    secondary_species_label = "He"

                elif secondary_species == "H":
                    # array_x_axis = array_X2
                    array_X2 = array_X = (array_T ** 0.) * 1e-3
                    secondary_species_label = "H2"

                array_x_axis = array_P

                x0 = np.log(array_X)
                x1 = array_P
                x2 = array_T

                if secondary_species == "H":
                    # plot the best fit line
                    y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
                elif secondary_species == "He":
                    y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
                # print(f"Best fit line for KD vs P, T: {y_fit}")
                # print(f"logX: {x0}")
                # print(f"T: {array_T}")
                # print(f"P: {array_P}")
                ax_KD.plot(
                    array_x_axis, y_fit,
                    linestyle='',
                    marker="s",
                    label=f"Best fit model",
                    color='black', markersize=10,
                    alpha=0.15
                )
            # ***************************
            # ***************************





        # show 1 colorbar for both KD and D_wt plots
        # 1) Create a colorbar for the temperature range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # only needed for older matplotlib versions
        # 2) Add the colorbar to the figure -- width == width of 1 plot
        # cbar = fig.colorbar(sm, ax=[ax_D_wt_low, ax_D_wt_high],
        #                     orientation='horizontal',
        #                     # fraction=1, pad=0.04,
        #                     pad=100,  # space between colorbar and plot
        #                     ticks=np.linspace(temp_min, temp_max, 5),
        #                     location='bottom',  # 'top' or 'bottom'
        #                     shrink=1,      # shrink to 80% of the original size
        #                     aspect=50,     # thinner bar
        #                     )
        cax = fig.add_subplot(gs_bottom[0, :])
        # cax.axis("off")  # hide frame and ticks
        # cax.set_xlabel("Pressure (GPa)", labelpad=10)  # shared label
        cbar = fig.colorbar(
            sm, cax=cax, orientation='horizontal',
            ticks=np.linspace(pres_min, pres_max, 5)
        )
        cbar.set_label("Pressure (GPa)", fontsize=font_size_labels)#, rotation=270, labelpad=15)
        # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_labels)
        cbar.ax.tick_params(labelsize=font_size_ticks)
        # bold boundary
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")





        # # x lim ,  y lim

        # for axes in [ax_KD_low, ax_KD_high, ax_D_wt_low, ax_D_wt_high]:
        for axes in [ax_KD_low, ax_D_wt_low]:

            # draw horizontal dashed line at y=1
            axes.axhline(y=1, color='gray', linestyle='-', linewidth=7.5, alpha=0.35)
            # annotate on top and immediately below it, on the right end side
            if axes in [ax_KD_low, ax_D_wt_low]:
                axes.annotate("Lithophile", xy=(0, 1), xytext=(-7, 0.6),
                            fontsize=12, ha='left', color='dimgray')
                axes.annotate("Siderophile", xy=(0, 1), xytext=(-7, 1.3),
                            fontsize=12, ha='left', color='dimgray')

            # if secondary_species == "He":
            #     y_min = 1e-5 # 1e-5
            #     y_max = 1e1 #1e1
            # elif secondary_species == "H":
            #     y_min = 1e-3 #1e-3
            #     y_max = 1e6
            axes.set_ylim(y_min, y_max)
            if axes in [ax_KD_low, ax_D_wt_low]:
                axes.set_xlim(x_min, x_mid)
            else:
                axes.set_xlim(x_mid, x_max)
            axes.set_yscale("log")
            axes.grid(True)

            # for bottom axes, set 1 x label
            # if axes in [ax_D_wt_low, ax_D_wt_high]:
            #     axes.set_xlabel("Pressure (GPa)")
            fig.supxlabel("Temperature (K)", y=0.15, fontsize=font_size_labels)

            # for left axes, set y labels
            if axes in [ax_KD_low]:
                axes.set_ylabel(r"Equilibrium Constant ($K_{D}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")
            elif axes in [ax_D_wt_low]:
                axes.set_ylabel(r"Partition Coefficient ($D_{wt}^{\mathrm{Fe}/\mathrm{MgSiO}_{3}}$)")

            # no y tick labels for right panels
            # if axes in [ax_KD_high, ax_D_wt_high]:
            #     axes.set_yticklabels([])

            axes.tick_params(axis='y', which='both', direction='in', pad=15)

            # x axis ticks and labels
            x_ticks = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
            x_tick_labels = [f"{tick}" for tick in x_ticks]
            if x_variable == "log(Target temperature (K))":
                x_ticks = np.log10( [2000, 4000, 8000, 16000] )
                x_tick_labels = [f"{tick}" for tick in [2000, 4000, 8000, 16000]]
            axes.set_xticks(x_ticks)
            axes.set_xticklabels(x_tick_labels)
            # axes.tick_params(axis='x', which='both', direction='in', pad=10)

            # move ticks to right side on right panels
            # if axes in [ax_KD_high, ax_D_wt_high]:
            #     axes.yaxis.tick_right()                     # move ticks to right side
            #     # axes.yaxis.set_label_position("right")      # move y-axis label too
            #     axes.tick_params(axis='y', which='both', direction='in', pad=5)  # adjust tick direction/padding

            # font sizes
            for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                label.set_fontsize(font_size_ticks)
            axes.set_xlabel(axes.get_xlabel(), fontsize=font_size_labels)
            axes.set_ylabel(axes.get_ylabel(), fontsize=font_size_labels)

            # bold boundaries
            for spine in axes.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("black")

        # Legend
        # 1) Grab handles & labels from one of the axes
        # handles, labels = ax_KD_low.get_legend_handles_labels()
        # Collect from both axes
        handles1, labels1 = ax_KD_low.get_legend_handles_labels()
        # handles2, labels2 = ax_KD_high.get_legend_handles_labels()

        # Merge
        # handles = handles2 + handles1
        # labels  = labels2 + labels1
        handles = handles1
        labels  = labels1

        # Deduplicate while keeping order
        unique = dict(zip(labels, handles))   # keys=labels, values=handles
        handles, labels = unique.values(), unique.keys()

        # 2) Create a single legend on the right side of the figure
        fig.legend(
            handles,
            labels,
            loc='lower center',        # center vertically on the right edge
            fontsize=font_size_legend,
            # borderaxespad=0.1,
            bbox_to_anchor=(0.50, 0.9),
            # frameon=False,
            ncol=3,  # number of columns in the legend
            # mode='expand',  # 'expand' to fill the space, 'none'
            labelspacing=1.,    # default is 0.5; larger → more space
            handletextpad=1.0,   # default is 0.8; larger → more space between handle & text
            columnspacing=2.0,   # if you have multiple columns, space between them
        )

        # 3) Shrink the subplots to make room for the legend
        # fig.subplots_adjust(right=0.82)  # push the plot area leftward






        # fig.subplots_adjust(top=0.88)

        # fig.suptitle(
        #     f"Equilibrium Constant ($K_D$) and Partition Coefficient ($D_{{wt}}$) "
        #     f"for the reaction {secondary_species}$_{{silicates}}$ $\\rightleftharpoons$ {secondary_species}$_{{metal}}$.\n"
        #     f"Note: Assumption that X$_{{{secondary_species},silicates}}$ $\\ll$ 1",
        #     fontsize=font_size_title,
        #     # y=1.03,            # default ≃0.98, smaller → more gap
        # )

        # plt.subplots_adjust(wspace=0.05, hspace=0.1)
        # plt.subplots_adjust(wspace=0.05)
    
        from collections.abc import Sequence

        from matplotlib.collections import PathCollection

        # --- Axes: remove empty collections & fix linewidths
        for ax in fig.axes:
            for coll in list(ax.collections):
                if isinstance(coll, PathCollection):
                    # remove truly empty offsets
                    if np.size(coll.get_offsets()) == 0:
                        coll.remove()
                        continue

                    lw = coll.get_linewidths()
                    # Normalize: scalar OK; list/array must be non-empty
                    if lw is None:
                        coll.set_linewidth(0.8)
                    elif np.isscalar(lw):
                        pass  # fine
                    elif isinstance(lw, Sequence):
                        if len(lw) == 0:
                            coll.set_linewidth(0.8)
                        # else: keep as-is
                    else:
                        # fallback: treat as array
                        if np.asarray(lw).size == 0:
                            coll.set_linewidth(0.8)

        # Legends too
        for leg in fig.legends:
            for h in leg.legend_handles:
                if isinstance(h, PathCollection):
                    lw = h.get_linewidths()
                    if lw is None or (not np.isscalar(lw) and len(np.atleast_1d(lw)) == 0):
                        h.set_linewidth(0.8)

        # for tick in ax.get_xticklabels() + ax.get_yticklabels():
        #     tick.set_fontweight("bold")

        # 3) Layout & save
        # plt.tight_layout()
        # plt.tight_layout(rect=[0, 0, 1, 1.0])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust layout to make room for the title
        plt.savefig(f"paper__KD_D_wt__vs_P_T__T__{secondary_species}.png", dpi=300)

        # pdf
        # plt.savefig(f"paper__KD_D_wt__vs_P_T_{secondary_species}.eps")
        plt.savefig(f"paper__KD_D_wt__vs_P_T__T__{secondary_species}.svg", format="svg", dpi=300)
























    if PLOT_MODE == 31 or PLOT_MODE < 0: # paper__KD_D_wt__vs_P_T_{secondary_species} -- same as 21 but w vertical cbar

        # fig, axes_KD_D_wt__P = plt.subplots(2, 2, figsize=(12, 10))#, sharex=True, sharey=True)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        # ax1, ax_KD, ax3, ax_D_wt = axes.flatten()
        # ax_KD_low, ax_KD_high, ax_D_wt_low, ax_D_wt_high = axes_KD_D_wt__P.flatten()
    
        import matplotlib.gridspec as gridspec

        # sort df_superset by pressure and KD_sil_to_metal
        df_superset = df_superset.sort_values(by=["Target pressure (GPa)", "KD_sil_to_metal"])

        fig = plt.figure(figsize=(12, 12))


        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times"]
        # plt.rcParams.update({
        #     "font.weight": "medium",          # bold text everywhere
        #     "axes.labelweight": "medium",     # x/y labels
        #     "axes.titleweight": "medium",     # titles
        #     # "text.latex.preamble": r"\usepackage{sfmath}",  # bold math
        # })


        # all font sizes
        font_size_title = 14
        font_size_labels = 16
        font_size_ticks = 14
        font_size_legend = 12


        x_min = -8
        x_mid = 70
        x_max = 1071


        # Big grid: 2 rows, 1 col
        outer = gridspec.GridSpec(1, 2, width_ratios=[1, 0.02], wspace=0.01)

        # Top block: looser spacing
        gs_top = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], 
                                                width_ratios=[0.5, 0.5],
                                                wspace=0.01)

        # Bottom block: tighter spacing
        gs_bottom = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1])


        # # 2 rows × 2 columns, with column widths 40% and 60%
        # gs = gridspec.GridSpec(2, 2, 
        #                     width_ratios=[0.4, 0.6],
        #                     height_ratios=[1, 1, 0.06],
        #                     hspace=0.15, wspace=0.05)

        ax_KD_low = fig.add_subplot(gs_top[0, 0])  # Top-left (40%)
        ax_KD_high = fig.add_subplot(gs_top[0, 1])  # Top-right (60%)
        ax_D_wt_low = fig.add_subplot(gs_top[1, 0])  # Bottom-left (40%)
        ax_D_wt_high = fig.add_subplot(gs_top[1, 1])  # Bottom-right (60%)



        marker_TI="o"  # marker for TI points
        marker_2phase="s"  # marker for two-phase points
        marker_other_comp_studies=["^", "D", "v", "<", ">"] # array of markers for other studies
        marker_other_expt_studies=["p", "H", "h", "*", ""]  # array of markers for other experimental studies

        marker_opts = dict(marker='o', linestyle='', markersize=10, alpha=1)#,color=base_color)
        marker_opts_scatter = dict(linestyle='', s=200, alpha=1,edgecolors='black', linewidths=1)
        marker_opts_error = dict(linestyle='', markersize=10, alpha=1, capsize=5, elinewidth=1)#, color='black',ecolor='black')
        marker_opts_scatter__others = dict(linestyle='', s=200, alpha=0.5)#, edgecolors='k',linewidths=0)#,color=base_color, alpha=0.5
        marker_opts_error__others = dict(linestyle='', markersize=10, capsize=5, elinewidth=1,alpha=0.5)#, color='black',ecolor='black')
        marker_opts_scatter__2phase = dict(linestyle='', s=400, alpha=0.8,edgecolors='black', linewidths=2)
        marker_opts_error__2phase = dict(linestyle='', markersize=10, alpha=0.8, capsize=5, elinewidth=2)
        # marker_opts_scatter__2phase_v2 = dict(linestyle='', s=200, alpha=1,edgecolors='black', linewidths=2)
        # marker_opts_error__2phase_v2 = dict(linestyle='', markersize=10, alpha=1, capsize=5, elinewidth=2)

        # add log temperature to df
        df_superset["log(Target temperature (K))"] = np.log10(df_superset["Target temperature (K)"])
        df_superset["log(Target pressure (GPa))"] = np.log10(df_superset["Target pressure (GPa)"])
        for data in datasets_comp:
            data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
            data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]
        for data in datasets_expt:
            data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
            data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]

        if secondary_species == "He" or secondary_species == "H":
            # data_He__two_phase_simulations["log(Target temperature (K))"] = [ np.log10(p) for p in data_He__two_phase_simulations["Target temperature (K)"] ]
            # data_He__two_phase_simulations["log(Target pressure (GPa))"] = [ np.log10(p) for p in data_He__two_phase_simulations["Target pressure (GPa)"] ]
            for data in datasets_2phases:
                data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
                data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]


        ######################################
        x_variable = "Target pressure (GPa)"  # x-axis variable for all plots
        # x_variable = "log(Target pressure (GPa))"  # x-axis variable for all plots
        # z_variable = "Target temperature (K)"  # z-axis variable for all plots -- color coding
        z_variable = "log(Target temperature (K))"
        ######################################
        
        if x_variable == "log(Target pressure (GPa))":
            x_min = np.log10(5)
            x_mid = np.log10(200.0)
            x_max = np.log10(1100)

        # create a colormap based on the temperature range
        if z_variable == "log(Target temperature (K))":
            temp_min = 1000  # minimum temperature in K
            temp_max = 16000  # maximum temperature in K
            log10_temp_min = np.log10(temp_min)  # minimum temperature in K
            log10_temp_max = np.log10(temp_max)  # maximum temperature in K
            cbar_ticks = np.log10([1000, 2000, 4000, 8000, 16000])
            cbar_labels = [1000, 2000, 4000, 8000, 16000]
            norm = plt.Normalize(
                vmin=log10_temp_min,
                vmax=log10_temp_max
            )
        elif z_variable == "Target temperature (K)":
            temp_min = 1000  # minimum temperature in K
            temp_max = 15000  # maximum temperature in K
            cbar_ticks = [1000, 3000, 6000, 9000, 12000, 15000]
            cbar_labels = [f"{int(t)}" for t in cbar_ticks]
            norm = plt.Normalize(
                vmin=temp_min,
                vmax=temp_max
            )


        magma = plt.get_cmap("magma")
        viridis = plt.get_cmap("viridis")
        coolwarm = plt.get_cmap("coolwarm")
        pastel_magma = pastel_cmap(magma, factor=0.25)  # tweak factor between 0 and 1
        pastel_viridis = pastel_cmap(viridis, factor=0.05)  # tweak factor between 0 and 1
        pastel_coolwarm = pastel_cmap(coolwarm, factor=0.05)  # tweak factor between 0 and 1
        cmap = pastel_viridis  # use pastel magma/viridis/... for the plots
        cmap = pastel_magma  # use pastel magma/viridis/... for the plots
        # cmap = pastel_coolwarm  # use pastel magma/viridis/... for the plots










        for axes_low_or_high in ["low", "high"]:


            # KD = all first elements of each row from df_superset[f"array_KD_{secondary_species}"]
            KD = df_superset[f"array_KD_{secondary_species}"].str[0].to_numpy()
            KD_low = df_superset[f"array_KD_{secondary_species}_lower"].str[0].to_numpy()
            KD_high = df_superset[f"array_KD_{secondary_species}_upper"].str[0].to_numpy()

            # D_wt = all first elements of each row from df_superset[f"array_D_wt_{secondary_species}"]
            D_wt = df_superset[f"array_D_wt_{secondary_species}"].str[0].to_numpy()
            D_wt_low = df_superset[f"array_D_wt_{secondary_species}_lower"].str[0].to_numpy()
            D_wt_high = df_superset[f"array_D_wt_{secondary_species}_upper"].str[0].to_numpy()

            # x and z variables
            df_x_variable = df_superset[x_variable].values
            df_z_variable = df_superset[z_variable].values



            if axes_low_or_high == "low":
                ax_KD = ax_KD_low
                ax_D_wt = ax_D_wt_low

                # limit df_x_variable, df_z_variable and KD, D_wt, ... to df_x_variable < x_mid
                mask = np.asarray(df_x_variable) < x_mid
                df_x_variable = df_x_variable[mask]
                df_z_variable = df_z_variable[mask]
                KD = KD[mask]
                KD_low = KD_low[mask]
                KD_high = KD_high[mask]
                D_wt = D_wt[mask]
                D_wt_low = D_wt_low[mask]
                D_wt_high = D_wt_high[mask]

            elif axes_low_or_high == "high":
                ax_KD = ax_KD_high
                ax_D_wt = ax_D_wt_high

                # limit ... to > x_mid
                mask = np.asarray(df_x_variable) > x_mid
                # print(f"Number of points in high P plot: {len(mask)}, x_mid: {x_mid}, df_x_variable: {df_x_variable}")
                df_x_variable = df_x_variable[mask]
                df_z_variable = df_z_variable[mask]
                KD = KD[mask]
                KD_low = KD_low[mask]
                KD_high = KD_high[mask]
                D_wt = D_wt[mask]
                D_wt_low = D_wt_low[mask]
                D_wt_high = D_wt_high[mask]

            # --- Panel 3: KD_sil_to_metal (log y) ---

            # 1) Plot the colored points
            sc = ax_KD.scatter(
                df_x_variable,
                KD,
                c=df_z_variable,
                cmap=cmap,
                norm=norm,
                **marker_opts_scatter,
                marker=marker_TI,  # use the TI marker for scatter points
                label="This study (TI+AIMD)",
                # rasterized=True
            )

            # 2) Draw per-point errorbars with matching colors
            temps  = df_z_variable
            colors = cmap(norm(temps))

            for x0, y0, y_low, y_high, c in zip(
                df_x_variable,
                KD,
                KD_low,
                KD_high,
                # df_superset["KD_sil_to_metal"],
                # df_superset["KD_sil_to_metal_low"],
                # df_superset["KD_sil_to_metal_high"],
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
                df_x_variable,
                D_wt,
                c=df_z_variable,
                cmap=cmap,
                norm=norm,
                **marker_opts_scatter,
                marker=marker_TI,  # use the TI marker for scatter points
                label="This study (TI-AIMD)",
                # rasterized=True
            )
            # 2) Draw per-point errorbars with matching colors
            for x0, y0, y_low, y_high, c in zip(
                df_x_variable,
                D_wt,
                D_wt_low,
                D_wt_high,
                # df_superset["KD_sil_to_metal"],
                # df_superset["KD_sil_to_metal_low"],
                # df_superset["KD_sil_to_metal_high"],
                colors
            ):

                low  = y0 - y_low
                high = y_high - y0

                # shape (2,1) array: [[low], [high]]
                yerr = [[low], [high]]

                ax_D_wt.errorbar(
                    x0, y0,
                    yerr=yerr,
                    fmt='none',       # no extra marker
                    ecolor=c,         # single RGBA tuple
                    **marker_opts_error,
                    marker=marker_TI  # use the TI marker for errorbars
                )














            # # if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
            if secondary_species == "He" or secondary_species == "H":

            #     ax_KD.scatter(data_He__two_phase_simulations[x_variable], 
            #                     data_He__two_phase_simulations["KD"],
            #                     **marker_opts_scatter,
            #                     marker=marker_2phase,
            #                     c=data_He__two_phase_simulations[z_variable],
            #                     cmap=cmap,
            #                     norm=norm,
            #                     label="This study (2P-AIMD)",
            #                     # rasterized=True
            #                     )
            #     ax_D_wt.scatter(data_He__two_phase_simulations[x_variable], 
            #                     data_He__two_phase_simulations["D_wt"],
            #                     **marker_opts_scatter,
            #                     marker=marker_2phase,
            #                     c=data_He__two_phase_simulations[z_variable],
            #                     cmap=cmap,
            #                     norm=norm,
            #                     label="This study (2P-AIMD)",
            #                     # rasterized=True
            #                     )
                plot_studies(
                ax_KD,
                ax_D_wt,
                datasets_2phases,
                x_variable,
                z_variable,
                cmap,
                norm,
                marker_2phase,
                marker_opts_scatter__2phase,
                marker_opts_error__2phase,
                x_low=x_mid if axes_low_or_high=="high" else None,
                x_high=x_mid if axes_low_or_high=="low" else None
                )










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
            marker_opts_error__others,
            x_low=x_mid if axes_low_or_high=="high" else None,
            x_high=x_mid if axes_low_or_high=="low" else None
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
            marker_opts_error__others,
            x_low=x_mid if axes_low_or_high=="high" else None,
            x_high=x_mid if axes_low_or_high=="low" else None
            )
            ###########################








            # ***************************
            # ***************************
            if FIT_MODE == 1:
                # combine T and P from datasets_expt, datasets_comp and df dataframe
                sources = list(datasets_expt) + list(datasets_comp) + [df]  # uncomment if needed
                # array_T = row[f"array_T_{secondary_species}"] # z_variable -- T
                # array_P = row[f"array_P_{secondary_species}"] # x_variable -- P
                array_T          = _concat_cols(sources, "Target temperature (K)")
                array_P          = _concat_cols(sources, "Target pressure (GPa)")

                if secondary_species == "He":
                    # array_x_axis = array_X
                    array_X = (array_T ** 0.) * 1e-3
                    secondary_species_label = "He"

                elif secondary_species == "H":
                    # array_x_axis = array_X2
                    array_X2 = array_X = (array_T ** 0.) * 1e-3
                    secondary_species_label = "H2"

                array_x_axis = array_P

                x0 = np.log(array_X)
                x1 = array_P
                x2 = array_T

                if secondary_species == "H":
                    # plot the best fit line
                    y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
                elif secondary_species == "He":
                    y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
                # print(f"Best fit line for KD vs P, T: {y_fit}")
                # print(f"logX: {x0}")
                # print(f"T: {array_T}")
                # print(f"P: {array_P}")
                ax_KD.plot(
                    array_x_axis, y_fit,
                    linestyle='',
                    marker="s",
                    label=f"Best fit model",
                    color='black', markersize=10,
                    alpha=0.15
                )
            # ***************************
            # ***************************





        # show 1 colorbar for both KD and D_wt plots
        # 1) Create a colorbar for the temperature range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # only needed for older matplotlib versions
        # 2) Add the colorbar to the figure -- width == width of 1 plot
        # cbar = fig.colorbar(sm, ax=[ax_D_wt_low, ax_D_wt_high],
        #                     orientation='horizontal',
        #                     # fraction=1, pad=0.04,
        #                     pad=100,  # space between colorbar and plot
        #                     ticks=np.linspace(temp_min, temp_max, 5),
        #                     location='bottom',  # 'top' or 'bottom'
        #                     shrink=1,      # shrink to 80% of the original size
        #                     aspect=50,     # thinner bar
        #                     )
        for gs_bottom_i in [gs_bottom[0], gs_bottom[1]]:
            cax = fig.add_subplot(gs_bottom_i)
            # cax.axis("off")  # hide frame and ticks
            # cax.set_xlabel("Pressure (GPa)", labelpad=10)  # shared label
            cbar = fig.colorbar(
                sm, cax=cax, orientation='vertical',
                ticks=np.linspace(temp_min, temp_max, 5),
                # shrink=0.9,
                # aspect=10
            )
            
            cbar.set_label("Temperature (K)", fontsize=font_size_labels)#, rotation=270, labelpad=15)
            # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_labels)
            cbar.ax.tick_params(labelsize=font_size_ticks)
            # bold boundary
            for spine in cbar.ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("black")

            pos = cax.get_position()
            cax.set_position([pos.x0, pos.y0, pos.width*1.25, pos.height])  # 1.5x wider





        # # x lim ,  y lim

        for axes in [ax_KD_low, ax_KD_high, ax_D_wt_low, ax_D_wt_high]:

            # draw horizontal dashed line at y=1
            axes.axhline(y=1, color='gray', linestyle='-', linewidth=7.5, alpha=0.35)
            # annotate on top and immediately below it, on the right end side
            if axes in [ax_KD_low, ax_D_wt_low]:
                axes.annotate("Lithophile", xy=(0, 1), xytext=(-7, 0.6),
                            fontsize=12, ha='left', color='dimgray')
                axes.annotate("Siderophile", xy=(0, 1), xytext=(-7, 1.3),
                            fontsize=12, ha='left', color='dimgray')

            # if secondary_species == "He":
            #     y_min = 1e-5 # 1e-5
            #     y_max = 1e1 #1e1
            # elif secondary_species == "H":
            #     y_min = 1e-3 #1e-3
            #     y_max = 1e6
            axes.set_ylim(y_min, y_max)
            if axes in [ax_KD_low, ax_D_wt_low]:
                axes.set_xlim(x_min, x_mid)
            else:
                axes.set_xlim(x_mid, x_max)
            axes.set_yscale("log")
            axes.grid(True)

            # for bottom axes, set 1 x label
            # if axes in [ax_D_wt_low, ax_D_wt_high]:
            #     axes.set_xlabel("Pressure (GPa)")
            fig.supxlabel(f"Pressure (GPa)", y=0.055, x=0.475,fontsize=font_size_labels)
            # fig.supxlabel("Pressure (GPa)", fontsize=font_size_labels)

            # for left axes, set y labels
            if axes in [ax_KD_low]:
                axes.set_ylabel(r"Equilibrium Constant ($K_{D}^{\mathrm{met}/\mathrm{sil}}$)")
            elif axes in [ax_D_wt_low]:
                if secondary_species == "H":
                    axes.set_ylabel(r"Partition Coefficient ($D_\mathrm{H}^{\mathrm{met}/\mathrm{sil}}$)")
                elif secondary_species == "He":
                    axes.set_ylabel(r"Partition Coefficient ($D_\mathrm{He}^{\mathrm{met}/\mathrm{sil}}$)")

            # no y tick labels for right panels
            if axes in [ax_KD_high, ax_D_wt_high]:
                axes.set_yticklabels([])

            axes.tick_params(axis='y', which='both', direction='in', pad=15)

            # move ticks to right side on right panels
            if axes in [ax_KD_high, ax_D_wt_high]:
                axes.yaxis.tick_right()                     # move ticks to right side
                # axes.yaxis.set_label_position("right")      # move y-axis label too
                axes.tick_params(axis='y', which='both', direction='in', pad=5)  # adjust tick direction/padding

            # font sizes
            for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                label.set_fontsize(font_size_ticks)
            axes.set_xlabel(axes.get_xlabel(), fontsize=font_size_labels)
            axes.set_ylabel(axes.get_ylabel(), fontsize=font_size_labels)

            # bold boundaries
            for spine in axes.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("black")

        # Legend
        # 1) Grab handles & labels from one of the axes
        # handles, labels = ax_KD_low.get_legend_handles_labels()
        # Collect from both axes
        handles1, labels1 = ax_KD_low.get_legend_handles_labels()
        handles2, labels2 = ax_KD_high.get_legend_handles_labels()

        # Merge
        handles = handles2 + handles1
        labels  = labels2 + labels1

        # Deduplicate while keeping order
        unique = dict(zip(labels, handles))   # keys=labels, values=handles
        handles, labels = unique.values(), unique.keys()

        if secondary_species == "He":
            ncol_legend = 4
        elif secondary_species == "H":
            ncol_legend = 4

        # 2) Create a single legend on the right side of the figure
        fig.legend(
            handles,
            labels,
            loc='lower center',        # center vertically on the right edge
            fontsize=font_size_legend,
            # borderaxespad=0.1,
            bbox_to_anchor=(0.50, 0.9),
            # frameon=False,
            ncol=ncol_legend,  # number of columns in the legend
            # mode='expand',  # 'expand' to fill the space, 'none'
            labelspacing=1.,    # default is 0.5; larger → more space
            handletextpad=1.0,   # default is 0.8; larger → more space between handle & text
            columnspacing=2,   # if you have multiple columns, space between them
            handlelength=1.0,
            markerscale=0.75             # shrink markers to ... % in legend
        )

        # 3) Shrink the subplots to make room for the legend
        # fig.subplots_adjust(right=0.82)  # push the plot area leftward






        # fig.subplots_adjust(top=0.88)

        # fig.suptitle(
        #     f"Equilibrium Constant ($K_D$) and Partition Coefficient ($D_{{wt}}$) "
        #     f"for the reaction {secondary_species}$_{{silicates}}$ $\\rightleftharpoons$ {secondary_species}$_{{metal}}$.\n"
        #     f"Note: Assumption that X$_{{{secondary_species},silicates}}$ $\\ll$ 1",
        #     fontsize=font_size_title,
        #     # y=1.03,            # default ≃0.98, smaller → more gap
        # )

        # plt.subplots_adjust(wspace=0.05, hspace=0.1)
        # plt.subplots_adjust(wspace=0.05)
    
        from collections.abc import Sequence

        from matplotlib.collections import PathCollection

        # --- Axes: remove empty collections & fix linewidths
        for ax in fig.axes:
            for coll in list(ax.collections):
                if isinstance(coll, PathCollection):
                    # remove truly empty offsets
                    if np.size(coll.get_offsets()) == 0:
                        coll.remove()
                        continue

                    lw = coll.get_linewidths()
                    # Normalize: scalar OK; list/array must be non-empty
                    if lw is None:
                        coll.set_linewidth(0.8)
                    elif np.isscalar(lw):
                        pass  # fine
                    elif isinstance(lw, Sequence):
                        if len(lw) == 0:
                            coll.set_linewidth(0.8)
                        # else: keep as-is
                    else:
                        # fallback: treat as array
                        if np.asarray(lw).size == 0:
                            coll.set_linewidth(0.8)

        # Legends too
        for leg in fig.legends:
            for h in leg.legend_handles:
                if isinstance(h, PathCollection):
                    lw = h.get_linewidths()
                    if lw is None or (not np.isscalar(lw) and len(np.atleast_1d(lw)) == 0):
                        h.set_linewidth(0.8)

        # for tick in ax.get_xticklabels() + ax.get_yticklabels():
        #     tick.set_fontweight("bold")

        # 3) Layout & save
        # plt.tight_layout()
        # plt.tight_layout(rect=[0, 0, 1, 1.0])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust layout to make room for the title
        plt.savefig(f"paper__KD_D_wt__vs_P_T_{secondary_species}__v_cbar.png", dpi=300)

        # pdf
        # plt.savefig(f"paper__KD_D_wt__vs_P_T_{secondary_species}.eps")
        plt.savefig(f"paper__KD_D_wt__vs_P_T_{secondary_species}__v_cbar.svg", format="svg", dpi=300)



























    if PLOT_MODE == 32 or PLOT_MODE < 0: # paper__KD_D_wt__vs_P_T__T__{secondary_species} -- same as 22 but w vertical cbar

        # fig, axes_KD_D_wt__P = plt.subplots(2, 2, figsize=(12, 10))#, sharex=True, sharey=True)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        # ax1, ax_KD, ax3, ax_D_wt = axes.flatten()
        # ax_KD_low, ax_KD_high, ax_D_wt_low, ax_D_wt_high = axes_KD_D_wt__P.flatten()
    
        import matplotlib.gridspec as gridspec

        # sort df_superset by pressure and KD_sil_to_metal
        df_superset = df_superset.sort_values(by=["Target pressure (GPa)", "KD_sil_to_metal"])

        fig = plt.figure(figsize=(12, 12))


        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times"]
        # plt.rcParams.update({
        #     "font.weight": "medium",          # bold text everywhere
        #     "axes.labelweight": "medium",     # x/y labels
        #     "axes.titleweight": "medium",     # titles
        #     # "text.latex.preamble": r"\usepackage{sfmath}",  # bold math
        # })


        # all font sizes
        font_size_title = 14
        font_size_labels = 16
        font_size_ticks = 14
        font_size_legend = 12


        if secondary_species == "He":
            x_min = 1800
            x_mid = 17777
            x_max = 18000
        elif secondary_species == "H":
            x_min = 1400
            x_mid = 17777
            x_max = 18000


        # Big grid: 2 rows, 1 col
        outer = gridspec.GridSpec(1, 2, width_ratios=[1, 0.02], wspace=0.01)

        # Top block: looser spacing
        gs_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0])#, 
                                                # width_ratios=[0.5, 0.5],
                                                # wspace=0.01)

        # Bottom block: tighter spacing
        gs_bottom = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1])


        # # 2 rows × 2 columns, with column widths 40% and 60%
        # gs = gridspec.GridSpec(2, 2, 
        #                     width_ratios=[0.4, 0.6],
        #                     height_ratios=[1, 1, 0.06],
        #                     hspace=0.15, wspace=0.05)

        # ax_KD_low = fig.add_subplot(gs_top[0, 0])  # Top-left (40%)
        # ax_KD_high = fig.add_subplot(gs_top[0, 1])  # Top-right (60%)
        # ax_D_wt_low = fig.add_subplot(gs_top[1, 0])  # Bottom-left (40%)
        # ax_D_wt_high = fig.add_subplot(gs_top[1, 1])  # Bottom-right (60%)
        ax_KD_low = fig.add_subplot(gs_top[0])  # Top (100%)
        ax_D_wt_low = fig.add_subplot(gs_top[1])  # Bottom (100%)



        marker_TI="o"  # marker for TI points
        marker_2phase="s"  # marker for two-phase points
        marker_other_comp_studies=["^", "D", "v", "<", ">"] # array of markers for other studies
        marker_other_expt_studies=["p", "H", "h", "*", ""]  # array of markers for other experimental studies

        marker_opts = dict(marker='o', linestyle='', markersize=10, alpha=1)#,color=base_color)
        marker_opts_scatter = dict(linestyle='', s=200, alpha=1,edgecolors='black', linewidths=1)
        marker_opts_error = dict(linestyle='', markersize=10, alpha=1, capsize=5, elinewidth=1)#, color='black',ecolor='black')
        marker_opts_scatter__others = dict(linestyle='', s=200, alpha=0.75)#, edgecolors='k',linewidths=0)#,color=base_color, alpha=0.5
        marker_opts_error__others = dict(linestyle='', markersize=10, capsize=5, elinewidth=1,alpha=0.75)#, color='black',ecolor='black')
        marker_opts_scatter__2phase = dict(linestyle='', s=400, alpha=0.9,edgecolors='black', linewidths=2)
        marker_opts_error__2phase = dict(linestyle='', markersize=10, alpha=0.9, capsize=5, elinewidth=2)
        # marker_opts_scatter__2phase_v2 = dict(linestyle='', s=200, alpha=1,edgecolors='black', linewidths=2)
        # marker_opts_error__2phase_v2 = dict(linestyle='', markersize=10, alpha=1, capsize=5, elinewidth=2)

        # add log temperature to df
        df_superset["log(Target temperature (K))"] = np.log10(df_superset["Target temperature (K)"])
        df_superset["log(Target pressure (GPa))"] = np.log10(df_superset["Target pressure (GPa)"])
        for data in datasets_comp:
            data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
            data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]
        for data in datasets_expt:
            data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
            data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]
        
        if secondary_species == "He" or secondary_species == "H":
            # data_He__two_phase_simulations["log(Target temperature (K))"] = [ np.log10(p) for p in data_He__two_phase_simulations["Target temperature (K)"] ]
            # data_He__two_phase_simulations["log(Target pressure (GPa))"] = [ np.log10(p) for p in data_He__two_phase_simulations["Target pressure (GPa)"] ]
            for data in datasets_2phases:
                data["log(Target temperature (K))"] = [ np.log10(p) for p in data["Target temperature (K)"] ]
                data["log(Target pressure (GPa))"] = [ np.log10(p) for p in data["Target pressure (GPa)"] ]


        ######################################
        x_variable = "log(Target temperature (K))"
        # x_variable = "Target temperature (K)"  # x-axis variable for all plots
        # x_variable = "Target pressure (GPa)"  # x-axis variable for all plots
        # x_variable = "log(Target pressure (GPa))"  # x-axis variable for all plots
        # z_variable = "Target temperature (K)"  # z-axis variable for all plots -- color coding
        # z_variable = "log(Target temperature (K))"
        # z_variable = "Target pressure (GPa)"  # z-axis variable for all plots -- color coding
        z_variable = "log(Target pressure (GPa))"
        ######################################
        
        if x_variable == "log(Target pressure (GPa))":
            x_min = np.log10(5)
            x_mid = np.log10(200.0)
            x_max = np.log10(1100)

        if x_variable == "log(Target temperature (K))":
            x_min = np.log10(x_min)
            x_mid = np.log10(x_mid)
            x_max = np.log10(x_max)

        # create a colormap based on the temperature range
        if z_variable == "log(Target temperature (K))":
            temp_min = 1000  # minimum temperature in K
            temp_max = 16000  # maximum temperature in K
            log10_temp_min = np.log10(temp_min)  # minimum temperature in K
            log10_temp_max = np.log10(temp_max)  # maximum temperature in K
            cbar_ticks = np.log10([1000, 2000, 4000, 8000, 16000])
            cbar_labels = [1000, 2000, 4000, 8000, 16000]
            norm = plt.Normalize(
                vmin=log10_temp_min,
                vmax=log10_temp_max
            )
        elif z_variable == "Target temperature (K)":
            temp_min = 1000  # minimum temperature in K
            temp_max = 15000  # maximum temperature in K
            cbar_ticks = [1000, 3000, 6000, 9000, 12000, 15000]
            cbar_labels = [f"{int(t)}" for t in cbar_ticks]
            norm = plt.Normalize(
                vmin=temp_min,
                vmax=temp_max
            )

        # create a colormap based on the temperature range
        if z_variable == "log(Target pressure (GPa))":
            pres_min = 0.8  # minimum pressure in GPa
            pres_max = 1200  # maximum pressure in GPa
            log10_pres_min = np.log10(pres_min)  # minimum pressure in GPa
            log10_pres_max = np.log10(pres_max)  # maximum pressure in GPa
            cbar_ticks = np.log10([1, 10, 100, 1000])
            cbar_labels = [1, 10, 100, 1000]
            norm = plt.Normalize(
                vmin=log10_pres_min,
                vmax=log10_pres_max
            )
        elif z_variable == "Target pressure (GPa)":
            pres_min = 0.8  # minimum pressure in GPa
            pres_max = 1200  # maximum pressure in GPa
            cbar_ticks = [0, 200, 400, 600, 800, 1000]
            cbar_labels = [0, 200, 400, 600, 800, 1000]
            norm = plt.Normalize(
                vmin=pres_min,
                vmax=pres_max
            )


        magma = plt.get_cmap("magma")
        viridis = plt.get_cmap("viridis")
        coolwarm = plt.get_cmap("coolwarm")
        pastel_magma = pastel_cmap(magma, factor=0.05)  # tweak factor between 0 and 1
        pastel_viridis = pastel_cmap(viridis, factor=0.05)  # tweak factor between 0 and 1
        pastel_coolwarm = pastel_cmap(coolwarm, factor=0.05)  # tweak factor between 0 and 1
        cmap = pastel_viridis  # use pastel magma/viridis/... for the plots
        # cmap = pastel_magma  # use pastel magma/viridis/... for the plots
        cmap = pastel_coolwarm  # use pastel magma/viridis/... for the plots










        # for axes_low_or_high in ["low", "high"]:
        for axes_low_or_high in ["low"]:


            # KD = all first elements of each row from df_superset[f"array_KD_{secondary_species}"]
            KD = df_superset[f"array_KD_{secondary_species}"].str[0].to_numpy()
            KD_low = df_superset[f"array_KD_{secondary_species}_lower"].str[0].to_numpy()
            KD_high = df_superset[f"array_KD_{secondary_species}_upper"].str[0].to_numpy()

            # D_wt = all first elements of each row from df_superset[f"array_D_wt_{secondary_species}"]
            D_wt = df_superset[f"array_D_wt_{secondary_species}"].str[0].to_numpy()
            D_wt_low = df_superset[f"array_D_wt_{secondary_species}_lower"].str[0].to_numpy()
            D_wt_high = df_superset[f"array_D_wt_{secondary_species}_upper"].str[0].to_numpy()

            # x and z variables
            df_x_variable = df_superset[x_variable].values
            df_z_variable = df_superset[z_variable].values



            if axes_low_or_high == "low":
                ax_KD = ax_KD_low
                ax_D_wt = ax_D_wt_low

                # limit df_x_variable, df_z_variable and KD, D_wt, ... to df_x_variable < x_mid
                mask = np.asarray(df_x_variable) < x_mid
                df_x_variable = df_x_variable[mask]
                df_z_variable = df_z_variable[mask]
                KD = KD[mask]
                KD_low = KD_low[mask]
                KD_high = KD_high[mask]
                D_wt = D_wt[mask]
                D_wt_low = D_wt_low[mask]
                D_wt_high = D_wt_high[mask]

            elif axes_low_or_high == "high":
                ax_KD = ax_KD_high
                ax_D_wt = ax_D_wt_high

                # limit ... to > x_mid
                mask = np.asarray(df_x_variable) > x_mid
                # print(f"Number of points in high P plot: {len(mask)}, x_mid: {x_mid}, df_x_variable: {df_x_variable}")
                df_x_variable = df_x_variable[mask]
                df_z_variable = df_z_variable[mask]
                KD = KD[mask]
                KD_low = KD_low[mask]
                KD_high = KD_high[mask]
                D_wt = D_wt[mask]
                D_wt_low = D_wt_low[mask]
                D_wt_high = D_wt_high[mask]

            # --- Panel 3: KD_sil_to_metal (log y) ---

            # 1) Plot the colored points
            sc = ax_KD.scatter(
                df_x_variable,
                KD,
                c=df_z_variable,
                cmap=cmap,
                norm=norm,
                **marker_opts_scatter,
                marker=marker_TI,  # use the TI marker for scatter points
                label="This study (TI+AIMD)",
                # rasterized=True
            )

            # 2) Draw per-point errorbars with matching colors
            temps  = df_z_variable
            colors = cmap(norm(temps))

            for x0, y0, y_low, y_high, c in zip(
                df_x_variable,
                KD,
                KD_low,
                KD_high,
                # df_superset["KD_sil_to_metal"],
                # df_superset["KD_sil_to_metal_low"],
                # df_superset["KD_sil_to_metal_high"],
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
                df_x_variable,
                D_wt,
                c=df_z_variable,
                cmap=cmap,
                norm=norm,
                **marker_opts_scatter,
                marker=marker_TI,  # use the TI marker for scatter points
                label="This study (TI-AIMD)",
                # rasterized=True
            )
            # 2) Draw per-point errorbars with matching colors
            for x0, y0, y_low, y_high, c in zip(
                df_x_variable,
                D_wt,
                D_wt_low,
                D_wt_high,
                # df_superset["KD_sil_to_metal"],
                # df_superset["KD_sil_to_metal_low"],
                # df_superset["KD_sil_to_metal_high"],
                colors
            ):

                low  = y0 - y_low
                high = y_high - y0

                # shape (2,1) array: [[low], [high]]
                yerr = [[low], [high]]

                ax_D_wt.errorbar(
                    x0, y0,
                    yerr=yerr,
                    fmt='none',       # no extra marker
                    ecolor=c,         # single RGBA tuple
                    **marker_opts_error,
                    marker=marker_TI  # use the TI marker for errorbars
                )














            # # if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
            if secondary_species == "He" or secondary_species == "H":

            #     ax_KD.scatter(data_He__two_phase_simulations[x_variable], 
            #                     data_He__two_phase_simulations["KD"],
            #                     **marker_opts_scatter,
            #                     marker=marker_2phase,
            #                     c=data_He__two_phase_simulations[z_variable],
            #                     cmap=cmap,
            #                     norm=norm,
            #                     label="This study (2P-AIMD)",
            #                     # rasterized=True
            #                     )
            #     ax_D_wt.scatter(data_He__two_phase_simulations[x_variable], 
            #                     data_He__two_phase_simulations["D_wt"],
            #                     **marker_opts_scatter,
            #                     marker=marker_2phase,
            #                     c=data_He__two_phase_simulations[z_variable],
            #                     cmap=cmap,
            #                     norm=norm,
            #                     label="This study (2P-AIMD)",
            #                     # rasterized=True
            #                     )
                plot_studies(
                ax_KD,
                ax_D_wt,
                datasets_2phases,
                x_variable,
                z_variable,
                cmap,
                norm,
                marker_2phase,
                marker_opts_scatter__2phase,
                marker_opts_error__2phase,
                x_low=x_mid if axes_low_or_high=="high" else None,
                x_high=x_mid if axes_low_or_high=="low" else None
                )










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
            marker_opts_error__others,
            x_low=x_mid if axes_low_or_high=="high" else None,
            x_high=x_mid if axes_low_or_high=="low" else None
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
            marker_opts_error__others,
            x_low=x_mid if axes_low_or_high=="high" else None,
            x_high=x_mid if axes_low_or_high=="low" else None
            )
            ###########################








            # ***************************
            # ***************************
            if FIT_MODE == 1:
                # combine T and P from datasets_expt, datasets_comp and df dataframe
                sources = list(datasets_expt) + list(datasets_comp) + [df]  # uncomment if needed
                # array_T = row[f"array_T_{secondary_species}"] # z_variable -- T
                # array_P = row[f"array_P_{secondary_species}"] # x_variable -- P
                array_T          = _concat_cols(sources, "Target temperature (K)")
                array_P          = _concat_cols(sources, "Target pressure (GPa)")

                if secondary_species == "He":
                    # array_x_axis = array_X
                    array_X = (array_T ** 0.) * 1e-3
                    secondary_species_label = "He"

                elif secondary_species == "H":
                    # array_x_axis = array_X2
                    array_X2 = array_X = (array_T ** 0.) * 1e-3
                    secondary_species_label = "H2"

                array_x_axis = array_P

                x0 = np.log(array_X)
                x1 = array_P
                x2 = array_T

                if secondary_species == "H":
                    # plot the best fit line
                    y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
                elif secondary_species == "He":
                    y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
                # print(f"Best fit line for KD vs P, T: {y_fit}")
                # print(f"logX: {x0}")
                # print(f"T: {array_T}")
                # print(f"P: {array_P}")
                ax_KD.plot(
                    array_x_axis, y_fit,
                    linestyle='',
                    marker="s",
                    label=f"Best fit model",
                    color='black', markersize=10,
                    alpha=0.15
                )
            # ***************************
            # ***************************





        # show 1 colorbar for both KD and D_wt plots
        # 1) Create a colorbar for the temperature range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # only needed for older matplotlib versions
        # 2) Add the colorbar to the figure -- width == width of 1 plot
        # cbar = fig.colorbar(sm, ax=[ax_D_wt_low, ax_D_wt_high],
        #                     orientation='horizontal',
        #                     # fraction=1, pad=0.04,
        #                     pad=100,  # space between colorbar and plot
        #                     ticks=np.linspace(temp_min, temp_max, 5),
        #                     location='bottom',  # 'top' or 'bottom'
        #                     shrink=1,      # shrink to 80% of the original size
        #                     aspect=50,     # thinner bar
        #                     )
        for gs_bottom_i in [gs_bottom[0], gs_bottom[1]]:
            cax = fig.add_subplot(gs_bottom_i)
            # cax.axis("off")  # hide frame and ticks
            # cax.set_xlabel("Pressure (GPa)", labelpad=10)  # shared label
            cbar = fig.colorbar(
                sm, cax=cax, orientation='vertical',
                ticks=np.linspace(pres_min, pres_max, 5),
                # shrink=0.9,
                # aspect=10
            )
            
            cbar.set_label("Pressure (GPa)", fontsize=font_size_labels)#, rotation=270, labelpad=15)
            # set colorbar ticks to be in K -- 3500, 6500, 9000, 13000, 17000
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_labels)
            cbar.ax.tick_params(labelsize=font_size_ticks)
            # bold boundary
            for spine in cbar.ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("black")

            pos = cax.get_position()
            cax.set_position([pos.x0, pos.y0, pos.width*1.25, pos.height])  # 1.5x wider





        # # x lim ,  y lim

        # for axes in [ax_KD_low, ax_KD_high, ax_D_wt_low, ax_D_wt_high]:
        for axes in [ax_KD_low, ax_D_wt_low]:

            # draw horizontal dashed line at y=1
            axes.axhline(y=1, color='gray', linestyle='-', linewidth=7.5, alpha=0.35)
            # annotate on top and immediately below it, on the right end side
            if axes in [ax_KD_low, ax_D_wt_low]:
                axes.annotate("Lithophile", xy=(0, 1), xytext=(-7, 0.6),
                            fontsize=12, ha='left', color='dimgray')
                axes.annotate("Siderophile", xy=(0, 1), xytext=(-7, 1.3),
                            fontsize=12, ha='left', color='dimgray')

            # if secondary_species == "He":
            #     y_min = 1e-5 # 1e-5
            #     y_max = 1e1 #1e1
            # elif secondary_species == "H":
            #     y_min = 1e-3 #1e-3
            #     y_max = 1e6
            axes.set_ylim(y_min, y_max)
            if axes in [ax_KD_low, ax_D_wt_low]:
                axes.set_xlim(x_min, x_mid)
            else:
                axes.set_xlim(x_mid, x_max)
            axes.set_yscale("log")
            axes.grid(True)

            # for bottom axes, set 1 x label
            # if axes in [ax_D_wt_low, ax_D_wt_high]:
            #     axes.set_xlabel("Pressure (GPa)")
            fig.supxlabel("Temperature (K)", y=0.055, fontsize=font_size_labels)

            # for left axes, set y labels
            if axes in [ax_KD_low]:
                axes.set_ylabel(r"Equilibrium Constant ($K_{D}^{\mathrm{met}/\mathrm{sil}}$)")
            elif axes in [ax_D_wt_low]:
                if secondary_species == "H":
                    axes.set_ylabel(r"Partition Coefficient ($D_\mathrm{H}^{\mathrm{met}/\mathrm{sil}}$)")
                elif secondary_species == "He":
                    axes.set_ylabel(r"Partition Coefficient ($D_\mathrm{He}^{\mathrm{met}/\mathrm{sil}}$)")

            # no y tick labels for right panels
            # if axes in [ax_KD_high, ax_D_wt_high]:
            #     axes.set_yticklabels([])

            axes.tick_params(axis='y', which='both', direction='in', pad=15)

            # x axis ticks and labels
            x_ticks = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
            x_tick_labels = [f"{tick}" for tick in x_ticks]
            if x_variable == "log(Target temperature (K))":
                x_ticks = np.log10( [2000, 4000, 8000, 16000] )
                x_tick_labels = [f"{tick}" for tick in [2000, 4000, 8000, 16000]]
            axes.set_xticks(x_ticks)
            axes.set_xticklabels(x_tick_labels)
            # axes.tick_params(axis='x', which='both', direction='in', pad=10)

            # move ticks to right side on right panels
            # if axes in [ax_KD_high, ax_D_wt_high]:
            #     axes.yaxis.tick_right()                     # move ticks to right side
            #     # axes.yaxis.set_label_position("right")      # move y-axis label too
            #     axes.tick_params(axis='y', which='both', direction='in', pad=5)  # adjust tick direction/padding

            # font sizes
            for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                label.set_fontsize(font_size_ticks)
            axes.set_xlabel(axes.get_xlabel(), fontsize=font_size_labels)
            axes.set_ylabel(axes.get_ylabel(), fontsize=font_size_labels)

            # bold boundaries
            for spine in axes.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("black")

        # Legend
        # 1) Grab handles & labels from one of the axes
        # handles, labels = ax_KD_low.get_legend_handles_labels()
        # Collect from both axes
        handles1, labels1 = ax_KD_low.get_legend_handles_labels()
        # handles2, labels2 = ax_KD_high.get_legend_handles_labels()

        # Merge
        # handles = handles2 + handles1
        # labels  = labels2 + labels1
        handles = handles1
        labels  = labels1

        # Deduplicate while keeping order
        unique = dict(zip(labels, handles))   # keys=labels, values=handles
        handles, labels = unique.values(), unique.keys()

        if secondary_species == "He":
            ncol_legend = 4
        elif secondary_species == "H":
            ncol_legend = 4

        # 2) Create a single legend on the right side of the figure
        fig.legend(
            handles,
            labels,
            loc='lower center',        # center vertically on the right edge
            fontsize=font_size_legend,
            # borderaxespad=0.1,
            bbox_to_anchor=(0.50, 0.9),
            # frameon=False,
            ncol=ncol_legend,  # number of columns in the legend
            # mode='expand',  # 'expand' to fill the space, 'none'
            labelspacing=1.,    # default is 0.5; larger → more space
            handletextpad=1.0,   # default is 0.8; larger → more space between handle & text
            columnspacing=2,   # if you have multiple columns, space between them
            handlelength=1.0,
            markerscale=0.75             # shrink markers to ... % in legend
        )

        # 3) Shrink the subplots to make room for the legend
        # fig.subplots_adjust(right=0.82)  # push the plot area leftward






        # fig.subplots_adjust(top=0.88)

        # fig.suptitle(
        #     f"Equilibrium Constant ($K_D$) and Partition Coefficient ($D_{{wt}}$) "
        #     f"for the reaction {secondary_species}$_{{silicates}}$ $\\rightleftharpoons$ {secondary_species}$_{{metal}}$.\n"
        #     f"Note: Assumption that X$_{{{secondary_species},silicates}}$ $\\ll$ 1",
        #     fontsize=font_size_title,
        #     # y=1.03,            # default ≃0.98, smaller → more gap
        # )

        # plt.subplots_adjust(wspace=0.05, hspace=0.1)
        # plt.subplots_adjust(wspace=0.05)
    
        from collections.abc import Sequence

        from matplotlib.collections import PathCollection

        # --- Axes: remove empty collections & fix linewidths
        for ax in fig.axes:
            for coll in list(ax.collections):
                if isinstance(coll, PathCollection):
                    # remove truly empty offsets
                    if np.size(coll.get_offsets()) == 0:
                        coll.remove()
                        continue

                    lw = coll.get_linewidths()
                    # Normalize: scalar OK; list/array must be non-empty
                    if lw is None:
                        coll.set_linewidth(0.8)
                    elif np.isscalar(lw):
                        pass  # fine
                    elif isinstance(lw, Sequence):
                        if len(lw) == 0:
                            coll.set_linewidth(0.8)
                        # else: keep as-is
                    else:
                        # fallback: treat as array
                        if np.asarray(lw).size == 0:
                            coll.set_linewidth(0.8)

        # Legends too
        for leg in fig.legends:
            for h in leg.legend_handles:
                if isinstance(h, PathCollection):
                    lw = h.get_linewidths()
                    if lw is None or (not np.isscalar(lw) and len(np.atleast_1d(lw)) == 0):
                        h.set_linewidth(0.8)

        # for tick in ax.get_xticklabels() + ax.get_yticklabels():
        #     tick.set_fontweight("bold")

        # 3) Layout & save
        # plt.tight_layout()
        # plt.tight_layout(rect=[0, 0, 1, 1.0])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust layout to make room for the title
        plt.savefig(f"paper__KD_D_wt__vs_P_T__T__{secondary_species}__v_cbar.png", dpi=300)

        # pdf
        # plt.savefig(f"paper__KD_D_wt__vs_P_T_{secondary_species}.eps")
        plt.savefig(f"paper__KD_D_wt__vs_P_T__T__{secondary_species}__v_cbar.svg", format="svg", dpi=300)










































    print("Plotting completed successfully.")