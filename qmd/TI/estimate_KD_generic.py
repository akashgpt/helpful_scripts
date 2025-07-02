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
            log.Ghp_analysis
        Config2/
            log.Ghp_analysis
    P100_T4000/
        Config1/
            log.Ghp_analysis
MgSiO3_{secondary_species}/
    P50_T3500/
        Config1/
            log.Ghp_analysis
        Config2/
            log.Ghp_analysis
This script will output a CSV file `all_TI_results.csv` with the following columns:
Phase, P_T_folder, Config_folder, Target pressure, Target temperature, Atom counts, Atomic masses,
Unique species, Total # of atoms, G_hp, G_hp_error, G_hp_per_atom, G_hp_per_atom_error,
HF_ig, TS (this is kB*T*S), Volume (Å³), Volume per atom (Å³), X_{secondary_species}, mu_excess_{secondary_species}, mu_{secondary_species}_TS_term, mu_{secondary_species},
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


# 9) Fit linear excess chemical potential mu_excess = a + b
# Initialize columns
df["a"] = np.nan
df["b"] = np.nan
df["a_error"] = np.nan
df["b_error"] = np.nan

# Group by Phase and P_T_folder to fit separate lines
for (phase, pt), sub in df.groupby(["Phase", "P_T_folder"]):
    x = sub[f"X_{secondary_species}"].values
    y = sub["G_hp_per_atom_w_TS"].values
    if len(x) > 1:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)  # fit y = a + b*x
        slope, intercept = coeffs
        # Calculate the standard error of the slope and intercept
        slope_error = np.sqrt(cov[0, 0])
        intercept_error = np.sqrt(cov[1, 1])
        # error in slope and intercept
    else:
        intercept, slope = np.nan, np.nan
        slope_error, intercept_error = np.nan, np.nan
    mask = (df["Phase"] == phase) & (df["P_T_folder"] == pt)
    df.loc[mask, "a"] = intercept
    df.loc[mask, "b"] = slope
    df.loc[mask, "a_error"] = intercept_error
    df.loc[mask, "b_error"] = slope_error

# Compute mu_excess = a + b (no X_{secondary_species} factor)
df[f"mu_excess_{secondary_species}"] = df["a"] + df["b"]

# 10) Compute mixing entropy term mu_TS and total mu_{secondary_species}
def compute_mu_TS(row):
    """
    Compute the TS term mixing entropy term for {secondary_species}, using different formulas per phase.
    Fe_{secondary_species}: TS = kB * T * ln(X)
    MgSiO3_{secondary_species}: TS = kB * T * ln( X / (5 - 4X) )
    """
    T = row.get("Target temperature (K)")
    X = row.get(f"X_{secondary_species}", 0)

    # add floor to X to avoid log(0)
    X = max(X, 1e-10)  # avoid log(0) or negative values

    if T is None or X < 0:
        return np.nan
    if row["Phase"] == f"Fe_{secondary_species}":
        # print(f"kB * T * np.log(X) for Fe_{secondary_species} = {kB * T * np.log(X)}")
        return kB * T * np.log(X)
    elif row["Phase"] == f"MgSiO3_{secondary_species}":
        # print(f"kB * T * np.log(X / (5 - 4 * X)) for MgSiO3_{secondary_species} = {kB * T * np.log(X / (5 - 4 * X))}")
        denom = 5 - 4 * X
        return kB * T * np.log(X / denom)# if denom > 0 else np.nan
    else:
        return np.nan

# apply TS term and total mu_{secondary_species}
df[f"mu_{secondary_species}_TS_term"] = df.apply(compute_mu_TS, axis=1)
df[f"mu_{secondary_species}"] = df[f"mu_excess_{secondary_species}"] + df[f"mu_{secondary_species}_TS_term"]


# print(f"df summary:\
# {df.describe()}\n")



# partiction coefficient: (1.78/5) * np.exp(-(mu_excess_{secondary_species}_for_Fe - mu_excess_{secondary_species}_for_MgSiO3) / (kB * T)) for the same P_T_folder
def compute_KD(row):
    # 1) Identify this row’s phase & P_T group
    phase = row["Phase"]
    pt    = row["P_T_folder"]
    # 2) Determine the other phase
    other_phase = f"MgSiO3_{secondary_species}" if phase == f"Fe_{secondary_species}" else f"Fe_{secondary_species}"
    # 3) Grab that phase’s mu_excess_{secondary_species} for the same P_T_folder
    partner = df.loc[(df["Phase"] == other_phase) & (df["P_T_folder"] == pt), f"mu_excess_{secondary_species}"]

    # Determine the multiplier for the exponent based on phase
    if phase == f"Fe_{secondary_species}":
        mult_factor = 1 # to ensure that KD is always for {secondary_species}_{silicate} -> {secondary_species}_{metal}
    else:
        mult_factor = -1

    if partner.empty:
        return np.nan
    other_mu = partner.iloc[0]
    # 4) Get the temperature (in K)
    T = row["Target temperature (K)"]
    if np.isnan(T) or T <= 0:
        return np.nan
    # 5) Compute KD
    # solve for KD = (x/y) such that (x/y) = (1/(5-4*y)) * np.exp(-mult_factor*(row[f"mu_excess_{secondary_species}"] - other_mu) / (kB * T))
    return (1/5.0) * np.exp(-mult_factor*(row[f"mu_excess_{secondary_species}"] - other_mu) / (kB * T)) # assuming y is ~ 0

# Apply it
df["KD_sil_to_metal"] = df.apply(compute_KD, axis=1)
df["D_wt"] = df["KD_sil_to_metal"] * (100/56)

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
        print(f"                                KD_sil_to_metal = {kd0:.4f} (mu_excess = {mu0:.4f})")
    else:
        print(f"{folder}, Phase={phase}: "
                "No valid KD_sil_to_metal values found.")

print(f"="*50)
print(f"Note: It is assumed here that y or X_{secondary_species} in silicates is << 1. If not, multiply K_D and D_wt by (5 / (5-4y)) !")
print(f"="*50)
print("")







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
    ax.scatter(
        sub[f"X_{secondary_species}"],
        sub["G_hp_per_atom_w_TS"],
        c=mapped_codes[sub.index],
        cmap=cmap, norm=norm,
        s=size_map[phase],
        alpha=alpha_map[phase],
        label=phase
    )

# 7) Overlay linear fits y = a + b*x for each (Phase, P_T_folder)
for (phase, pt), sub in df.groupby(["Phase", "P_T_folder"]):
    a = sub["a"].iloc[0]
    b = sub["b"].iloc[0]
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
#     a = sub["a"].iloc[0]
#     b = sub["b"].iloc[0]
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







fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax1, ax2, ax3, ax4 = axes.flatten()

# 2) Common settings
marker_opts = dict(marker='o', linestyle='', markersize=10, alpha=0.5)
marker_opts__others = dict(marker='s', linestyle='', markersize=15, alpha=0.25)

# --- Panel 1: KD_sil_to_metal (linear) ---
ax1.plot(df["P_T_folder"], df["KD_sil_to_metal"], **marker_opts)
ax1.set_ylabel("K_D (silicate → metal)")
# ax1.set_title("Partition Coefficient (linear)")
ax1.grid(True)
# remove x-tick labels on top row
ax1.tick_params(labelbottom=False)

# --- Panel 2: D_wt (linear) ---
ax3.plot(df["P_T_folder"], df["D_wt"], **marker_opts)
ax3.set_ylabel("D_wt (silicate → metal)")
# ax3.set_title("Distribution Coefficient (linear)")
ax3.grid(True)
# x label
ax3.set_xlabel("P, T")
# rotate bottom x-labels

# --- Panel 3: KD_sil_to_metal (log y) ---
ax2.plot(df["P_T_folder"], df["KD_sil_to_metal"], **marker_opts)
ax2.set_yscale("log")
ax2.set_ylabel("K_D (silicate → metal; log scale)")
# ax2.set_title("Partition Coefficient (log)")
ax2.grid(True)
ax2.tick_params(labelbottom=False)


# --- Panel 4: D_wt (log y) ---
ax4.plot(df["P_T_folder"], df["D_wt"], **marker_opts)
ax4.set_yscale("log")
ax4.set_ylabel("D_wt (silicate → metal; log scale)")
# ax4.set_title("Distribution Coefficient (log)")
ax4.grid(True)
ax4.set_xlabel("P, T")

# super title
fig.suptitle(f"Partition Coefficient (K_D) and Weight Distribution Coefficient (D_wt) for {secondary_species} in Fe and MgSiO3. \n Note: Assumption that X_{secondary_species} in silicates is << 1", fontsize=12)


# if secondary_species is "He" -- in all plots, add two data points at P500_T9000 0.032 and at P1000_T13000, 1
if secondary_species == "He":
    ax1.plot("P500_T9000", 0.032, **marker_opts)
    ax1.plot("P500_T9000", 0.07, **marker_opts)
    # ax1.plot("P1000_T13000", 1, **marker_opts)
    ax1.plot("P1000_T13000", 0.32, **marker_opts)
    ax2.plot("P500_T9000", 0.032, **marker_opts)
    ax2.plot("P500_T9000", 0.07, **marker_opts)
    # ax2.plot("P1000_T13000", 1, **marker_opts)
    ax2.plot("P1000_T13000", 0.32, **marker_opts)
    ax3.plot("P500_T9000", 0.032*1.78, **marker_opts)
    ax3.plot("P500_T9000", 0.07*1.78, **marker_opts)
    # ax3.plot("P1000_T13000", 1*1.78, **marker_opts)
    ax3.plot("P1000_T13000", 0.32*1.78, **marker_opts)
    ax4.plot("P500_T9000", 0.032*1.78, **marker_opts)
    ax4.plot("P500_T9000", 0.07*1.78, **marker_opts)
    # ax4.plot("P1000_T13000", 1*1.78, **marker_opts)
    ax4.plot("P1000_T13000", 0.32*1.78, **marker_opts)

# previous studies
if secondary_species == "He":
    # P50_T3500: 2.6E-3
    # P135_T4200: 8.7E-4
    ax1.plot("P50_T3500", 2.6E-3, **marker_opts__others, color='red', label="Previous Study: P50_T3500")
    # ax1.plot("P135_T4200", 8.7E-4, **marker_opts__others, color='red', label="Previous Study: P135_T4200")
    ax2.plot("P50_T3500", 2.6E-3, **marker_opts__others, color='red', label="Previous Study: P50_T3500")
    # ax2.plot("P135_T4200", 8.7E-4, **marker_opts__others, color='red', label="Previous Study: P135_T4200")
    ax3.plot("P50_T3500", 2.6E-3*1.78, **marker_opts__others, color='red', label="Previous Study: P50_T3500")
    # ax3.plot("P135_T4200", 8.7E-4*1.78, **marker_opts__others, color='red', label="Previous Study: P135_T4200")
    ax4.plot("P50_T3500", 2.6E-3*1.78, **marker_opts__others, color='red', label="Previous Study: P50_T3500")
    # ax4.plot("P135_T4200", 8.7E-4*1.78, **marker_opts__others, color='red', label="Previous Study: P135_T4200")


if secondary_species == "H":
    # Luo et al.
    # P500_T9000: 10**1.4
    # P1000_T13000: 10**1.8
    ax1.plot("P500_T9000", 10**1.4, **marker_opts__others, color='red', label="Luo et al. P500_T9000")
    ax1.plot("P1000_T13000", 10**1.8, **marker_opts__others, color='red', label="Luo et al. P1000_T13000")
    ax2.plot("P500_T9000", 10**1.4, **marker_opts__others, color='red', label="Luo et al. P500_T9000")
    ax2.plot("P1000_T13000", 10**1.8, **marker_opts__others, color='red', label="Luo et al. P1000_T13000")
    ax3.plot("P500_T9000", 10**1.4*1.78, **marker_opts__others, color='red', label="Luo et al. P500_T9000")
    ax3.plot("P1000_T13000", 10**1.8*1.78, **marker_opts__others, color='red', label="Luo et al. P1000_T13000")
    ax4.plot("P500_T9000", 10**1.4*1.78, **marker_opts__others, color='red', label="Luo et al. P500_T9000")
    ax4.plot("P1000_T13000", 10**1.8*1.78, **marker_opts__others, color='red', label="Luo et al. P1000_T13000")

    # previous study
    # P50_T3500: 9.1
    ax1.plot("P50_T3500", 9.1, **marker_opts__others, color='blue', label="Previous Study: P50_T3500")
    ax2.plot("P50_T3500", 9.1, **marker_opts__others, color='blue', label="Previous Study: P50_T3500")
    ax3.plot("P50_T3500", 9.1*1.78, **marker_opts__others, color='blue', label="Previous Study: P50_T3500")
    ax4.plot("P50_T3500", 9.1*1.78, **marker_opts__others, color='blue', label="Previous Study: P50_T3500")

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