#!/usr/bin/env python3

# # directories are structured as Fe_He and MgSiO3_He. Each dir contains folders such as PX_TY 
# # where X is pressure in GPa and Y is temperature in K. In each PX_TY folder, go to the sub-folders
# # which have SCALEE_1 folder inside and designate these as CONFIG_DIR. In each CONFIG_DIR, read the entire content of
# # the file log.Ghp_analysis which is as follows:
# # 

# Setup
# -----
# Base directory      : /scratch/gpfs/BURROWS/akashgpt/qmd_data/Fe_MgSiO3_He/TI/Fe_He/P50_T3500/64Fe_0He
# Directory name      : 64Fe_0He
# Target pressure     : 50.000  GPa
# Target temperature  : 3500.0  K
# Unique species      : ['He', 'Fe']
# Atom counts         : [ 0 64]
# Atomic masses       : [ 4.002602 55.845   ]
# Total # of atoms    : 64

# WARNING: WHY '+R11' in Ghp calculation? This is correct, right?
# WARNING: ΔHF_hp_m_ab is hardcoded to -0.571 eV. Not needed for my calculations, right? Not being used right now though


# Results Summary (in eV unless mentioned)
# -----------------------
# G_hp                : -530.496
# G_hp_error          : 0.32
# G_hp_per_atom       : -8.289
# G_hp_per_atom_error : 0.005
# HF_ig               : -249.44532890907502
# TS                  : 0.0
# Volume (Å³)         : 622.208
# Volume per atom (Å³): 9.722


# Saved results to /scratch/gpfs/BURROWS/akashgpt/qmd_data/Fe_MgSiO3_He/TI/Fe_He/P50_T3500/64Fe_0He/TI_analysis.csv

# # save all this data in pandas dataframe.

# Usage: nohup python $HELP_SCRIPTS_TI/estimate_KD.py > log.estimate_KD 2>&1 &

#!/usr/bin/env python3
import re
import ast
from pathlib import Path
import pandas as pd
import numpy as np

kB = 8.617333262145e-5  # eV/K

# ─── 1) CONFIGURE YOUR ROOT FOLDERS ────────────────────────────────────────────────
ROOT_DIRS = ["Fe_He", "MgSiO3_He"]

# ─── 2) A VERY SIMPLE KEY-VALUE REGEX ──────────────────────────────────────────────
#    Captures everything before the first ':' as key, and everything after as val.
KV_RE = re.compile(r'^\s*([^:]+?)\s*:\s*(.+)$')

def parse_log(path):
    """
    Read a log.Ghp_analysis and return a dict of {key: raw_string_value}.
    We do *no* stripping of brackets or units here—just grab the raw text.
    """
    out = {}
    for line in path.read_text().splitlines():
        m = KV_RE.match(line)
        if not m:
            continue
        key, raw = m.group(1).strip(), m.group(2).strip()
        out[key] = raw
    return out

# ─── 3) HELPERS TO CLEAN EACH COLUMN ───────────────────────────────────────────────
int_re   = re.compile(r'-?\d+')
float_re = re.compile(r'[-+]?\d*\.\d+|\d+')
str_re   = re.compile(r"'([^']*)'|\"([^\"]*)\"")  # things in quotes

def clean_int_list(raw):
    """Turn ' [0 64' or '[ 0,64 ]' into [0,64]."""
    if not isinstance(raw, str):
        return list(raw)
    return [int(x) for x in int_re.findall(raw)]

def clean_float_list(raw):
    """Turn '[4.002602 55.845' (or any mix) into [4.002602,55.845]."""
    if not isinstance(raw, str):
        return list(raw)
    return [float(x) for x in float_re.findall(raw)]

def clean_species_list(raw):
    """
    Turn \"['He', 'Fe'\"  (missing ]) into ['He','Fe'].
    1) Add trailing ] if needed.
    2) Try ast.literal_eval.
    3) Fall back to regex on quotes.
    """
    if not isinstance(raw, str):
        return list(raw)
    s = raw.strip()
    # if it looks like a Python list but missing the ], add it
    if s.startswith('[') and not s.endswith(']'):
        s = s + ']'
    # try a safe literal_eval
    try:
        lst = ast.literal_eval(s)
        return [str(x) for x in lst]
    except Exception:
        # fallback: grab anything in quotes
        return [a or b for a,b in str_re.findall(s)]

def clean_scalar(raw):
    """Try to cast to float, else leave as string."""
    try:
        return float(raw)
    except:
        return raw

# ─── 4) WALK THE TREE & PARSE LOGS ────────────────────────────────────────────────
records = []
for phase in ROOT_DIRS:
    for ptdir in Path(phase).iterdir():
        if not ptdir.is_dir(): 
            continue
        for cfg in ptdir.iterdir():
            # if not (cfg / "SCALEE_1").is_dir():
            #     continue
            if not (cfg / "log.Ghp_analysis").exists():
                continue
            logfile = cfg / "log.Ghp_analysis"
            if not logfile.exists():
                continue
            rec = parse_log(logfile)
            # record provenance
            rec["Phase"]         = phase
            rec["P_T_folder"]    = ptdir.name
            rec["Config_folder"] = cfg.name
            records.append(rec)

# ─── 5) BUILD A DATAFRAME ────────────────────────────────────────────────────────
df = pd.DataFrame(records)

# ─── 6) CLEAN EVERY COLUMN YOU CARE ABOUT ────────────────────────────────────────
df["Atom counts"]     = df["Atom counts"].apply(clean_int_list)
df["Atomic masses"]   = df["Atomic masses"].apply(clean_float_list)
df["Unique species"]  = df["Unique species"].apply(clean_species_list)

# Clean all the simple scalar columns at once:
to_scalar = [
    "Target pressure", "Target temperature",
    "Total # of atoms",
    "G_hp", "G_hp_error", "G_hp_per_atom", "G_hp_per_atom_error",
    "HF_ig", "TS",
    "Volume (Å³)", "Volume per atom (Å³)",
]
for col in to_scalar:
    df[col] = df[col].apply(clean_scalar)

# ─── 7) COMPUTE X_He ─────────────────────────────────────────────────────────────
def frac_he(row):
    mapping = dict(zip(row["Unique species"], row["Atom counts"]))
    total = sum(row["Atom counts"])
    return mapping.get("He", 0) / total if total else 0.0

df["X_He"] = df.apply(frac_he, axis=1)

# delete WARNING
df.drop(columns=["WARNING"], inplace=True)

# extract only values from pressure and temperature columns
# Option B: strip out any non-numeric characters (allowing dot and minus)
df["Target pressure"] = (
    df["Target pressure"]
        .astype(str)
        .str.replace(r'[^0-9\.\-]+', '', regex=True)
        .astype(float)
)

df["Target temperature"] = (
    df["Target temperature"]
        .astype(str)
        .str.replace(r'[^0-9\.\-]+', '', regex=True)
        .astype(float)
    )


# Rename to include units
df.rename(
    columns={
        "Target pressure":    "Target pressure (GPa)",
        "Target temperature": "Target temperature (K)",
    },
    inplace=True
)









# Prepare empty columns
df["a"] = np.nan
df["b"] = np.nan

# 1) Loop over (phase, P_T_folder) groups
for (phase, pt), sub in df.groupby(["Phase", "P_T_folder"]):
    x = sub["X_He"].to_numpy()
    y = sub["G_hp_per_atom"].to_numpy()
    # polyfit returns [slope, intercept]
    b, a = np.polyfit(x, y, 1)
    # mask for this group
    mask = (df["Phase"] == phase) & (df["P_T_folder"] == pt)
    df.loc[mask, "a"] = a
    df.loc[mask, "b"] = b



# add mu_excess_He = a+b
df["mu_excess_He"] = df["a"] + df["b"]


# for each row, if phase is Fe_He, then TS_mix = kB * T * np.log(X_He). If phase is MgSiO3_He, then TS_mix = kB * T * np.log(X_He/(5 - 4*X_He))
def mu_He_TS_term(row):
    if row["Phase"] == "Fe_He":
        return kB * row["Target temperature (K)"] * np.log(row["X_He"])
    elif row["Phase"] == "MgSiO3_He":
        # print(f"np.log(<denominator>) = {5 - 4*row["X_He"]}")
        # print("Warning: X_He = 1.25, division by zero")
        # print(row)            
        return kB * row["Target temperature (K)"] * np.log(row["X_He"] / (5 - 4*row["X_He"]))
    else:
        return np.nan
df["mu_He_TS_term"] = df.apply(mu_He_TS_term, axis=1)

# mu_He = mu_excess_He - TS
df["mu_He"] = df["mu_excess_He"] + df["mu_He_TS_term"]





# ─── 8) OUTPUT ───────────────────────────────────────────────────────────────────
pd.set_option("display.width", 180)
# print(df)
df.to_csv("all_TI_results.csv", index=False)














# plot X_He vs G_hp_per_atom, and color by P_T_folder and size by phase
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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
size_map  = {"Fe_He": 200, "MgSiO3_He": 100}
alpha_map = {"Fe_He": 0.5,  "MgSiO3_He": 1.0}

# --- Make the figure & axes ---
fig, ax = plt.subplots(figsize=(10,6))

# 1) Scatter the raw data, grouping by Phase so we get two sizes/alphas
for phase, sub in df.groupby("Phase"):
    ax.scatter(
        sub["X_He"], sub["G_hp_per_atom"],
        c=mapped_codes[sub.index],      # use remapped folder codes
        cmap=cmap, norm=norm,
        s=size_map[phase],
        alpha=alpha_map[phase],
        label=phase
    )

# 2) Overlay the fitted lines for each (Phase, P_T_folder)
#    We pick the same x-range per folder so the line spans the full data
for (phase, pt), sub in df.groupby(["Phase","P_T_folder"]):
    a = sub["a"].iloc[0]
    b = sub["b"].iloc[0]
    # line x from min to max of that subgroup
    x_line = np.linspace(sub["X_He"].min(), sub["X_He"].max(), 200)
    # get the original code for this folder, then map→0..N-1 for color lookup
    orig_code = folder_cats.cat.categories.get_loc(pt)
    mcode     = remap[orig_code]
    ax.plot(
        x_line, a + b*x_line,
        color=cmap(mcode),
        linestyle="--",
        linewidth=2#,
        # label=f"{phase}, {pt} fit"
    )

# 3) Colorbar for the folders
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(
    sm,
    boundaries=np.arange(len(used_codes)+1)-0.5,
    ticks=np.arange(len(used_codes)),
    ax=ax
)
cbar.ax.set_yticklabels([folder_cats.cat.categories[i] for i in used_codes])
cbar.set_label("P_T_folder")

# 4) Final styling
ax.set_xlabel("X_He")
ax.set_ylabel("G_hp_per_atom (eV)")
ax.set_title("X_He vs G_hp_per_atom with Line Fits by Phase+P_T_folder")
ax.legend()
# ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small", title="Legend")
ax.grid(True)
plt.tight_layout()
plt.savefig("X_He_vs_G_hp_per_atom.png")
# plt.show()


# narrow df to Fe_He phase and P_T_folder = P50_T3500
# df = df[ (df["P_T_folder"] == "P50_T3500")]

# plot mu_He vs X_He, and color by P_T_folder and size by phase
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
size_map  = {"Fe_He": 100, "MgSiO3_He": 50}
alpha_map = {"Fe_He": 0.5,  "MgSiO3_He": 1.0}
# --- Make the figure & axes ---
fig, ax = plt.subplots(figsize=(10,6))
# 1) Scatter the raw data, grouping by Phase so we get two sizes/alphas
for phase, sub in df.groupby("Phase"):
    ax.scatter(
        sub["X_He"], sub["mu_He"],
        c=mapped_codes[sub.index],      # use remapped folder codes
        cmap=cmap, norm=norm,
        s=size_map[phase],
        alpha=alpha_map[phase],
        label=phase
    )
    print(f"phase = {phase}")
    print(f"sub['X_He'] = {sub['X_He']}")
    print(f"sub['mu_He'] = {sub['mu_He']}")
# # 2) Overlay the fitted lines for each (Phase, P_T_folder)
# #    We pick the same x-range per folder so the line spans the full data
# for (phase, pt), sub in df.groupby(["Phase","P_T_folder"]):
#     a = sub["a"].iloc[0]
#     b = sub["b"].iloc[0]
#     # line x from min to max of that subgroup
#     x_line = np.linspace(sub["X_He"].min(), sub["X_He"].max(), 200)
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
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(
    sm,
    boundaries=np.arange(len(used_codes)+1)-0.5,
    ticks=np.arange(len(used_codes)),
    ax=ax
)
cbar.ax.set_yticklabels([folder_cats.cat.categories[i] for i in used_codes])
cbar.set_label("P_T_folder")
# 4) Final styling
ax.set_xlabel("X_He")
ax.set_ylabel("mu_He (eV)")
ax.set_title("X_He vs mu_He with Line Fits by Phase+P_T_folder")
ax.legend()
# ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small", title="Legend")
ax.grid(True)
plt.tight_layout()
plt.savefig("X_He_vs_mu_He.png")



# plot all mu_He vs X_He, and color by P_T_folder
plt.figure(figsize=(10,6))
plt.scatter(
    df["X_He"], df["mu_He"],
    s=100,
    alpha=0.5
)
plt.savefig("test.png")

print("Created: dataframe with G_hp_per_atom, G_hp_per_atom_error, X_He, etc. from all systems")
print(f"Files created: all_TI_results_with_XHe.csv, X_He_vs_G_hp_per_atom.png")