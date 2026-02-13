import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import argparse

# python $HELP_SCRIPTS_plmd/plot_plmd_COLVAR.py > log.plot_plmd_COLVAR 2>&1 &

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--num_variables_2_print", "-n", type=int, default=3,
                    help="Number of variables (columns after 'time') from COLVAR to plot")
args = parser.parse_args()

# Filenames
filename_OG = "COLVAR"
filename_plot = "COLVAR.plot"

# Copy original file to a new file for processing
os.system(f'cp {filename_OG} {filename_plot}')

# Remove the "#! FIELDS " prefix from the first line of the file
os.system(f"sed -i '1s/^#! FIELDS //' {filename_plot}")

# Read the processed file into a pandas DataFrame
try:
    df = pd.read_csv(filename_plot, delim_whitespace=True)
except Exception as e:
    print("Error: could not read file COLVAR")
    sys.exit(1)

# Number of variables to plot (from command-line argument)
num_vars = args.num_variables_2_print

# Check that there are enough columns (the first column is assumed to be "time")
if len(df.columns) < 1 + num_vars:
    print(f"Error: The file only has {len(df.columns)-1} variable columns after 'time'.")
    sys.exit(1)

if num_vars == 0: # select all columns after time
    num_vars = len(df.columns) - 1
    print(f"--num_variables_2_print given as '0'. Selecting all {num_vars} variable columns after 'time' for plotting.")

# Determine the variable columns to plot: all columns after the 'time' column
# Assumes that the first column is "time"
var_columns = df.columns[1:1+num_vars]
print("Plotting variables:", list(var_columns))

print("Generating plot...")

# Create subplots for each variable (vertically stacked, sharing the x-axis)
fig, axes = plt.subplots(nrows=num_vars, ncols=1, sharex=True, figsize=(10, 3*num_vars))

print("Generating plot... Pt 2")

# If only one variable is to be plotted, axes is not a list; so force it to be iterable
if num_vars == 1:
    axes = [axes]

print("Plotting variables...")


# Plot each selected variable versus time
for ax, col in zip(axes, var_columns):
    ax.plot(df['time'], df[col], marker='', linestyle='-', alpha=0.5)
    ax.set_ylabel(col)
    ax.grid(True)

# Label the shared x-axis
axes[-1].set_xlabel('Time (ps)')

# increase label font size
for ax in axes:
    ax.yaxis.label.set_size(20)
    ax.tick_params(axis='both', which='major', labelsize=16)
axes[-1].xaxis.label.set_size(20)


# set y axes limits to exclude outliers (robust to NaN/Inf)
for ax, col in zip(axes, var_columns):
    data = pd.to_numeric(df[col], errors="coerce").to_numpy()
    data = data[np.isfinite(data)]   # drop NaN/Inf

    if data.size < 5:
        # not enough finite data to set limits safely
        continue

    lower = np.percentile(data, 1)
    if lower < 0:
        lower = lower * 1.10
    else:
        lower = lower * 0.90

    upper = np.percentile(data, 99)
    if upper < 0:
        upper = upper * 0.90
    else:
        upper = upper * 1.10

    # if still bad / identical (flat signal), skip or add tiny padding
    if (not np.isfinite(lower)) or (not np.isfinite(upper)):
        continue
    if lower == upper:
        pad = 1e-12 if lower == 0 else 1e-6 * abs(lower)
        ax.set_ylim(lower - pad, upper + pad)
    else:
        ax.set_ylim(lower, upper)

    # set x axis limit to 0 to 600 or df['time'].max()
    # x_axis_max = max(df['time'].max(), 600)
    # ax.set_xlim(0, x_axis_max)

    # if col="cn_Ratio", set y axis limit to 0 to 1
    if col == "cn_Ratio" or col == "cn_ratio":
        ax.set_ylim(0, 1)
        # ax.set_ylim(0.15, .6)
    # if col == "vol":
    #     ax.set_ylim(500, 2000)
    # if col == "energy":
    #     ax.set_ylim(-90000, -60000)




# decrease vertical distance between subplots
# plt.subplots_adjust(hspace=0.01)

print("Saving figure to plot_COLVAR.png")


# Adjust layout, save the figure, and display the plot
plt.tight_layout()
plt.savefig("plot_COLVAR.png")
plt.show()