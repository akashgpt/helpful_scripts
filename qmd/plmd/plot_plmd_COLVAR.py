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
axes[-1].set_xlabel('Time')

# increase label font size
for ax in axes:
    ax.yaxis.label.set_size(20)
axes[-1].xaxis.label.set_size(20)

# font size of tick labels
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=16)

# decrease vertical distance between subplots
# plt.subplots_adjust(hspace=0.01)

print("Saving figure to plot_COLVAR.png")


# Adjust layout, save the figure, and display the plot
plt.tight_layout()
plt.savefig("plot_COLVAR.png")
plt.show()