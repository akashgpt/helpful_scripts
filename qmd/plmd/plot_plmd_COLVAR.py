import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--num_variables_2_print", "-n", type=int, default=3,
                    help="Number of variables (columns after 'time') from COLVAR to plot")
args = parser.parse_args()

num_vars = args.num_variables_2_print

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

# Check that there are enough columns (the first column is assumed to be "time")
if len(df.columns) < 1 + num_vars:
    print(f"Error: The file only has {len(df.columns)-1} variable columns after 'time'.")
    sys.exit(1)

# Determine the variable columns to plot: all columns after the 'time' column
# Assumes that the first column is "time"
var_columns = df.columns[1:1+num_vars]
print("Plotting variables:", list(var_columns))

# Create subplots for each variable (vertically stacked, sharing the x-axis)
fig, axes = plt.subplots(nrows=num_vars, ncols=1, sharex=True, figsize=(10, 3*num_vars))

# If only one variable is to be plotted, axes is not a list; so force it to be iterable
if num_vars == 1:
    axes = [axes]

# Plot each selected variable versus time
for ax, col in zip(axes, var_columns):
    ax.plot(df['time'], df[col], marker='', linestyle='-', alpha=0.5)
    ax.set_ylabel(col)
    ax.grid(True)

# Label the shared x-axis
axes[-1].set_xlabel('Time')

# Adjust layout, save the figure, and display the plot
plt.tight_layout()
plt.savefig("plot_COLVAR.png")
plt.show()