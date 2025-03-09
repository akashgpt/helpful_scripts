import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

# Read file COLVAR as a pandas DataFrame
filename_OG = "COLVAR"
filename_plot = "COLVAR.plot"
# filename = "COLVAR"

#copy from COLVAR to COLVAR.plot
os.system(f'cp {filename_OG} {filename_plot}')

# Remove "#! FIELDS " from the first line of the filename_plot
os.system(f"sed -i '1s/^#! FIELDS //' {filename_plot}")

try:
    df = pd.read_csv(filename_plot, delim_whitespace=True)
except Exception as e:
    print("Error: could not read file COLVAR")
    sys.exit(1)

# Create a figure with three vertical subplots sharing the same x-axis (time)
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 8))

# Plot XRD vs time
axes[0].plot(df['time'], df['xrd'], marker='', linestyle='-',alpha=0.5)
axes[0].set_ylabel('XRD')
axes[0].set_title('PLUMED COLVAR Data')

# Plot vol vs time
axes[1].plot(df['time'], df['vol'], marker='', linestyle='-',alpha=0.5)
axes[1].set_ylabel('Volume (A$^3$)')

# Plot energy vs time
axes[2].plot(df['time'], df['energy']/96.485, marker='', linestyle='-',alpha=0.5)
axes[2].set_ylabel('Energy (eV)')#(kJ/mol)')
axes[2].set_xlabel('Time (ps)')

# grid for each subplot
for ax in axes:
    ax.grid()


# Adjust layout for better spacing between subplots
plt.tight_layout()
plt.savefig('plmd__XRD_V_E_vs_t.png')
plt.show()

