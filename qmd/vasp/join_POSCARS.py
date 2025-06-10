import numpy as np
import matplotlib.pyplot as plt
from ase.io.vasp import read_vasp, write_vasp
from ase import Atoms
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# Usage: python $HELP_SCRIPTS_vasp/join_POSCARS.py -d1 f1002 -d2 f1001 -do f1003 -f CONTCAR -g 0.05
# Usage: python $HELP_SCRIPTS_vasp/join_POSCARS.py -d1 g1002 -d2 g1001 -do g1003 -f CONTCAR -g 0.05
# This script merges two VASP POSCAR files into a single POSCAR file with a specified gap between the two structures.

# Load two POSCAR files
# dir1 = "f0002"
# dir2 = "f0004"

# take dir1 and dir2 as input
parser = argparse.ArgumentParser(description="Join CONTCARs, etc. to create a joint (2-phase) POSCAR.")
parser.add_argument("-d1", "--dir1", type=str, required=True, help="directory 1.")
parser.add_argument("-d2", "--dir2", type=str, required=True, help="directory 2.")
parser.add_argument("-do", "--dir_out", type=str, default="joint_POSCAR", help="directory out.")
parser.add_argument("-f", "--file_to_read", type=str, default="CONTCAR", help="file to read, default: CONTCAR.")
parser.add_argument("-g", "--gap", type=float, default=0.25, help="gap between cells in Å, default: 0.25 Å.")
args = parser.parse_args()
dir1 = args.dir1
dir2 = args.dir2
dir_out = args.dir_out
file_to_read = args.file_to_read
gap_between_cells = args.gap

# trim all str arguments
dir1 = dir1.strip()
dir2 = dir2.strip()
dir_out = dir_out.strip()
file_to_read = file_to_read.strip()


# print all arguments
print(f"dir1: {dir1}")
print(f"dir2: {dir2}")
print(f"dir_out: {dir_out}")
print(f"file_to_read: {file_to_read}")
print(f"gap_between_cells: {gap_between_cells}")
# print(dir1 + "/" + file_to_read)



# atoms1 = read_vasp(dir1 + "/POSCAR")
# atoms2 = read_vasp(dir2 + "/POSCAR")
atoms1 = read_vasp(dir1 + "/" + file_to_read)
atoms2 = read_vasp(dir2 + "/" + file_to_read)

# wrap atoms to unit cell
atoms1.wrap()
atoms2.wrap()

# Translate atoms2 to avoid overlap (Shift along z-axis)
# shift_z = max(atoms1.positions[:, 2]) - min(atoms2.positions[:, 2]) + gap_between_cells  # 2 Å gap
# shift_z by z dimension of atoms1
shift_z = atoms1.cell[2, 2] + gap_between_cells  # 2 Å gap
atoms2.translate([0, 0, shift_z])

# Combine both structures
combined_atoms = atoms1 + atoms2

# make the z-axis as the sum of the two structures
combined_atoms.cell[2, 2] = atoms1.cell[2, 2] + atoms2.cell[2, 2] + gap_between_cells  # 2 Å gap

# check if dir_out exists, if not create it
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Write new POSCAR
write_vasp(dir_out + "/POSCAR", combined_atoms, direct=True)
print(f"Merged structures from {dir1} and {dir2} into {dir_out}")

# create log.join_POSCARS with all arguments
with open(dir_out + "/log.join_POSCARS", "w") as f:
    f.write(f"dir1: {dir1}\n")
    f.write(f"dir2: {dir2}\n")
    f.write(f"dir_out: {dir_out}\n")
    f.write(f"file_to_read: {file_to_read}\n")
    f.write(f"gap_between_cells: {gap_between_cells}\n")


### Visualization
# Get atomic positions and elements
positions = combined_atoms.get_positions()
elements = combined_atoms.get_chemical_symbols()

# Assign colors based on element types
unique_elements = list(set(elements))

# Create a color map
# color_map = {element: plt.cm.jet(i / len(unique_elements)) for i, element in enumerate(unique_elements)}
# start color map from 0
color_map = {element: plt.cm.jet(i / (len(unique_elements) - 1)) for i, element in enumerate(unique_elements)}

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot atoms
for element in unique_elements:
    indices = [i for i, e in enumerate(elements) if e == element]
    ax.scatter(
        positions[indices, 0], positions[indices, 1], positions[indices, 2],
        label=element, color=color_map[element], alpha=0.5, s=60
    )

# Labels and visualization settings
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
ax.set_title("Visualization of Merged POSCAR Structure")
ax.legend()

# black background
# ax.set_facecolor('black')
# fig.patch.set_facecolor('black')

# set aspect ratio
ax.set_box_aspect([1, 1, 1])

# set view angle
ax.view_init(elev=0, azim=0)

# tight layout
plt.tight_layout()

# save
plt.savefig(dir_out + "/POSCAR_combined.png")