from ase.build import molecule
import math
import numpy as np
from ase.io import write

# ----------------------------------------------------------------------------------
# Create NH₃ Unit Cell (with a defined cell)
# ----------------------------------------------------------------------------------
# Create an NH₃ molecule and then assign it a unit cell.
chosen_molecule_str = 'NH3'
# Adjust the cell to be cubic with a specific edge length.
edge_length = 12.049  # adjust this value as needed
# total number of molecules in the supercell
molecules_target = 27

# Create the molecule and center it in its cell.
chosen_molecule = molecule(chosen_molecule_str)
chosen_molecule.center()

# center the molecule in the unit cell
chosen_molecule.set_positions(chosen_molecule.get_positions() - np.mean(chosen_molecule.get_positions(), axis=0))

# Define a cubic unit cell for the molecule
a = ((edge_length**3) / molecules_target) ** (1/3)  # Calculate the cubic lattice parameter so as to maximize the volume of a cell given total volume and number of molecules
print(f"Calculated lattice parameter a: {a:.2f} Å")
chosen_molecule.set_cell([[a, 0, 0], [0, a, 0], [0, 0, a]])
chosen_molecule.set_pbc(True)  # Enable periodic boundary conditions

# Print number of atoms in the primitive cell (for H2O, expected: 3 atoms)
print(f"Atoms in primitive cell: {len(chosen_molecule)}")

# ----------------------------------------------------------------------------------
# Build a Supercell with a Target Number of H2O Molecules
# ----------------------------------------------------------------------------------
target_atoms = len(chosen_molecule) * molecules_target  # total number of atoms

# Calculate the minimum cubic replication factor needed to reach at least target_atoms
n = math.ceil((target_atoms / len(chosen_molecule)) ** (1 / 3))
big = chosen_molecule.repeat((n, n, n))
print(f"Atoms in supercell before trimming: {len(big)}")

# ----------------------------------------------------------------------------------
# Adjust the Cell Dimensions and Scale Atom Positions
# ----------------------------------------------------------------------------------
# Instead of directly modifying the diagonal values, create a new cell,
# and use set_cell with scale_atoms=True so that atoms are uniformly rescaled.
new_cell = np.diag([edge_length, edge_length, edge_length])
# big.set_cell(new_cell, scale_atoms=True)
big.set_cell(new_cell, scale_atoms=False) # Here, setting scale_atoms=False ensures that the molecules keep their internal geometry unchanged—you’re just placing them in a larger cell.
print(f"New cell dimensions:\n{big.cell}")

# Optional: Trim any extra atoms if the supercell exceeds target_atoms.
if len(big) > target_atoms:
    big = big[np.arange(target_atoms)]
print(f"Total atoms in supercell: {len(big)}")

# ----------------------------------------------------------------------------------
# Sort the Atoms by Atomic Number (if desired) and Invert Order if Needed
# ----------------------------------------------------------------------------------
big = big[np.argsort(big.numbers)]
big = big[::-1]



# ----------------------------------------------------------------------------------
# Write the POSCAR file using ASE's built-in VASP5 writer.
# ----------------------------------------------------------------------------------
write(f'POSCAR_{chosen_molecule_str}', big, format='vasp', direct=True)