from ase.io import read, write
from ase.data import atomic_masses

# Read the POSCAR file into an Atoms object.
atoms = read('POSCAR')

# Set masses for each atom.
atoms.set_masses([atomic_masses[atom.number] for atom in atoms])

# Write the basic LAMMPS data file.
write('conf.lmp.ase', atoms, format='lammps-data',specorder=['Mg','Si','O','H','N'],masses=True)