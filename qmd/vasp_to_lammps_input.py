from ase.io import read, write
from ase.data import atomic_masses
import argparse

# Create an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser()

# Add arguments to the parser with their short and long option names and help strings
parser.add_argument("--VASP_input","-vi",default="POSCAR",type=str,help="supplied: VASP input file")
parser.add_argument("--LAMMPS_input","-li",default="conf.lmp.ase",type=str,help="required: LAMMPS input file")

# Parse the command-line arguments
args = parser.parse_args()

# Read the POSCAR file into an Atoms object.
atoms = read(args.VASP_input, format='vasp')

# wrap atoms to unit cell
atoms.wrap()

# Set masses for each atom.
atoms.set_masses([atomic_masses[atom.number] for atom in atoms])

# Write the basic LAMMPS data file.
write(args.LAMMPS_input, atoms, format='lammps-data',specorder=['Mg','Si','O','H','N'],masses=True)