from ase.io import read, write
from ase.data import atomic_masses
import argparse

# Usage: python $HELP_SCRIPTS_vasp/remove_negative_values_in_POSCAR.py --VASP_input POSCAR --LAMMPS_input conf.lmp.ase

# Create an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser()

# Add arguments to the parser with their short and long option names and help strings
parser.add_argument("--VASP_input","-i",default="POSCAR",type=str,help="supplied: VASP input file")
parser.add_argument("--VASP_output","-o",default="POSCAR.ase",type=str,help="required: VASP output file")


# Parse the command-line arguments
args = parser.parse_args()

# Read the POSCAR file into an Atoms object.
atoms = read(args.VASP_input, format='vasp')

# wrap atoms to unit cell
atoms.wrap()

# rename POSCAR as old.POSCAR
import os
os.rename(args.VASP_input, "old.POSCAR")

# Write out a VASP POSCAR file.
write(args.VASP_output, atoms, format='vasp',direct=True)
write("POSCAR", atoms, format='vasp',direct=True)
