from ase.io import read, write
import argparse

# Create an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser()

# Add arguments to the parser with their short and long option names and help strings
parser.add_argument("--LAMMPS_input","-li",default="conf.lmp",type=str,help="supplied: LAMMPS input file")
parser.add_argument("--VASP_input","-vi",default="POSCAR.ase",type=str,help="required: VASP input file")

# Parse the command-line arguments
args = parser.parse_args()

# Read the LAMMPS data file into an Atoms object.
atoms = read(args.LAMMPS_input, format='lammps-data')

# wrap atoms to unit cell
atoms.wrap()

# Write out a VASP POSCAR file.
write(args.VASP_input, atoms, format='vasp',direct=True)