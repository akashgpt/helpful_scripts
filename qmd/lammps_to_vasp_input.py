from ase.io import read, write


# Read the LAMMPS data file into an Atoms object.
atoms = read('conf.lmp', format='lammps-data')


# Write out a VASP POSCAR file.
write('POSCAR.ase', atoms, format='vasp',direct=True)