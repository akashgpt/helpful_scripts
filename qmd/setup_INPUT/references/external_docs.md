# External Docs

## ASE

- ASE file I/O: https://wiki.fysik.dtu.dk/ase/ase/io/io.html

## VASP

- POSCAR: https://www.vasp.at/wiki/index.php/POSCAR
- POTCAR: https://www.vasp.at/wiki/index.php/POTCAR

## LAMMPS

- Manual landing page: https://docs.lammps.org/
- `read_data`: https://docs.lammps.org/read_data.html
- `dump`: https://docs.lammps.org/dump.html
- `units`: https://docs.lammps.org/units.html

## Why These Matter Here

- The join and conversion scripts use ASE to read and write both VASP and LAMMPS formats.
- Species ordering must stay consistent across `POSCAR`, `POTCAR`, and any LAMMPS type labels.
- `join_conf_lmps.py` preserves LAMMPS-data semantics more carefully than a naive text concatenation, so use the LAMMPS docs when extending it.
