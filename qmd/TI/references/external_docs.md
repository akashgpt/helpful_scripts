# External Docs

The TI workflow here is mostly repo-specific, so local script comments are the primary reference. Use these external docs only for the file formats and helper libraries the workflow depends on.

## VASP

- INCAR: https://www.vasp.at/wiki/INCAR
- POSCAR: https://www.vasp.at/wiki/index.php/POSCAR
- KPOINTS: https://www.vasp.at/wiki/index.php/KPOINTS
- OUTCAR: https://www.vasp.at/wiki/index.php/OUTCAR
- NSW: https://www.vasp.at/wiki/NSW

## ASE

- ASE file I/O: https://wiki.fysik.dtu.dk/ase/ase/io/io.html

## When These Matter Here

- `calculate_GFE_v2.sh` and the VASP helpers assume standard `OUTCAR`, `POSCAR`, and `KPOINTS` behavior.
- `Ghp_analysis.py` uses ASE to inspect structures in the first `SCALEE_*` directory.
- The thermodynamic logic itself is better documented in `estimate_KD_generic_v8.py` and `Ghp_analysis.py` than in any generic external guide.

