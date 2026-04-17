# External Docs

Use these official references when local script behavior depends on VASP or ASE file semantics.

## VASP Wiki

- INCAR: https://www.vasp.at/wiki/INCAR
- POSCAR: https://www.vasp.at/wiki/index.php/POSCAR
- KPOINTS: https://www.vasp.at/wiki/index.php/KPOINTS
- POTCAR: https://www.vasp.at/wiki/index.php/POTCAR
- OUTCAR: https://www.vasp.at/wiki/index.php/OUTCAR
- NSW: https://www.vasp.at/wiki/NSW
- ISMEAR: https://www.vasp.at/wiki/index.php/ISMEAR

## ASE

- ASE file I/O: https://wiki.fysik.dtu.dk/ase/ase/io/io.html

## When These Matter Here

- `continue_run_ase.py` and `merge_vasp_runs.py` rely on ASE read/write behavior for `XDATCAR`, `POSCAR`, and `vasp-xdatcar`.
- `data_4_analysis.sh` and the TI helpers rely on stable `OUTCAR` text patterns.
- Any change to species order must respect the `POSCAR` to `POTCAR` ordering rules from the VASP Wiki.

