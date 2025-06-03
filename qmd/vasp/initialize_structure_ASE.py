#!/usr/bin/env python3
"""
initialize_random_structure.py

Creates a random-packed, multi-element orthogonal or skew cell with PBC-aware placement,
then writes both VASP POSCAR and LAMMPS conf.lmp (with Masses) files.
"""

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, atomic_masses
from ase.io import write
import time


def pbc_distance(p1, p2, cell):
    """
    Shortest distance between p1 and p2 under PBC defined by 'cell'.
    """
    delta = p1 - p2
    # fractional delta
    frac = np.linalg.inv(cell) @ delta
    frac -= np.round(frac)
    # back to Cartesian
    delta = cell @ frac
    return np.linalg.norm(delta)


def make_random_packing(composition, cell,
                        min_dist=None, slack=0.9,
                        max_attempts=1000):
    """
    composition: dict, e.g. {'Fe': 10, 'O': 20}
    cell: array-like shape (3,3) for cell vectors
    min_dist: float or None; if None, auto-computed from v_atom**(1/3)
    slack: scale factor <=1.0 for auto min_dist
    """
    # flatten list of symbols
    symbols = []
    for el, cnt in composition.items():
        symbols.extend([el] * cnt)
    natoms = len(symbols)

    # ensure cell is array
    cell = np.array(cell, dtype=float).reshape(3,3)

    # auto-calc min_dist
    if min_dist is None:
        v_cell = abs(np.linalg.det(cell))
        v_atom = v_cell / natoms
        min_dist = (v_atom**(1/3))# * slack
        print(f"Auto-calculated min_dist: {min_dist:.2f} Å from V per atom={v_atom:.2f} Å³")
        min_dist *= slack
        print(f"Adjusted min_dist with slack {slack:.2f}: {min_dist:.2f} Å")
    current_min = min_dist
    print(f"Using min_dist: {current_min:.2f} Å")

    # packing loop
    while True:
        positions = []
        try:
            for i in range(natoms):
                for _ in range(max_attempts):
                    frac = np.random.random(3)
                    trial = cell.dot(frac)
                    if all(pbc_distance(trial, p, cell) >= current_min for p in positions):
                        positions.append(trial)
                        break
                else:
                    raise RuntimeError(f"placement failed for atom {i}")
            # if all placed, break
            break
        except RuntimeError:
            current_min *= 0.95
            print(f"Retry: min_dist reduced to {current_min:.2f} Å")
            if current_min < 0.1:
                raise RuntimeError(
                    f"min_dist below threshold; aborting (current_min={current_min:.2f} Å)"
                )

    atoms = Atoms(symbols=symbols,
                    positions=positions,
                    cell=cell,
                    pbc=True)
    return atoms



















if __name__ == "__main__":

    start_time = time.time()

    # user parameters
    # comp = {'Fe': 10, 'O': 20, 'Sc': 5}
    # comp = {'Fe': 128}
    comp = {'Mg': 64, "Si": 64, "O": 192, "H":80}  # MgSiO3 perovskite + H
    # define non-cubic cell matrix
    cell_L = 9.373
    cell = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0]]
    # cell = [[1.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0],
    #         [0.0, 0.0, 1.0]] # cubic cell
    slack = 0.9

    print("")
    print(f"Creating random packing with composition: {comp}")
    print(f"Total # of atoms: {sum(comp.values())}")
    print(f"Cell characteristic length: {cell_L:.3f} Å")
    print(f"Cell matrix:{cell}")
    print(f"Slack factor: {slack:.2f}")
    print("")


    cell = np.array(cell, dtype=float) * cell_L


    atoms = make_random_packing(comp, cell,
                                min_dist=None,
                                slack=slack)
    print("")

    write("POSCAR", atoms,
            format="vasp", direct=True)
    print("Written POSCAR")

    syms = atoms.get_chemical_symbols()
    unique_syms = []
    for s in syms:
        if s not in unique_syms:
            unique_syms.append(s)
    masses = {sym: float(atomic_masses[atomic_numbers[sym]]) for sym in unique_syms}

    write("conf.lmp", atoms,
            format="lammps-data",
            atom_style="atomic",
            masses=masses,
            specorder=unique_syms)
    print("Written conf.lmp")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    print("")