#!/usr/bin/env python3

# Usage: python $HELP_SCRIPTS_vasp/continue_run_ase.py -r 500

import os
import sys
import time
import subprocess
from ase.io import read, write
import argparse
import shutil


parser = argparse.ArgumentParser(description="Continue a VASP run from the last structure in XDATCAR.")
parser.add_argument(
    "-r", "--restart_shift",
    type=int,
    default=100,
    help="Number of steps to go back in XDATCAR to find the last structure.",
)

args = parser.parse_args()
RESTART_SHIFT = args.restart_shift


def find_a_previous_structure(step_back=RESTART_SHIFT):
    """
    Read all images from XDATCAR, pick the one step_back before the end,
    and write it out as a new POSCAR.
    """
    traj = read("XDATCAR", index=":")        # list of Atoms
    n = len(traj)
    i = max(0, n - step_back)

    # backup old POSCAR
    shutil.copy("POSCAR", "POSCAR_old")

    write("POSCAR", traj[i], format="vasp", direct=True)
    print(f"> Restarting from image {i} of {n}")

    # create a log file as log.continue_run_ase
    with open("log.continue_run_ase", "w") as f:
        f.write(f"Restarting from image {i} of {n}\n")
        f.write(f"Last structure written to POSCAR\n")
        f.write(f"Step back: {step_back}\n")

    return n, i


find_a_previous_structure(step_back=RESTART_SHIFT)