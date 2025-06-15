#!/usr/bin/env python3
import os, sys
import argparse

# BASENAME -- take in as command line argument for "-b"
parser = argparse.ArgumentParser(description="Rearrange VASP run files with suffixes.")
parser.add_argument('-b', '--basename', type=str, required=True, help="Base name for the files to rearrange.")
args = parser.parse_args()
BASENAME = args.basename
if not BASENAME:
    print("Error: Please provide a base name using -b or --basename.")
    sys.exit(1)

# list & sort only the suffixed files
files = sorted(f for f in os.listdir('.') if f.startswith(BASENAME) and f != BASENAME)
for idx, old in enumerate(files):
    new = f"{BASENAME}{chr(ord('a')+idx)}"
    if old != new:
        print(f"{old} → {new}")
        os.rename(old, new)