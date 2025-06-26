#!/usr/bin/env python3
import os, sys
import argparse

# This script rearranges VASP run files with suffixes like BASEa, BASEb, etc.
# It renames them to BASEa, BASEb, BASEc, etc., removing any gaps in the sequence.
# Usage: python $HELP_SCRIPTS_vasp/RUN_VASP/continue_run__rearrange.py -b BASENAME

# BASENAME -- take in as command line argument for "-b"
parser = argparse.ArgumentParser(description="Rearrange VASP run files with suffixes.")
parser.add_argument('-b', '--basename', type=str, required=True, help="Base name for the files to rearrange.")
args = parser.parse_args()
BASENAME = args.basename
if not BASENAME:
    print("Error: Please provide a base name using -b or --basename.")
    sys.exit(1)

# 1. List and sort all entries that start with BASENAME but are not exactly BASENAME
items = sorted(
    entry for entry in os.listdir('.')
    if entry.startswith(BASENAME) and entry != BASENAME
)

# 2. Rename them a→b→c… so that gaps are removed
for idx, old in enumerate(items):
    new = f"{BASENAME}{chr(ord('a') + idx)}"
    if old != new:
        print(f"Renaming {old} → {new}")
        os.rename(old, new)

