#!/usr/bin/env python3
"""
merge_vasp_runs.py

Merges XDATCAR frames, OUTCAR logs, and VASP vasprun.xml files from multiple runs
(e.g., BASEa, BASEb, BASEc, ... BASEz to BASEaa, BASEab, BASEac, ... BASEaz, to BASEba, BASEbb, and so on)
into a single set of files under the directory BASE.
If BASE does not exist, it is bootstrapped by copying BASEa.

Outputs in BASE/:
    XDATCAR     (merged trajectory)
    OUTCAR      (concatenated logs)
    vasprun.xml (merged XML with appended <calculation> blocks)

Usage:
    python $HELP_SCRIPTS_vasp/merge_vasp_runs.py BASE 
    python $LOCAL_HELP_SCRIPTS/qmd/vasp/merge_vasp_runs.py BASE
    python merge_vasp_runs.py SCALEE_7 --run_data_script 
"""
import os
import sys
import glob
import shutil
from ase.io import read, write
import subprocess
import argparse



parse = argparse.ArgumentParser(description="Merge VASP runs into a single set of files.")
# run data_4_analysis.sh script or not
parse.add_argument('base', type=str, help="Base directory name for merging runs (e.g., SCALEE_7)")
# add argument for --run_data_script
parse.add_argument('--run_data_script', "-r", action='store_true', default=1,
                    help="Run data_4_analysis.sh script after merging (default: 1, i.e., True)")
args = parse.parse_args()

# get base directory from args
if not args.base:
    print("Error: Base directory name is required.")
    sys.exit(1)
base = args.base
# check if base directory exists
flag_run_data_script = args.run_data_script
if flag_run_data_script == 1:
    flag_run_data_script = True
else:
    flag_run_data_script = False


# define data_4_analysis.sh script path
HELP_SCRIPTS_vasp = "/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/qmd/vasp"
data_4_analysis_script = os.path.join(HELP_SCRIPTS_vasp, 'data_4_analysis.sh')

def merge_runs(base):
    base_dir = base
    first_run = base + 'a'

    # 0) Check if first run directory exists and exit if not
    if not os.path.isdir(first_run):
        raise FileNotFoundError(f"First run directory '{first_run}' does not exist.")
    print(f"Base directory for merging: {base_dir}")

    # 1) Ensure BASE directory exists
    if not os.path.isdir(base_dir):
        if os.path.isdir(first_run):
            print(f"Directory '{base_dir}' not found. Copying '{first_run}' to '{base_dir}'.")
            shutil.copytree(first_run, base_dir)
        else:
            raise FileNotFoundError(f"Neither '{base_dir}' nor '{first_run}' exist.")


    # 2) Identify all run directories (BASEa, BASEb, ...)
    # runs = sorted(d for d in glob.glob(base + '?') if os.path.isdir(d)) # old way
    candidates = glob.glob(f"{base}[a-z]*")
    # sort by (suffix_length, full_name)
    runs = sorted(
        candidates,
        key=lambda d: (len(d) - len(base), d)
    )
    if not runs:
        raise RuntimeError(f"No run directories matching '{base}?' found.")
    print(f"Merging runs in the order: {runs}")

    # 3) Merge XDATCAR trajectories
    frames = []
    for run in runs:
        xdat_path = os.path.join(run, 'XDATCAR')
        if os.path.isfile(xdat_path):
            traj = read(xdat_path, index=':')
            frames.extend(traj)
        else:
            print(f"Warning: '{xdat_path}' not found; skipping.")
    out_xdat = os.path.join(base_dir, 'XDATCAR')
    write(out_xdat, frames, format='vasp-xdatcar')
    print(f"Written merged XDATCAR to '{out_xdat}'")

    # 4) Merge OUTCAR logs, appending later runs only after marker
    out_outcar = os.path.join(base_dir, 'OUTCAR')
    with open(out_outcar, 'w') as fout:
        for idx, run in enumerate(runs):
            outc_path = os.path.join(run, 'OUTCAR')
            if not os.path.isfile(outc_path):
                print(f"Warning: '{outc_path}' not found; skipping.")
                continue
            with open(outc_path) as fin:
                if idx == 0:
                    # first run: write entire OUTCAR
                    fout.write(fin.read())
                else:
                    # subsequent runs: append after marker line
                    append = False
                    for line in fin:
                        if append:
                            fout.write(line)
                        elif 'Fermi-smearing in eV' in line:
                            append = True
                    if not append:
                        print(f"Marker not found in '{outc_path}', appending entire file.")
                        fin.seek(0)
                        fout.write(fin.read())
            print(f"Appended OUTCAR from '{outc_path}'")
    print(f"Written merged OUTCAR to '{out_outcar}'")

    # 5) Merge vasprun.xml by concatenating <calculation> blocks
    xml_files = [os.path.join(run, 'vasprun.xml') for run in runs
                 if os.path.isfile(os.path.join(run, 'vasprun.xml'))]
    if xml_files:
        print(f"Merging XML files: {xml_files}")
        out_xml = os.path.join(base_dir, 'vasprun.xml')
        with open(out_xml, 'w') as fout:
            # Header from first file
            with open(xml_files[0]) as fin:
                for line in fin:
                    if line.strip().startswith('</'):
                        break
                    fout.write(line)
            # Append all <calculation> blocks
            for xml_file in xml_files:
                inside = False
                with open(xml_file) as fin:
                    for line in fin:
                        if '<calculation' in line:
                            inside = True
                        if inside:
                            fout.write(line)
                            if '</calculation>' in line:
                                inside = False
                print(f"Appended calculations from '{xml_file}'")
            # Determine root tag name
            root_tag = None
            with open(xml_files[0]) as fin:
                for line in fin:
                    if line.lstrip().startswith('<') and not line.lstrip().startswith('<?'):
                        root_tag = line.lstrip().split()[0].lstrip('<')
                        break
            fout.write(f"</{root_tag}>\n" if root_tag else '</modeling>\n')
        print(f"Written merged vasprun.xml to '{out_xml}'")
    else:
        print("No vasprun.xml found; skipping XML merge.")

    # 6) Execute data_4_analysis.sh if present in BASE

    if flag_run_data_script:
        print(f"flag_run_data_script is set to {flag_run_data_script}, running data_4_analysis.sh script")
        # go to base directory
        os.chdir(base_dir)
        # check if data_4_analysis.sh exists
        data_script = 'data_4_analysis.sh'
        # cp data_4_analysis.sh from data_4_analysis_script
        if os.path.isfile(data_4_analysis_script):
            shutil.copy(data_4_analysis_script, data_script)
            print(f"Updated '{data_4_analysis_script}' to '{data_script}'")
        else:
            print(f"Warning: '{data_4_analysis_script}' not found; skipping copy.")
        if os.path.isfile(data_script):
            print(f"Executing '{data_script}' via Bash")
            subprocess.run(['bash', data_script], check=True)
        else:
            print(f"Warning: '{data_script}' not found; skipping execution.")






if __name__ == '__main__':
    parent_dir = os.getcwd()
    print(f"Current working directory: {parent_dir}")

    if len(sys.argv) != 2:
        print("Usage: python merge_vasp_runs.py BASE")
        sys.exit(1)
    merge_runs(sys.argv[1])

    # go back to parent directory
    os.chdir(parent_dir)