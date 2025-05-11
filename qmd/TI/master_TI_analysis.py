#!/usr/bin/env python3
"""
Driver for analysing all SCALEE folders and writing TI_analysis.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from ase.io import read  # needed by get_n_atoms

import os
import pandas as pd
import numpy as np

# 1) abscissae
SCALEE = np.array([
    1.0,
    0.71792289,
    0.3192687,
    0.08082001,
    0.00965853,
    0.00035461,
    1.08469e-06
])

# matching folder names
DIRS = [f"SCALEE_{i}" for i in range(1, len(SCALEE)+1)]

# 2) quadrature weights
W = np.array([
    0.0357143,
    0.2107042,
    0.3411227,
    0.4124588,
    0.4124588,
    0.3411227,
    0.2107042       
])

def parse_peavg(path):
    """
    Given a path to analysis/peavg.out, return a dict with keys:
      'F', 'F_err', 'P', 'P_err', and optionally 'V' for volume.
    """
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Free" and parts[1] == "energy":
                # grep ... | awk '{print $4}'  → parts[3]
                out['F']      = float(parts[3])
                out['F_err']  = float(parts[5])
            elif parts[0] == "Pressure":
                # awk '{print $3,$5}'
                out['P']      = float(parts[2])
                out['P_err']  = float(parts[4])
            elif parts[0] == "Volume" and parts[1] == "of" and parts[2] == "cell":
                # awk '{print $11}'
                out['V']      = float(parts[10])
    return out




def get_n_atoms(base_dir="."):
    """
    Look in the first SCALEE folder for a VASP POSCAR or CONTCAR,
    read it with ASE, and return the number of atoms.
    """
    # try CONTCAR first, then POSCAR
    for fname in ("CONTCAR", "POSCAR"):
        candidate = os.path.join(base_dir, DIRS[0], fname)
        if os.path.isfile(candidate):
            atoms = read(candidate)
            return len(atoms)
    raise FileNotFoundError(
        f"Neither CONTCAR nor POSCAR found in {os.path.join(base_dir, DIRS[0])}"
    )




def process_all(base_dir="."):
    # collect lists
    Fs, F_errs, Ps, P_errs   = [], [], [], []
    Vs = []
    volume_cell = None

    for idx, (lam, d) in enumerate(zip(SCALEE, DIRS)):
        peavg = os.path.join(base_dir, d, "analysis", "peavg.out")
        data = parse_peavg(peavg)
        Fs.append(data['F'])
        F_errs.append(data['F_err'])
        # only read volume, pressure, pressure_error once (first abscissa)
        if idx == 0:
            volume_cell = data.get('V')
            P_SCALEE_1 = data['P']
            P_err_SCALEE_1 = data['P_err']
        Ps.append(P_SCALEE_1)
        P_errs.append(P_err_SCALEE_1)
        Vs.append(volume_cell)
    print(f"WARNING: Need to correct Pressure -> (Pressure + KPOINT 222 correction)")

    # for SCALEE_1, read volume, pressure, and pressure error
    # for idx, (lam, d) in enumerate(zip(SCALEE[-1], DIRS[-1])):
    #     peavg = os.path.join(base_dir, d, "analysis", "peavg.out")
    #     data = parse_peavg(peavg)
    #     # only read volume once (first abscissa)


    # build DataFrame
    df = pd.DataFrame({
        'DIR': DIRS,
        'scalee': SCALEE,
        'F @ OSZICAR': Fs,
        'F err @ OSZICAR': F_errs,
        'P (GPa)': Ps,
        'P_error (GPa)': P_errs,
        'V (Å³)': Vs,
        'P_target (GPa)': P_target,
        'T_target (K)': T_target
    })

    # same pipeline as before:
    df['ab-ig']    = df['F @ OSZICAR'] / df['scalee']
    df['err']      = df['F err @ OSZICAR'] / df['scalee']
    k = 0.8
    df['lambda^k'] = df['scalee'] ** k
    df['w']        = W
    df['err_scaled']    = df['err']    * df['lambda^k'] * df['w'] / 0.4
    df['deltaU_scaled'] = df['ab-ig'] * df['lambda^k'] * df['w'] / 0.4

    # total ΔU & its error
    ΔU_ab_m_ig     = df['deltaU_scaled'].sum()
    ΔU_ab_m_ig_err = np.sqrt((df['err_scaled']**2).sum())
    df['ΔU_ab_m_ig (eV)'] = ΔU_ab_m_ig
    df['ΔU_ab_m_ig_err (eV)'] = ΔU_ab_m_ig_err

    # PV → eV
    df['pv (eV)'] = (
        df['P (GPa)'].iloc[-1] * 1e9 *
        df['V (Å³)'].iloc[-1] * 1e-30 *
        6.242e18
    )
    # print(f"PV term: {df['pv (eV)'].iloc[-1]:.3f} eV")
    # print(f"P: {df['P (GPa)'].iloc[-1]:.3f} GPa, V: {df['V (Å³)'].iloc[-1]:.3f} Å³")

    df['pv_correction (eV)'] = (
        df['P_target (GPa)'].iloc[-1] * 1e9 *
        df['V (Å³)'].iloc[-1] * 1e-30 *
        6.242e18
    ) - df['pv (eV)']
    print(f"WARNING: WHY '+R11' in Ghp calculation? And, why not abs() of the resulting value here?")



    ΔU_ig = -503.181 # eV ## needs to be updated
    print("WARNING: ΔU_ig is hardcoded to -503.181 eV")
    df['ΔU_ig (eV)'] = ΔU_ig

    ΔU_hp_m_ab = -0.571 # eV ## needs to be updated
    print("WARNING: ΔU_hp_m_ab is hardcoded to -0.571 eV")
    df['ΔU_hp_m_ab (eV)'] = ΔU_hp_m_ab

    TS_term = 37.9997320160768 # eV ## needs to be updated
    print("WARNING: TS_term is hardcoded to 37.9997320160768 eV")
    df['TS_term (eV)'] = TS_term


    Ghp = ΔU_hp_m_ab + ΔU_ab_m_ig + ΔU_ig + df['pv (eV)'] + df['pv_correction (eV)']
    df['Ghp (eV)'] = Ghp
    Ghp_error = ΔU_ab_m_ig_err
    df['Ghp_error (eV)'] = Ghp_error

    # per atom Ghp_per_atom
    Ghp_per_atom = Ghp / n_atoms
    Ghp_per_atom_error = Ghp_error / n_atoms
    df['Ghp_per_atom (eV)'] = Ghp_per_atom
    df['Ghp_per_atom_error (eV)'] = Ghp_per_atom_error



    return df









# -----------------------------------------------------------------------------#
# ------------------------  Helper routines  ----------------------------------#
# -----------------------------------------------------------------------------#
def _read_keyword(file_path: Path, keyword: str) -> float:
    """
    Return the numeric value that follows KEYWORD either in

        KEYWORD=value          # e.g. PSTRESS_CHOSEN_GPa=10
    or  KEYWORD  value         # e.g. PSTRESS_CHOSEN_GPa   10

    Ignores blank lines and lines that start with '#'.
    Raises ValueError if the keyword is not found.
    """
    with open(file_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # --- 1. KEY=value style ----------------------------------------
            if '=' in line:
                k, v = line.split('=', 1)
                if k.strip() == keyword:
                    return float(v.strip())

            # --- 2. KEY  value style ---------------------------------------
            parts = line.split()
            if parts and parts[0] == keyword:
                if len(parts) < 2:
                    raise ValueError(
                        f"Found '{keyword}' but no value after it in {file_path}"
                    )
                return float(parts[1])

    raise ValueError(f"{keyword!r} not found in {file_path}")


def resolve_targets(
    base: str | None,
    p_target: float | None,
    t_target: float | None,
) -> tuple[Path, float, float]:
    """
    Resolve base directory, pressure (GPa) and temperature (K).

    Any of the three may be passed in as -1, None, or an empty string to request
    automatic lookup in   <base>/../master_setup_TI/input.calculate_GFE
    """
    # --- base directory -------------------------------------------------------
    if not base or str(base) == "-1":
        base_path = Path.cwd()
    else:
        base_path = Path(base).expanduser().resolve()

    if not base_path.is_dir():
        raise NotADirectoryError(f"Base directory does not exist: {base_path}")

    # --- auxiliary file with defaults ----------------------------------------
    aux_file = base_path.parent / "master_setup_TI" / "input.calculate_GFE"
    if not aux_file.is_file():
        raise FileNotFoundError(f"Auxiliary file not found: {aux_file}")

    # --- fill in any missing targets -----------------------------------------
    if p_target in (None, -1):
        p_target = _read_keyword(aux_file, "PSTRESS_CHOSEN_GPa")
    if t_target in (None, -1):
        t_target = _read_keyword(aux_file, "TEMP_CHOSEN")

    return base_path, float(p_target), float(t_target)


# -----------------------------------------------------------------------------#
# ------------------------------  main  ---------------------------------------#
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process all SCALEE folders and calculate free energy."
    )
    parser.add_argument(
        "-b", "--base",
        metavar="PATH",
        help="Top-level directory containing SCALEE_*/ subfolders "
            "(default: current working directory)."
    )
    parser.add_argument(
        "-p", "--pressure",
        metavar="GPa",
        type=float,
        default=None,
        help="Target pressure in GPa (omit or -1 to read from input.calculate_GFE)."
    )
    parser.add_argument(
        "-t", "--temperature",
        metavar="K",
        type=float,
        default=None,
        help="Target temperature in K (omit or -1 to read from input.calculate_GFE)."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ setup
    base_dir, P_target, T_target = resolve_targets(
        args.base, args.pressure, args.temperature
    )

    print(f"Base directory      : {base_dir}")
    print(f"Target pressure     : {P_target:.3f}  GPa")
    print(f"Target temperature  : {T_target:.1f}  K")

    # ------------------------------------------------------------------ run
    n_atoms = get_n_atoms(base_dir)          # <-- your helper must exist
    print(f"System size         : {n_atoms} atoms")

    print("")

    df = process_all(base_dir)               # <-- your helper must exist

    pd.set_option("display.max_columns", None)
    # print("\nResults\n-------")
    # print(df.to_string(index=False))

    out_file = base_dir / "TI_analysis.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file}")

# -----------------------------------------------------------------------------#
# if __name__ == "__main__":
#     main()