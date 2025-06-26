#!/usr/bin/env python3

# Usage:                       nohup python $HELP_SCRIPTS_TI/GhP_analysis.py > log.GhP_analysis 2>&1 &
# or (if no master_setup_TI)   python GhP_analysis.py -b <base_dir> -p <pressure> -t <temperature>

"""
Driver for analysing all SCALEE folders and writing TI_analysis.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from ase.io import read  # needed by get_atomic_system_info

import os
import pandas as pd
import numpy as np
import scipy.special
import matplotlib.pyplot as plt


# parameters
H_mass    = 1.66054e-27 # kg
ev2j      = 1.60218e-19
boltz_ev  = 8.61733e-5 #eV/K
boltz     = boltz_ev*ev2j
avogadro  = 6.022e23
planck    = 6.62607e-34 #si, j/hz
planck_ev = planck/boltz_ev

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
                # print(f"Free energy: {parts}")
                # print(f"paths: {path}")
                out['F']      = float(parts[4])
                out['F_err']  = float(parts[6])
            elif parts[0] == "Pressure":
                # awk '{print $3,$5}'
                out['P']      = float(parts[2])
                out['P_err']  = float(parts[4])
            elif parts[0] == "Volume" and parts[1] == "of" and parts[2] == "cell":
                # awk '{print $11}'
                out['V']      = float(parts[10])
    return out



def _energy(atomic_mass, natoms,vol,temp):
    """
    Input 
    atomic mass : atomic mass #
    natoms     : # of atoms
    vol   : total volume in Ang^3
    temp  : temperature in K
    ------
    Return
    F : Helmholtz free energy in eV
    ------
    
    benchmark example from Dorner 2017, page 183
    atomic_mass  = 28.085
    natoms       = 64
    vol          = 20.49*1e-30*natoms
    temp         = 1687
    inverse_temp = 1/boltz/temp
    thermal_lambda = planck/(2*np.pi*H_mass*atomic_mass/inverse_temp)**.5
    F              = -1/inverse_temp*natoms*(np.log(vol/(thermal_lambda**3)/natoms) + 1)

    Author: Jie Deng & Haiyang Luo
    """
    vol = vol*1e-30
    inverse_temp = 1/boltz/temp
    thermal_lambda = planck/(2*np.pi*H_mass*atomic_mass/inverse_temp)**.5
    #F              = -1/inverse_temp*natoms*(np.log(vol/(thermal_lambda**3)/natoms) + 1) #Stirling approximation
    F              = -1/inverse_temp*(natoms*np.log(vol/thermal_lambda**3)-scipy.special.gammaln(natoms+1)) #exact equation
    return F/ev2j

def estimate_IdealGas_Helmholtz_Free_Energy(atomic_mass, natoms,vol,temp):
    """
    wrapper function of _energy to handle >=1 element case
    
    benchmark example from Yuan & Steinle‐Neumann, 2020, Table S2
    ---------
    atomic_mass = 55.845
    natoms = 50
    temp = 4000
    vol = 424.19
    hf = estimate_IdealGas_Helmholtz_Free_Energy(atomic_mass, natoms,vol,temp)
    print(hf)

    vol = 485.1
    natoms = np.array([15, 15, 45])
    atomic_mass = np.array([24.305, 28.085, 15.999])

    hf = estimate_IdealGas_Helmholtz_Free_Energy(atomic_mass, natoms,vol,temp)
    print(hf)
    
    """
    try:
        nele     = len(atomic_mass)
        vol4each = vol*(np.ones(nele))#*(np.array(natoms)/sum(natoms))
        f4each   = np.zeros(nele)
        for i in range(nele):
            f4each[i] = _energy(atomic_mass[i], natoms[i], vol4each[i],temp) 
            f = sum(f4each)
    except:
        f = _energy(atomic_mass, natoms,vol,temp)
    return f



def estimate_TS(natoms,temp):
    nele     = len(natoms)
    lnfactorial4each   = np.zeros(nele)
    for i in range(nele):
            lnfactorial4each[i] = scipy.special.gammaln(natoms[i]+1)
    lnfactorial = sum(lnfactorial4each)
    total_atoms = sum(natoms)
    lnfactorial_total = scipy.special.gammaln(total_atoms+1)
    TS = boltz_ev*temp*(lnfactorial_total-lnfactorial)
    return TS



import os
from ase.io import read
from collections import Counter

def get_atomic_system_info(base_dir="."):
    """
    Look in the first SCALEE folder (using DIRS[0]) for a VASP POSCAR or CONTCAR,
    read it with ASE, and return:
      1) total number of atoms (int),
      2) list of atom counts per species,
      3) list of atomic masses per species.

    Species order matches first appearance in the structure.
    """
    # Require a global DIRS list to locate the first folder
    try:
        first_dir = DIRS[0]
    except Exception:
        raise ValueError("DIRS must be defined and contain at least one directory name.")

    # Try CONTCAR then POSCAR
    for fname in ("CONTCAR", "POSCAR"):
        candidate = os.path.join(base_dir, first_dir, fname)
        if os.path.isfile(candidate):
            atoms = read(candidate)
            symbols = atoms.get_chemical_symbols()
            masses = atoms.get_masses()

            # Count atoms per species
            counts = Counter(symbols)
            # Preserve order of first appearance
            unique_species = []
            for sym in symbols:
                if sym not in unique_species:
                    unique_species.append(sym)

            atom_counts = [counts[sym] for sym in unique_species]
            # Map each species to its mass (first occurrence)
            mass_map = {}
            for sym, m in zip(symbols, masses):
                if sym not in mass_map:
                    mass_map[sym] = m
            atomic_masses = [mass_map[sym] for sym in unique_species]

            total_atoms = len(atoms)
            return total_atoms, atom_counts, atomic_masses, unique_species

    # If neither file is found
    raise FileNotFoundError(
        f"Neither CONTCAR nor POSCAR found in {os.path.join(base_dir, first_dir)}"
    )





def process_all(base_dir="."):
    # collect lists
    Fs, F_errs, Ps, P_errs, Ps_KP2_corr   = [], [], [], [], []
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

            file_pressure_correction = os.path.join(base_dir, d, "analysis", "pressure_correction.dat") # file units: kBar
            if not os.path.isfile(file_pressure_correction):
                print("")
                print(f"WARNING: {file_pressure_correction} not found. Assuming no pressure correction.")
                print("")
                P_SCALEE_1_corr = 0.0
            else:
                # read second line of file_pressure_correction
                P_SCALEE_1_corr = np.loadtxt(file_pressure_correction, skiprows=1)
                #P_SCALEE_1_corr make float from array
                P_SCALEE_1_corr = float(P_SCALEE_1_corr)
                P_SCALEE_1_corr = P_SCALEE_1_corr * 0.1 # convert from kBar to GPa
                # print(f"Pressure correction: {P_SCALEE_1_corr:.3f} GPa")

        Ps.append(P_SCALEE_1)
        P_errs.append(P_err_SCALEE_1)
        Ps_KP2_corr.append(P_SCALEE_1_corr)
        Vs.append(volume_cell)

    # print(f"WARNING: Need to correct Pressure -> (Pressure + KPOINT 222 correction)")

    P_target_arr = P_target * np.ones(len(DIRS))
    T_target_arr = T_target * np.ones(len(DIRS))

    Ps_corrected = [p1 + p2 for p1, p2 in zip(Ps, Ps_KP2_corr)] # SCALEE_1 + KPOINT 222 correction
    # print(f"P_corrected: {P_corrected}")
    # print(f"Ps: {Ps}")
    # print(f"Ps_KP2_corr: {Ps_KP2_corr}")


    # for SCALEE_1, read volume, pressure, and pressure error
    # for idx, (lam, d) in enumerate(zip(SCALEE[-1], DIRS[-1])):
    #     peavg = os.path.join(base_dir, d, "analysis", "peavg.out")
    #     data = parse_peavg(peavg)
    #     # only read volume once (first abscissa)

    # print(f" shape of following df: DIR: {np.shape(DIRS)}, SCALEE: {np.shape(SCALEE)}, "
    #         f"Fs: {np.shape(Fs)}, F_errs: {np.shape(F_errs)}, Ps: {np.shape(Ps)}, P_errs: {np.shape(P_errs)}, "
    #         f"Ps_KP2_corr: {np.shape(Ps_KP2_corr)}, Vs: {np.shape(Vs)}, "
    #         f"P_target: {np.shape(P_target_arr)}, T_target: {np.shape(T_target_arr)}, Ps_corrected: {np.shape(Ps_corrected)}")

    # cols = {
    #     'DIR': DIRS,
    #     'SCALEE': SCALEE,
    #     'Fs': Fs,
    #     'F_errs': F_errs,
    #     'Ps': Ps,
    #     'P_errs': P_errs,
    #     'Ps_KP2_corr': Ps_KP2_corr,
    #     'Vs': Vs,
    #     'P_target': P_target_arr,        # whatever you passed here
    #     'T_target (K)': T_target_arr,     # and here
    #     'Ps_corrected': Ps_corrected
    # }
    # for name, arr in cols.items():
    #     try:
    #         length = len(arr)
    #     except TypeError:
    #         length = "scalar!"
    #     print(f"{name:12s} → {length}")


    # build DataFrame
    df = pd.DataFrame({
        'DIR': DIRS,
        'SCALEE': SCALEE,
        'TOTEN (eV)': Fs,
        'TOTEN (eV)_error (eV)': F_errs,
        'P_KP1 (GPa)': Ps,
        'P (GPa)': Ps_corrected,
        'P_error (GPa)': P_errs,
        'P_KP2_corr (GPa)': Ps_KP2_corr,
        'V (Å³)': Vs,
        'P_target (GPa)': P_target_arr,
        'T_target (K)': T_target_arr
    })

    # same pipeline as before:
    df['ab-ig']    = df['TOTEN (eV)']# / df['SCALEE']
    df['err']      = df['TOTEN (eV)_error (eV)']# / df['SCALEE']
    k = 0.8
    df['lambda^k'] = df['SCALEE'] ** k
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
    print(f"WARNING: WHY '+R11' in Ghp calculation? This is correct, right?")



    # HF_ig = -503.181 # eV ## needs to be updated
    HF_ig = estimate_IdealGas_Helmholtz_Free_Energy(atomic_masses, atom_counts, volume_cell, T_target)
    # # type
    # print(f"type of HF_ig: {type(HF_ig)}")
    # print(f"HF_ig: {HF_ig:.3f} eV")
    # print("WARNING: HF_ig is hardcoded to -503.181 eV")
    df['HF_ig (eV)'] = HF_ig

    # TS_term = 37.9997320160768 # eV ## needs to be updated
    TS_term = estimate_TS(atom_counts, T_target) # eV
    # # type
    # print(f"type of TS_term: {type(TS_term)}")
    # print(f"TS_term: {TS_term:.3f} eV")
    # print("WARNING: TS_term is hardcoded to 37.9997320160768 eV (not being used right now though)")
    df['TS_term (eV)'] = TS_term


    # ΔHF_hp_m_ab = -0.571 # eV ## needs to be updated
    # print("WARNING: ΔHF_hp_m_ab is hardcoded to -0.571 eV. Not needed for my calculations, right? Not being used right now though")
    ΔHF_hp_m_ab = 0.0
    df['ΔHF_hp_m_ab (eV)'] = ΔHF_hp_m_ab


    # Ghp = ΔHF_hp_m_ab + ΔU_ab_m_ig + HF_ig + df['pv (eV)'] + df['pv_correction (eV)']
    Ghp = ΔU_ab_m_ig + HF_ig + df['pv (eV)'] + df['pv_correction (eV)']
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

    print("")
    print(f"\nSetup\n-----")
    print(f"Base directory      : {base_dir}")
    print(f"Directory name      : {base_dir.name}")
    print(f"Target pressure     : {P_target:.3f}  GPa")
    print(f"Target temperature  : {T_target:.1f}  K")

    # ------------------------------------------------------------------ run
    n_atoms, atom_counts, atomic_masses, unique_species = get_atomic_system_info(base_dir)     # get number of atoms from first SCALEE folder     
    # convert atomic_masses and atom_counts to np.array
    atomic_masses = np.array(atomic_masses)
    atom_counts = np.array(atom_counts)
    print(f"Unique species      : {unique_species}")
    print(f"Atom counts         : {atom_counts}")
    print(f"Atomic masses       : {atomic_masses}")
    print(f"Total # of atoms    : {n_atoms}")
    # print(f"type of all output variables: {type(n_atoms)}, {type(atom_counts)}, {type(atomic_masses)}, {type(unique_species)}")

    print("")

    df = process_all(base_dir) # process all SCALEE folders

    print("")
    print("\nResults Summary (in eV unless mentioned)\n-----------------------")
    print(f"G_hp                : {df['Ghp (eV)'].iloc[0]:.3f}") # all values are the same
    print(f"G_hp_error          : {df['Ghp_error (eV)'].iloc[0]:.3f}") # all values are the same
    print(f"G_hp_per_atom       : {df['Ghp_per_atom (eV)'].iloc[0]:.3f}") # all values are the same
    print(f"G_hp_per_atom_error : {df['Ghp_per_atom_error (eV)'].iloc[0]:.3f}") # all values are the same
    print(f"HF_ig               : {df['HF_ig (eV)'].iloc[0]:.3f}") # all values are the same
    print(f"TS                  : {df['TS_term (eV)'].iloc[0]:.3f}") # all values are the same
    print(f"Volume (Å³)         : {df['V (Å³)'].iloc[0]:.3f}") # all values are the same
    print(f"Volume per atom (Å³): {df['V (Å³)'].iloc[0]/n_atoms:.3f}") # all values are the same
    print("")

    pd.set_option("display.max_columns", None)
    # print("\nResults\n-------")
    # print(df.to_string(index=False))

    out_file = base_dir / "TI_analysis.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file}")
    print("")

# -----------------------------------------------------------------------------#
# if __name__ == "__main__":
#     main()