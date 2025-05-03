#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

# 1) abscissae
SCALEE = np.array([
    1.08469e-06,
    0.00035461,
    0.00965853,
    0.08082001,
    0.3192687,
    0.71792289,
    1.0
])
# matching folder names
DIRS = [f"SCALEE_{i}" for i in range(1, len(SCALEE)+1)]

# 2) quadrature weights
W = np.array([
    0.2107042,
    0.3411227,
    0.4124588,
    0.4124588,
    0.3411227,
    0.2107042,
    0.0357143
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
        Ps.append(data['P'])
        P_errs.append(data['P_err'])
        # only read volume once (first abscissa)
        if idx == 0:
            volume_cell = data.get('V')
        Vs.append(volume_cell)

    # build DataFrame
    df = pd.DataFrame({
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
    df['pv_correction (eV)'] = (
        df['P_error (GPa)'].iloc[-1] * 1e9 *
        df['V (Å³)'].iloc[-1] * 1e-30 *
        6.242e18
    ) - df['pv (eV)']



    ΔU_ig = -503.181 # eV ## needs to be updated
    df['ΔU_ig (eV)'] = ΔU_ig

    ΔU_hp_m_ab = -0.571 # eV ## needs to be updated
    df['ΔU_hp_m_ab (eV)'] = ΔU_hp_m_ab

    TS_term = 37.9997320160768 # eV ## needs to be updated
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







if __name__ == "__main__":
    import sys
    import argparse
    import os
    from ase.io import read


    parser = argparse.ArgumentParser(
        description="Process all SCALEE folders and calculate free energy."
    )
    parser.add_argument(
        "-b", "--base", type=str, default=".", help="Base directory for SCALEE folders."
    )
    # P_target and T_target parse
    parser.add_argument(
        "-p", "--pressure", type=float, default=1000.0, help="Target pressure in GPa."
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=13000.0, help="Target temperature in K."
    )
    args = parser.parse_args()
    base = args.base
    P_target = args.pressure
    T_target = args.temperature

    # P_target = 10 # GPa
    # T_target = 300 # K

    # use ASE to get the number of atoms
    n_atoms = get_n_atoms(base)

    df = process_all(base)

    pd.set_option('display.max_columns', None)
    print(df.to_string(index=False))
    # print(f"\n▶ Free energy    = {ΔU:.6f} eV")
    # print(f"▶ Free energy err= {ΔU_err:.6f} eV")

    # save to CSV
    output_file = os.path.join(base, "analysis", "TI_analysis.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")