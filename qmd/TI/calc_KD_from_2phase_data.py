#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

** Commented using ChatGPT **

Compute distribution/partitioning metrics (D_wt, K_D, and site fractions X)
for He, H, and H₂ between Fe (metal) and MgSiO3 (silicate), propagating input
uncertainties via Monte Carlo with asymmetric bounds.

Notes:
- Inputs are means and 1σ-like spreads (interpreted as +/- for sampling bounds).
- All N_* are NUMBER OF ATOMS (or molecules for H₂ cases) in the counted phase.
- μ_* are molar masses (g/mol). D_wt uses mass fractions, K_D uses atomic/molecular fractions.
- The mc_error.monte_carlo_error_asymmetric callable is expected to:
    (a) sample each parameter within [lower, upper],
    (b) evaluate fn(params) over many draws,
    (c) return central estimate and asymmetric CI (lower, upper).
"""

import re
import ast
from pathlib import Path
import numpy as np
import sys
import argparse
import os

# Import MC error propagation helper
from mc_error import monte_carlo_error_asymmetric  # located at $HELP_SCRIPTS/general/mc_error.py


# --------------------------
# He CASE: means and spreads
# --------------------------

# Mean counts with ~1σ spreads (interpretation depends on mc_error implementation).
mean_N_He_in_Fe     = 8.957
sigma_N_He_in_Fe    = 1.551

mean_N_He_in_MgSiO3 = 55.261
sigma_N_He_in_MgSiO3= 4.002

# Matrix atom counts (Fe in metal, MgSiO3 in silicate)
mean_N_Fe           = 96.652
sigma_N_Fe          = 1.799

mean_N_MgSiO3       = 57.652  # Using Mg as a proxy for formula count (comment retained)
sigma_N_MgSiO3      = 1.799

# Molar masses (g/mol). Used only for D_wt (mass-basis partition coefficient).
mu_He     = 4.002602
mu_Fe     = 55.845
mu_MgSiO3 = 100.388


# --------------------------
# He CASE: derived functions
# --------------------------

def fn_D_wt(N_He_in_Fe, N_He_in_MgSiO3, N_Fe, N_MgSiO3):
    """
    Mass-based partition coefficient:
        D_wt = (w_He_in_Fe) / (w_He_in_MgSiO3)

    w_frac_in_phase = (N_He * μ_He) / (N_He * μ_He + N_matrix * μ_matrix)
    """
    w_frac_in_Fe = (N_He_in_Fe * mu_He) / (N_He_in_Fe * mu_He + N_Fe * mu_Fe)
    w_frac_in_MgSiO3 = (N_He_in_MgSiO3 * mu_He) / (N_He_in_MgSiO3 * mu_He + N_MgSiO3 * mu_MgSiO3)
    D_wt = w_frac_in_Fe / w_frac_in_MgSiO3
    return D_wt

def fn_K_D(N_He_in_Fe, N_He_in_MgSiO3, N_Fe, N_MgSiO3):
    """
    Atom-fraction-based partition coefficient (distribution coefficient):
        K_D = X_He_in_Fe / X_He_in_MgSiO3
    where X_He_in_phase = N_He / (N_He + N_matrix_atoms)
    """
    X_frac_in_Fe = N_He_in_Fe / (N_He_in_Fe + N_Fe)
    X_frac_in_MgSiO3 = N_He_in_MgSiO3 / (N_He_in_MgSiO3 + N_MgSiO3)
    K_D = X_frac_in_Fe / X_frac_in_MgSiO3
    return K_D

def fn_X_He_in_MgSiO3(N_He_in_MgSiO3, N_MgSiO3):
    """ He atomic fraction in silicate. """
    return N_He_in_MgSiO3 / (N_He_in_MgSiO3 + N_MgSiO3)

def fn_X_He_in_Fe(N_He_in_Fe, N_Fe):
    """ He atomic fraction in metal. """
    return N_He_in_Fe / (N_He_in_Fe + N_Fe)


# --------------------------
# He CASE: Monte Carlo stats
# --------------------------

D_wt, D_wt_err, D_wt_lower, D_wt_upper = monte_carlo_error_asymmetric(
    fn_D_wt,
    # central values (tuple)
    (mean_N_He_in_Fe, mean_N_He_in_MgSiO3, mean_N_Fe, mean_N_MgSiO3),
    # lower bounds for each parameter
    (mean_N_He_in_Fe - sigma_N_He_in_Fe,
     mean_N_He_in_MgSiO3 - sigma_N_He_in_MgSiO3,
     mean_N_Fe - sigma_N_Fe,
     mean_N_MgSiO3 - sigma_N_MgSiO3),
    # upper bounds for each parameter
    (mean_N_He_in_Fe + sigma_N_He_in_Fe,
     mean_N_He_in_MgSiO3 + sigma_N_He_in_MgSiO3,
     mean_N_Fe + sigma_N_Fe,
     mean_N_MgSiO3 + sigma_N_MgSiO3),
)

K_D, K_D_err, K_D_lower, K_D_upper = monte_carlo_error_asymmetric(
    fn_K_D,
    (mean_N_He_in_Fe, mean_N_He_in_MgSiO3, mean_N_Fe, mean_N_MgSiO3),
    (mean_N_He_in_Fe - sigma_N_He_in_Fe,
     mean_N_He_in_MgSiO3 - sigma_N_He_in_MgSiO3,
     mean_N_Fe - sigma_N_Fe,
     mean_N_MgSiO3 - sigma_N_MgSiO3),
    (mean_N_He_in_Fe + sigma_N_He_in_Fe,
     mean_N_He_in_MgSiO3 + sigma_N_He_in_MgSiO3,
     mean_N_Fe + sigma_N_Fe,
     mean_N_MgSiO3 + sigma_N_MgSiO3),
)

X_He_in_MgSiO3, X_He_in_MgSiO3_err, X_He_in_MgSiO3_lower, X_He_in_MgSiO3_upper = monte_carlo_error_asymmetric(
    fn_X_He_in_MgSiO3,
    (mean_N_He_in_MgSiO3, mean_N_MgSiO3),
    (mean_N_He_in_MgSiO3 - sigma_N_He_in_MgSiO3, mean_N_MgSiO3 - sigma_N_MgSiO3),
    (mean_N_He_in_MgSiO3 + sigma_N_He_in_MgSiO3, mean_N_MgSiO3 + sigma_N_MgSiO3),
)

X_He_in_Fe, X_He_in_Fe_err, X_He_in_Fe_lower, X_He_in_Fe_upper = monte_carlo_error_asymmetric(
    fn_X_He_in_Fe,
    (mean_N_He_in_Fe, mean_N_Fe),
    (mean_N_He_in_Fe - sigma_N_He_in_Fe, mean_N_Fe - sigma_N_Fe),
    (mean_N_He_in_Fe + sigma_N_He_in_Fe, mean_N_Fe + sigma_N_Fe),
)

print("\n\nFor He:")
print(f"D_wt = {D_wt:.3f}, D_wt_lower = {D_wt_lower:.3f}, D_wt_upper = {D_wt_upper:.3f}")
print(f"K_D = {K_D:.3f}, K_D_lower = {K_D_lower:.3f}, K_D_upper = {K_D_upper:.3f}")
print(f"X_He_in_MgSiO3 = {X_He_in_MgSiO3:.5f}, X_He_in_MgSiO3_lower = {X_He_in_MgSiO3_lower:.5f}, X_He_in_MgSiO3_upper = {X_He_in_MgSiO3_upper:.5f}")
print(f"X_He_in_Fe = {X_He_in_Fe:.5f}, X_He_in_Fe_lower = {X_He_in_Fe_lower:.5f}, X_He_in_Fe_upper = {X_He_in_Fe_upper:.5f}")



# --------------------------
# H / H₂ CASE: means & spreads
# --------------------------

# Raw H atom counts & 1σ
mean_N_H_in_Fe       = 38.522
sigma_N_H_in_Fe      = 5.151

# Convert to H₂ molecule counts assuming H atoms pair as H₂
# (If your model uses atomic H activity, keep the H-atom form instead.)
mean_N_H2_in_Fe      = mean_N_H_in_Fe / 2
sigma_N_H2_in_Fe     = sigma_N_H_in_Fe / 2

mean_N_H_in_MgSiO3   = 16.957
sigma_N_H_in_MgSiO3  = 3.198

mean_N_H2_in_MgSiO3  = mean_N_H_in_MgSiO3 / 2
sigma_N_H2_in_MgSiO3 = sigma_N_H_in_MgSiO3 / 2

# Matrix counts (may differ from He case; keep separate)
mean_N_Fe            = 89.130
sigma_N_Fe           = 3.946

mean_N_MgSiO3        = 52.957  # Using Mg as a proxy for formula count (comment retained)
sigma_N_MgSiO3       = 2.205

# Molar masses for H-based D_wt
mu_H      = 1.00784
mu_Fe     = 55.845
mu_MgSiO3 = 100.388


# --------------------------
# H / H₂ CASE: derived funcs
# --------------------------

def fn_D_wt(N_H_in_Fe, N_H_in_MgSiO3, N_Fe, N_MgSiO3):
    """
    D_wt for atomic H based on mass fractions of H in each phase.
    """
    w_frac_in_Fe = (N_H_in_Fe * mu_H) / (N_H_in_Fe * mu_H + N_Fe * mu_Fe)
    w_frac_in_MgSiO3 = (N_H_in_MgSiO3 * mu_H) / (N_H_in_MgSiO3 * mu_H + N_MgSiO3 * mu_MgSiO3)
    return w_frac_in_Fe / w_frac_in_MgSiO3

def fn_D_wt_H2(N_H2_in_Fe, N_H2_in_MgSiO3, N_Fe, N_MgSiO3):
    """
    D_wt for molecular H₂ based on mass fractions (2*μ_H).
    """
    w2_frac_in_Fe = (N_H2_in_Fe * 2 * mu_H) / (N_H2_in_Fe * 2 * mu_H + N_Fe * mu_Fe)
    w2_frac_in_MgSiO3 = (N_H2_in_MgSiO3 * 2 * mu_H) / (N_H2_in_MgSiO3 * 2 * mu_H + N_MgSiO3 * mu_MgSiO3)
    return w2_frac_in_Fe / w2_frac_in_MgSiO3

def fn_K_D(N_H_in_Fe, N_H_in_MgSiO3, N_Fe, N_MgSiO3):
    """
    K_D for atomic H: ratio of atomic fractions of H in metal vs silicate.
    """
    X_frac_in_Fe = N_H_in_Fe / (N_H_in_Fe + N_Fe)
    X_frac_in_MgSiO3 = N_H_in_MgSiO3 / (N_H_in_MgSiO3 + N_MgSiO3)
    return X_frac_in_Fe / X_frac_in_MgSiO3

def fn_K_D_H2(N_H2_in_Fe, N_H2_in_MgSiO3, N_Fe, N_MgSiO3):
    """
    K_D for molecular H₂. NOTE: you used (X2_Fe**2) / X2_MgSiO3.
    That implies an equilibrium/stoichiometric rationale (e.g., Heusler-type or
    2H ↔ H₂ relation). If intentional, keep; otherwise a purely molecular fraction
    ratio would be (X2_Fe / X2_MgSiO3). Keeping your original expression.
    """
    X2_frac_in_Fe = N_H2_in_Fe / (N_H2_in_Fe + N_Fe)
    X2_frac_in_MgSiO3 = N_H2_in_MgSiO3 / (N_H2_in_MgSiO3 + N_MgSiO3)
    K_D_H2 = X2_frac_in_Fe**2 / X2_frac_in_MgSiO3  # original choice retained
    return K_D_H2

def fn_X_H_in_MgSiO3(N_H_in_MgSiO3, N_MgSiO3):
    """ Atomic H fraction in silicate. """
    return N_H_in_MgSiO3 / (N_H_in_MgSiO3 + N_MgSiO3)

def fn_X_H2_in_MgSiO3(N_H2_in_MgSiO3, N_MgSiO3):
    """ Molecular H₂ fraction in silicate. """
    return N_H2_in_MgSiO3 / (N_H2_in_MgSiO3 + N_MgSiO3)

def fn_X_H_in_Fe(N_H_in_Fe, N_Fe):
    """ Atomic H fraction in metal. """
    return N_H_in_Fe / (N_H_in_Fe + N_Fe)

def fn_X_H2_in_Fe(N_H2_in_Fe, N_Fe):
    """ Molecular H₂ fraction in metal. """
    return N_H2_in_Fe / (N_H2_in_Fe + N_Fe)


# --------------------------
# H (atomic) CASE: MC stats
# --------------------------

D_wt, D_wt_err, D_wt_lower, D_wt_upper = monte_carlo_error_asymmetric(
    fn_D_wt,
    (mean_N_H_in_Fe, mean_N_H_in_MgSiO3, mean_N_Fe, mean_N_MgSiO3),
    (mean_N_H_in_Fe - sigma_N_H_in_Fe,
     mean_N_H_in_MgSiO3 - sigma_N_H_in_MgSiO3,
     mean_N_Fe - sigma_N_Fe,
     mean_N_MgSiO3 - sigma_N_MgSiO3),
    (mean_N_H_in_Fe + sigma_N_H_in_Fe,
     mean_N_H_in_MgSiO3 + sigma_N_H_in_MgSiO3,
     mean_N_Fe + sigma_N_Fe,
     mean_N_MgSiO3 + sigma_N_MgSiO3),
)

K_D, K_D_err, K_D_lower, K_D_upper = monte_carlo_error_asymmetric(
    fn_K_D,
    (mean_N_H_in_Fe, mean_N_H_in_MgSiO3, mean_N_Fe, mean_N_MgSiO3),
    (mean_N_H_in_Fe - sigma_N_H_in_Fe,
     mean_N_H_in_MgSiO3 - sigma_N_H_in_MgSiO3,
     mean_N_Fe - sigma_N_Fe,
     mean_N_MgSiO3 - sigma_N_MgSiO3),
    (mean_N_H_in_Fe + sigma_N_H_in_Fe,
     mean_N_H_in_MgSiO3 + sigma_N_H_in_MgSiO3,
     mean_N_Fe + sigma_N_Fe,
     mean_N_MgSiO3 + sigma_N_MgSiO3),
)

X_H_in_MgSiO3, X_H_in_MgSiO3_err, X_H_in_MgSiO3_lower, X_H_in_MgSiO3_upper = monte_carlo_error_asymmetric(
    fn_X_H_in_MgSiO3,
    (mean_N_H_in_MgSiO3, mean_N_MgSiO3),
    (mean_N_H_in_MgSiO3 - sigma_N_H_in_MgSiO3, mean_N_MgSiO3 - sigma_N_MgSiO3),
    (mean_N_H_in_MgSiO3 + sigma_N_H_in_MgSiO3, mean_N_MgSiO3 + sigma_N_MgSiO3),
)

X_H_in_Fe, X_H_in_Fe_err, X_H_in_Fe_lower, X_H_in_Fe_upper = monte_carlo_error_asymmetric(
    fn_X_H_in_Fe,
    (mean_N_H_in_Fe, mean_N_Fe),
    (mean_N_H_in_Fe - sigma_N_H_in_Fe, mean_N_Fe - sigma_N_Fe),
    (mean_N_H_in_Fe + sigma_N_H_in_Fe, mean_N_Fe + sigma_N_Fe),
)

print("\n\nFor H:")
print(f"D_wt = {D_wt:.3f}, D_wt_lower = {D_wt_lower:.3f}, D_wt_upper = {D_wt_upper:.3f}")
print(f"K_D = {K_D:.3f}, K_D_lower = {K_D_lower:.3f}, K_D_upper = {K_D_upper:.3f}")
print(f"X_H_in_MgSiO3 = {X_H_in_MgSiO3:.5f}, X_H_in_MgSiO3_lower = {X_H_in_MgSiO3_lower:.5f}, X_H_in_MgSiO3_upper = {X_H_in_MgSiO3_upper:.5f}")
print(f"X_H_in_Fe = {X_H_in_Fe:.5f}, X_H_in_Fe_lower = {X_H_in_Fe_lower:.5f}, X_H_in_Fe_upper = {X_H_in_Fe_upper:.5f}")



# --------------------------
# H₂ (molecular) CASE: MC stats
# --------------------------

D_wt_H2, D_wt_H2_err, D_wt_H2_lower, D_wt_H2_upper = monte_carlo_error_asymmetric(
    fn_D_wt_H2,
    (mean_N_H2_in_Fe, mean_N_H2_in_MgSiO3, mean_N_Fe, mean_N_MgSiO3),
    (mean_N_H2_in_Fe - sigma_N_H2_in_Fe,
     mean_N_H2_in_MgSiO3 - sigma_N_H2_in_MgSiO3,
     mean_N_Fe - sigma_N_Fe,
     mean_N_MgSiO3 - sigma_N_MgSiO3),
    (mean_N_H2_in_Fe + sigma_N_H2_in_Fe,
     mean_N_H2_in_MgSiO3 + sigma_N_H2_in_MgSiO3,
     mean_N_Fe + sigma_N_Fe,
     mean_N_MgSiO3 + sigma_N_MgSiO3),
)

K_D_H2, K_D_H2_err, K_D_H2_lower, K_D_H2_upper = monte_carlo_error_asymmetric(
    fn_K_D_H2,
    (mean_N_H2_in_Fe, mean_N_H2_in_MgSiO3, mean_N_Fe, mean_N_MgSiO3),
    (mean_N_H2_in_Fe - sigma_N_H2_in_Fe,
     mean_N_H2_in_MgSiO3 - sigma_N_H2_in_MgSiO3,
     mean_N_Fe - sigma_N_Fe,
     mean_N_MgSiO3 - sigma_N_MgSiO3),
    (mean_N_H2_in_Fe + sigma_N_H2_in_Fe,
     mean_N_H2_in_MgSiO3 + sigma_N_H2_in_MgSiO3,
     mean_N_Fe + sigma_N_Fe,
     mean_N_MgSiO3 + sigma_N_MgSiO3),
)

X_H2_in_MgSiO3, X_H2_in_MgSiO3_err, X_H2_in_MgSiO3_lower, X_H2_in_MgSiO3_upper = monte_carlo_error_asymmetric(
    fn_X_H2_in_MgSiO3,
    (mean_N_H2_in_MgSiO3, mean_N_MgSiO3),
    (mean_N_H2_in_MgSiO3 - sigma_N_H2_in_MgSiO3, mean_N_MgSiO3 - sigma_N_MgSiO3),
    (mean_N_H2_in_MgSiO3 + sigma_N_H2_in_MgSiO3, mean_N_MgSiO3 + sigma_N_MgSiO3),
)

X_H2_in_Fe, X_H2_in_Fe_err, X_H2_in_Fe_lower, X_H2_in_Fe_upper = monte_carlo_error_asymmetric(
    fn_X_H2_in_Fe,
    (mean_N_H2_in_Fe, mean_N_Fe),
    (mean_N_H2_in_Fe - sigma_N_H2_in_Fe, mean_N_Fe - sigma_N_Fe),
    (mean_N_H2_in_Fe + sigma_N_H2_in_Fe, mean_N_Fe + sigma_N_Fe),
)

print("\n\nFor H2:")
print(f"D_wt_H2 = {D_wt_H2:.3f}, D_wt_H2_lower = {D_wt_H2_lower:.3f}, D_wt_H2_upper = {D_wt_H2_upper:.3f}")
print(f"K_D_H2 = {K_D_H2:.3f}, K_D_H2_lower = {K_D_H2_lower:.3f}, K_D_H2_upper = {K_D_H2_upper:.3f}")
print(f"X_H2_in_MgSiO3 = {X_H2_in_MgSiO3:.5f}, X_H2_in_MgSiO3_lower = {X_H2_in_MgSiO3_lower:.5f}, X_H2_in_MgSiO3_upper = {X_H2_in_MgSiO3_upper:.5f}")
print(f"X_H2_in_Fe = {X_H2_in_Fe:.5f}, X_H2_in_Fe_lower = {X_H2_in_Fe_lower:.5f}, X_H2_in_Fe_upper = {X_H2_in_Fe_upper:.5f}")