
import numpy as np
import re
from ast import literal_eval
import argparse

from mc_error import monte_carlo_error_asymmetric # located at $HELP_SCRIPTS/general/mc_error.pys

"""
Usage:  python $HELP_SCRIPTS_TI/fit_KD_PTX.py -s H > log.fit_KD_PTX 2>&1
        python $HELP_SCRIPTS_TI/fit_KD_PTX.py -s He > log.fit_KD_PTX 2>&1
"""

ANALYSIS_MODE=0 # >0: analysis, <=0: no analysis
PLOT_MODE=1 # >0: plot, <=0: no plot
H_STOICH_MODE = 1 # 2: calculate for H2, 1: calculate for H
X_in_Fe_MODE = 0 # 1: plot w array_X_in_Fe, 0: plot with the default (array_X or array_X_in_MgSiO3)


# read secondary species from terminal, e.g., He, H, C, etc.
parser = argparse.ArgumentParser(description="Fit KD/KD_prime for a species in Fe and MgSiO3 systems from TI data.")
parser.add_argument(
    "-s", "--secondary_species",
    type=str,
    default="He",
    help="Secondary species to estimate KD for (default: He)."
)
args = parser.parse_args()
# Use the parsed secondary species
secondary_species = args.secondary_species


print(f"Fitting KD/KD_prime for {secondary_species} in Fe and MgSiO3 systems from TI data.")








_num_token_re = re.compile(
    r"""
    [+-]? (?:                                   # sign
        (?:\d+\.\d*|\.\d+|\d+)                  # number parts
        (?:[eE][+-]?\d+)?                       # optional exponent
        |                                       # OR
        inf(?:inity)?                           # inf / infinity
        | nan
    )
    """,
    re.VERBOSE | re.IGNORECASE
)

def parse_numpy_repr_list(s):
    """
    Parse strings like:
        "[1.0 2.5e-3  nan 4.]" (possibly multi-line)
    into a float64 NumPy array.

    Returns original object if it's already an array or not a bracketed string.
    """
    if isinstance(s, np.ndarray):
        return s
    if not isinstance(s, str):
        return s
    txt = s.strip()
    if len(txt) < 2 or txt[0] != '[' or txt[-1] != ']':
        return s  # not our pattern

    inner = txt[1:-1].strip()
    if not inner:
        return np.array([], dtype=float)

    # Replace any commas with spaces (just in case), collapse whitespace
    inner = inner.replace(',', ' ')
    # Find all numeric tokens
    tokens = _num_token_re.findall(inner)
    if not tokens:
        # fallback: try splitting on whitespace
        parts = inner.split()
    else:
        parts = tokens

    out = []
    for t in parts:
        tl = t.lower()
        if tl in ('nan',):
            out.append(np.nan)
        elif tl in ('inf', '+inf', 'infinity', '+infinity'):
            out.append(np.inf)
        elif tl in ('-inf', '-infinity'):
            out.append(-np.inf)
        else:
            try:
                out.append(float(t))
            except ValueError:
                # unexpected token → treat as NaN
                out.append(np.nan)
    return np.array(out, dtype=float)



from matplotlib.colors import LinearSegmentedColormap
def pastel_cmap(cmap, factor=0.7, N=256):
    """
    Return a pastel version of `cmap` by blending each color toward white.
    
    Parameters
    ----------
    cmap : Colormap
        The original colormap (e.g. plt.get_cmap('hot')).
    factor : float, optional
        How “pastel” it is: 0 → original, 1 → pure white.  Default 0.7.
    N : int, optional
        Number of entries in the colormap.  Default 256.
    """
    # sample the original colormap
    colors = cmap(np.linspace(0, 1, N))
    # blend RGB channels toward 1.0 (white)
    colors[:, :3] = colors[:, :3] + (1.0 - colors[:, :3]) * factor
    # rebuild a new colormap
    return LinearSegmentedColormap.from_list(f'pastel_{cmap.name}', colors)










import numpy as np
import pandas as pd

df_superset = pd.read_csv("all_TI_results_superset.csv")
# narrow down to Config_folder with *_8H*
df_superset = df_superset[df_superset["Config_folder"].str.contains("_8H")]

# print df_superset[f'array_X_{secondary_species}'][0]
# print(df_superset[f'array_X_{secondary_species}'][0])

# convert all columns beginning with "array_" to pd.Series. These have been read as str: [1.0, 2.0, 3.0]
array_cols = [c for c in df_superset.columns if c.startswith("array_")]
for col in array_cols:
    df_superset[col] = df_superset[col].apply(parse_numpy_repr_list)




# X_H = n_H / (n_H + n_MgSiO3) and X_H2 = n_H2 / (n_H2 + n_MgSiO3) = 0.5 n_H / (0.5 * n_H + n_MgSiO3)

# X_H = n_H / (n_H + n_MgSiO3) >> X_H * (n_H + n_MgSiO3) = n_H >> n_H * (1 - X_H) = n_MgSiO3 * X_H
# n_H = (X_H / (1 - X_H)) * n_MgSiO3
# X_H2 = n_H2 / (n_H2 + n_MgSiO3) = 0.5 n_H / (0.5 * n_H + n_MgSiO3)
# X_H2 = 0.5 * (X_H / (1 - X_H)) * n_MgSiO3 / ( ((0.5*X_H / (1 - X_H)) + 1) * n_MgSiO3)
# X_H2 = 0.5 * (X_H / (1 - X_H)) / (0.5 * (X_H / (1 - X_H)) + 1)
# X_H2 = (0.5*X_H / (1 - X_H)) / (((1 - 0.5*X_H) / (1 - X_H)))
# X_H2 = 0.5*X_H / (1 - 0.5*X_H) = X_H / (2 - X_H)



# create X_array, P_array, T_array
array_X = np.array([df_superset[f'array_X_{secondary_species}'][i] for i in range(len(df_superset))])
array_X2 = array_X / (2 - array_X)  # X_H2 = X_H / (2 - X_H)
array_logX = np.log(array_X)
array_logX2 = np.log(array_X2)
array_P = np.array([df_superset[f'array_P_{secondary_species}'][i] for i in range(len(df_superset))])
array_T = np.array([df_superset[f'array_T_{secondary_species}'][i] for i in range(len(df_superset))])

array_KD = np.array([df_superset[f'array_KD_{secondary_species}'][i] for i in range(len(df_superset))])
array_logKD = np.log(array_KD)
array_KD_prime = np.array([df_superset[f'array_KD_prime_{secondary_species}'][i] for i in range(len(df_superset))])
array_logKD_prime = np.log(array_KD_prime)
array_D_wt = np.array([df_superset[f'array_D_wt_{secondary_species}'][i] for i in range(len(df_superset))])

array_KD_lower = np.array([df_superset[f'array_KD_{secondary_species}_lower'][i] for i in range(len(df_superset))])
array_logKD_lower = np.log(array_KD_lower)
array_KD_upper = np.array([df_superset[f'array_KD_{secondary_species}_upper'][i] for i in range(len(df_superset))])
array_logKD_upper = np.log(array_KD_upper)
array_KD_prime_lower = np.array([df_superset[f'array_KD_prime_{secondary_species}_lower'][i] for i in range(len(df_superset))])
array_logKD_prime_lower = np.log(array_KD_prime_lower)
array_KD_prime_upper = np.array([df_superset[f'array_KD_prime_{secondary_species}_upper'][i] for i in range(len(df_superset))])
array_logKD_prime_upper = np.log(array_KD_prime_upper)
array_D_wt_lower = np.array([df_superset[f'array_D_wt_{secondary_species}_lower'][i] for i in range(len(df_superset))])
array_D_wt_upper = np.array([df_superset[f'array_D_wt_{secondary_species}_upper'][i] for i in range(len(df_superset))])

# array_KD_error = np.array([df_superset[f'array_KD_{secondary_species}_error'][i] for i in range(len(df_superset))])
# array_logKD_error = np.log(array_KD_error)
# array_KD_prime_error = np.array([df_superset[f'array_KD_prime_{secondary_species}_error'][i] for i in range(len(df_superset))])
# array_logKD_prime_error = np.log(array_KD_prime_error)
array_logKD_error = 0.5 * (array_logKD_upper - array_logKD_lower)
array_logKD_prime_error = 0.5 * (array_logKD_prime_upper - array_logKD_prime_lower)


flat_array_X = array_X.flatten()
flat_array_X2 = array_X2.flatten()
flat_array_logX = array_logX.flatten()
flat_array_logX2 = array_logX2.flatten()
flat_array_P = array_P.flatten()
flat_array_T = array_T.flatten()
# flat_array_KD_prime = array_KD_prime.flatten()
# flat_array_KD_prime_error = array_KD_prime_error.flatten()
flat_array_logKD = array_logKD.flatten()
flat_array_logKD_error = array_logKD_error.flatten()
flat_array_logKD_prime = array_logKD_prime.flatten()
flat_array_logKD_prime_error = array_logKD_prime_error.flatten()






test_var = flat_array_logKD_prime / flat_array_logKD_prime_error
print(f"min, max, mean, std of array_KD_prime: {np.min(array_KD_prime)}, {np.max(array_KD_prime)}, {np.mean(array_KD_prime)}, {np.std(array_KD_prime)}")
print(f"min, max, mean, std of flat_array_logKD_prime: {np.min(flat_array_logKD_prime)}, {np.max(flat_array_logKD_prime)}, {np.mean(flat_array_logKD_prime)}, {np.std(flat_array_logKD_prime)}")
print(f"min, max, mean, std of flat_array_logKD_prime_error: {np.min(flat_array_logKD_prime_error)}, {np.max(flat_array_logKD_prime_error)}, {np.mean(flat_array_logKD_prime_error)}, {np.std(flat_array_logKD_prime_error)}")
print(f"min, max, mean, std of test_var: {np.min(test_var)}, {np.max(test_var)}, {np.mean(test_var)}, {np.std(test_var)}")

# similarly for T and P
print(f"min, max, mean, std of array_T: {np.min(array_T)}, {np.max(array_T)}, {np.mean(array_T)}, {np.std(array_T)}")
print(f"min, max, mean, std of array_P: {np.min(array_P)}, {np.max(array_P)}, {np.mean(array_P)}, {np.std(array_P)}")

# exit(0)









if ANALYSIS_MODE > 0:
    # PySR
    from pysr import PySRRegressor
    if secondary_species == "He":
        print("Fitting KD for He in MgSiO3")
        y = flat_array_logKD
        error_y = flat_array_logKD_error
        x = np.column_stack((flat_array_logX, flat_array_P, flat_array_T))

    elif secondary_species == "H":
        print("Fitting KD_prime for H_2 in MgSiO3")
        
        if H_STOICH_MODE == 2:
            y = flat_array_logKD_prime
            # error_y = 1e-10 * flat_array_logKD_prime_error
            error_y = flat_array_logKD_prime_error
            x = np.column_stack((flat_array_logX2, flat_array_P, flat_array_T))

        if H_STOICH_MODE == 1:
            # For H, we need to adjust the x values
            y = flat_array_logKD
            error_y = flat_array_logKD_error
            x = np.column_stack((flat_array_logX, flat_array_P, flat_array_T))
    
    weights = 1 / (error_y ** 2 + 1e-10)  # Avoid division by zero

    


    # shape of the data
    print(f"x shape: {x.shape}, y shape: {y.shape}, weights shape: {weights.shape}")

    # exit(0)

    model = PySRRegressor(
        populations=8,
        # ^ Assuming we have 4 cores, this means 2 populations per core, so one is always running.
        population_size=50,
        # ^ Slightly larger populations, for greater diversity.
        ncycles_per_iteration=500,
        # ^ Generations between migrations.
        niterations=10000000,  # Run forever
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
            # Stop early if we find a good and simple equation
        ),
        timeout_in_seconds=60 * 1,
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=[
            "log",
            "exp"
        ],
        # loss="loss(x, y, w) = w*((x - y)^2)",
        elementwise_loss="LPDistLoss{2}()",
    )

    model.fit(x, y, weights=weights)


    # increase the width of the pysr output
    import sys


    print(model)


    # print the best model
    print("Best model:")
    print(model.get_best())


    print(model.latex()[0])





















# plot the best model
if PLOT_MODE > 0:
    import matplotlib.pyplot as plt

    ########################################################
    ########################################################
    # x0= np.log(array_X)
    # x1 = array_P
    # x2 = array_T
    def best_fn(x0, x1, x2):# x0: log(X or X2 in MgSiO3), x1: P, x2: T

        if secondary_species == "He":
            fn = np.exp(x0) - (x2 * (0.07335293 / x1))
            fn = ((x2 * -0.06782242) + -44.47209) / x1 # loss: 0.12429151 -- one of the simplest w/o X dependence
            fn = ((x2 * -0.0678) + -44.5) / x1 # loss: 0.12429151 -- one of the simplest w/o X dependence -- simplified

        elif secondary_species == "H":
            if H_STOICH_MODE == 2:
                fn = (((x1 * -0.0048796246) + 6.5670815) / np.exp(np.exp(x0))) + x0 # loss: 1.2789929; one of the simplest w no T dependence
                fn = (((x1 * -0.00488) + 6.567) / np.exp(np.exp(x0))) + x0 # loss: 1.2789929; one of the simplest w no T dependence -- simplified
                # fn =  #loss: ??? ; one of the simplest w T dependence
                fn = ((6.7519736 - (0.0051717 * x1)) + x0) / ((x1 ** (1.6505091 + (50.947327 / x0))) + ((x2 ** 0.1455538) ** (np.log(x1) ** ((x0 + (0.6689549 ** x0)) * x0)))) #loss: 1.1131124; the most complex and lowest loss soln
            elif H_STOICH_MODE == 1:
                fn = ((x1 * -0.0033529147) + 3.52548) ** (0.44919842 ** (np.exp(x0) ** np.exp(((76120.4 / x2) + (-62.58456 / (-1.609248 - x0))) / x1))) # loss: 0.13513389; one of the simplest w T dependence
                # fn = ((x1 * -0.0033594905) + 3.5382779) ** ((0.3938717 ** (np.exp(x0) ** np.exp(((-43.5826 / (-1.6072097 - x0)) + (np.exp((12387.732 / x2) + x0) + 4.5243583)) / x1))) - -0.0033594905) # loss: 0.12907255
                # fn = (x1 * -0.0032485272) + (x2 ** 0.13168865)
        return np.exp(fn) # {return KD for He} or {KD_prime for H}
    


    def best_fn_v2(x0, x1, x2, H_STOICH_MODE=2):# x0: log(X or X2 in MgSiO3), x1: P, x2: T

        if secondary_species == "He":
            fn = np.exp(x0) - (x2 * (0.07335293 / x1))
            fn = ((x2 * -0.06782242) + -44.47209) / x1 # loss: 0.12429151 -- one of the simplest w/o X dependence
            fn = ((x2 * -0.0678) + -44.5) / x1 # loss: 0.12429151 -- one of the simplest w/o X dependence -- simplified

        elif secondary_species == "H":
            if H_STOICH_MODE == 2:
                fn = (((x1 * -0.0048796246) + 6.5670815) / np.exp(np.exp(x0))) + x0 # loss: 1.2789929; one of the simplest w no T dependence
                fn = (((x1 * -0.00488) + 6.567) / np.exp(np.exp(x0))) + x0 # loss: 1.2789929; one of the simplest w no T dependence -- simplified
                # fn =  #loss: ??? ; one of the simplest w T dependence
                fn = ((6.7519736 - (0.0051717 * x1)) + x0) / ((x1 ** (1.6505091 + (50.947327 / x0))) + ((x2 ** 0.1455538) ** (np.log(x1) ** ((x0 + (0.6689549 ** x0)) * x0)))) #loss: 1.1131124; the most complex and lowest loss soln
            elif H_STOICH_MODE == 1:
                fn = ((-0.0034771473 * (x1 + (8.470072 / np.log(x2 / 3054.1853)))) + 3.665851) ** (0.47189304 ** (np.exp(x0) ** np.exp(-1.0673585 / (-1.5176716 - x0)))) # loss: 0.12597673; one of the simplest w T dependence
                # fn = ((x1 * -0.0033594905) + 3.5382779) ** ((0.3938717 ** (np.exp(x0) ** np.exp(((-43.5826 / (-1.6072097 - x0)) + (np.exp((12387.732 / x2) + x0) + 4.5243583)) / x1))) - -0.0033594905) # loss: 0.12907255
                # fn = (x1 * -0.0032485272) + (x2 ** 0.13168865)

        return np.exp(fn) # {return KD for He} or {KD_prime for H}



    ########################################################
    ########################################################



    # plot array_KD vs array_X only for phase = MgSiO3 -- df_superset

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times"]
    # plt.rcParams.update({
    #     "font.weight": "medium",          # bold text everywhere
    #     "axes.labelweight": "medium",     # x/y labels
    #     "axes.titleweight": "medium",     # titles
    #     # "text.latex.preamble": r"\usepackage{sfmath}",  # bold math
    # })


    # all font sizes
    font_size_title = 14
    font_size_labels = 16
    font_size_ticks = 14
    font_size_legend = 12
    
    # 4 separate figures for KD vs X, KD_prime vs X, D_wt vs X, and D_wt vs Xw
    # create a 4 x 5 grid of subplots where each row corresponds to a different target "Target pressure (GPa)"
    # and each column corresponds to a different "Target temperature (K)" in ascending order
    fig_2, axes_2 = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes_2 = axes_2.flatten()  # flatten the 2D array of axes to 1D for easier iteration
    
    marker_opts = dict(marker='o', linestyle='', markersize=10, alpha=1)#,color=base_color)
    marker_opts_scatter = dict(linestyle='', s=5, alpha=0.5)#,edgecolor='black',
    marker_opts_error = dict(linestyle='', markersize=10, alpha=0.25, capsize=3, elinewidth=1)#, color='black',ecolor='black')
    
    magma = plt.get_cmap("magma")
    pastel_magma = pastel_cmap(magma, factor=0.25)  # tweak factor between 0 and 1
    cmap = pastel_magma  # use pastel magma for the plots

    # sort df_superset by "Target pressure (GPa)" and "Target temperature (K)" -- largest to smallest pressure but smallest to largest temperature
    df_superset = df_superset.sort_values(
        by=["Target pressure (GPa)", "Target temperature (K)"],
        ascending=[False, True]  # largest to smallest pressure, smallest to largest temperature
    )


    i_axes = -1  # counter for the axes

    # iterate over all systems in df_superset
    for i, row in df_superset.iterrows():

        phase = row["Phase"]
        pt = row["P_T_folder"]

        # only continue if (Target pressure (GPa) = 50 and Target temperature (K)=3500) or (Target pressure (GPa)= 1000 and Target temperature (K)=13000)
        if secondary_species == "He":
            if not ((row["Target pressure (GPa)"] == 50 and row["Target temperature (K)"] == 3850) or
                    (row["Target pressure (GPa)"] == 250 and row["Target temperature (K)"] == 6500) or
                    (row["Target pressure (GPa)"] == 1000 and row["Target temperature (K)"] == 14300)):
                continue # skip to next iteration
        elif secondary_species == "H":
            if not ((row["Target pressure (GPa)"] == 50 and row["Target temperature (K)"] == 3850) or
                    (row["Target pressure (GPa)"] == 250 and row["Target temperature (K)"] == 6500) or
                    (row["Target pressure (GPa)"] == 1000 and row["Target temperature (K)"] == 14300)):
                continue # skip to next iteration

        assert not isinstance(phase, pd.Series)  # should be a scalar

        if phase != f"MgSiO3_{secondary_species}":
            continue

        i_axes += 1
        # if i_axes == 4, 9 or 14, +1
        if i_axes in [4, 9, 14]:
            i_axes += 1 # to maintain 1 P per row

        # 2) Determine the other phase
        other_phase = f"MgSiO3_{secondary_species}" if phase == f"Fe_{secondary_species}" else f"Fe_{secondary_species}"

        mask = (
            (df_superset["Phase"] == other_phase) &
            (df_superset["P_T_folder"] == pt) &
            (df_superset["Target temperature (K)"] == row["Target temperature (K)"])
            )
        mask_idx = df_superset.index[mask][0]
        

        array_KD = row[f"array_KD_{secondary_species}"]
        array_KD_lower = row[f"array_KD_{secondary_species}_lower"]
        array_KD_upper = row[f"array_KD_{secondary_species}_upper"]

        array_KD_prime = row[f"array_KD_prime_{secondary_species}"]
        array_KD_prime_error = row[f"array_KD_prime_{secondary_species}_error"]
        array_KD_prime_lower = row[f"array_KD_prime_{secondary_species}_lower"]
        array_KD_prime_upper = row[f"array_KD_prime_{secondary_species}_upper"]

        array_X = row[f"array_X_{secondary_species}"]
        array_X_lower = row[f"array_X_{secondary_species}_lower"]
        array_X_upper = row[f"array_X_{secondary_species}_upper"]

        array_X_in_Fe = row[f"array_X_{secondary_species}_in_Fe"]
        array_X_in_Fe_lower = row[f"array_X_{secondary_species}_in_Fe_lower"]
        array_X_in_Fe_upper = row[f"array_X_{secondary_species}_in_Fe_upper"]

        if X_in_Fe_MODE == 1:
            array_X_chosen = array_X_in_Fe
            array_X_lower_chosen = array_X_in_Fe_lower
            array_X_upper_chosen = array_X_in_Fe_upper
        else:
            array_X_chosen = array_X
            array_X_lower_chosen = array_X_lower
            array_X_upper_chosen = array_X_upper

        array_T = row[f"array_T_{secondary_species}"]
        array_P = row[f"array_P_{secondary_species}"]

        if secondary_species == "He":
            KD_chosen = array_KD
            KD_chosen_lower = array_KD_lower
            KD_chosen_upper = array_KD_upper
            
            array_x_axis = array_X_chosen
            secondary_species_label = "He"

        elif secondary_species == "H":
            if H_STOICH_MODE == 2:
                KD_chosen = array_KD_prime
                KD_chosen_lower = array_KD_prime_lower
                KD_chosen_upper = array_KD_prime_upper
                def fn_X2(X):
                    return X / (2 - X)  # X_H2 = X_H / (2 - X_H)
                array_X2 = array_X_chosen / (2 - array_X_chosen)  # X_H2 = X_H / (2 - X_H)
                # array_X2_lower = row[f"array_X_{secondary_species}_lower"]
                # array_X2_upper = row[f"array_X_{secondary_species}_upper"]
                # array_X2, array_X2_err, array_X2_lower, array_X2_upper = monte_carlo_error_asymmetric(fn_X2,
                #                                                                                     array_X_chosen,
                #                                                                                     array_X_lower_chosen,
                #                                                                                     array_X_upper_chosen)

                array_x_axis = array_X2
                secondary_species_label = "H2"
            elif H_STOICH_MODE == 1:
                KD_chosen = array_KD
                KD_chosen_lower = array_KD_lower
                KD_chosen_upper = array_KD_upper
                array_x_axis = array_X_chosen
                secondary_species_label = "H"
        # KD_chosen = array_KD
        # KD_chosen_lower = array_KD_lower
        # KD_chosen_upper = array_KD_upper

        ########################
        axes_2[i_axes].scatter(
            array_x_axis, KD_chosen,
            label = (
                f"P={row['Target pressure (GPa)']:.0f} GPa, "
                f"T={row['Target temperature (K)']:.0f} K"
            ),
            **marker_opts_scatter
        )

        axes_2[i_axes].fill_between(
            array_x_axis,
            KD_chosen_lower,
            KD_chosen_upper,
            # KD_chosen-KD_chosen_error,
            # KD_chosen+KD_chosen_error,
            alpha=0.2, label='Error range',
            color=axes_2[i_axes].collections[0].get_edgecolor()  # same as the scatter points
        )

        ########################
        ########################
        # plot the best fit line
        x0 = np.log(array_x_axis)
        x1 = array_P
        x2 = array_T
        # y_fit = best_fn(x0, x1, x2)
        if secondary_species == "H":
            y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=H_STOICH_MODE)
        elif secondary_species == "He":
            y_fit = best_fn_v2(x0, x1, x2, H_STOICH_MODE=1)
        print(f"Best fit line for KD vs P, T: {y_fit}")
        print(f"logX: {x0}")
        print(f"T: {array_T}")
        print(f"P: {array_P}")
        axes_2[i_axes].plot(
            array_x_axis, y_fit,
            linestyle='--',
            label=f"Best fit model",
            color='black', linewidth=1.5
        )
        ########################
        ########################



        ###############################################
        ###############################################
        if secondary_species == "He" and i_axes == 1:
            # from 2 Phase simulations
            K_D, K_D_lower, K_D_upper = (0.174, 0.145, 0.202)
            X_He_in_MgSiO3, X_He_in_MgSiO3_lower, X_He_in_MgSiO3_upper = (0.48915, 0.46913, 0.50840)
            X_He_in_Fe, X_He_in_Fe_lower, X_He_in_Fe_upper = (0.08455, 0.07091, 0.09803)

            if X_in_Fe_MODE == 1:
                X_He_in_MgSiO3 = X_He_in_Fe
                X_He_in_MgSiO3_lower = X_He_in_Fe_lower
                X_He_in_MgSiO3_upper = X_He_in_Fe_upper

            axes_2[i_axes].scatter(
                X_He_in_MgSiO3, K_D,
                label = (
                    f"2P-AIMD")
                ,
                marker='s', s=200, color='red', edgecolor='black', zorder=5, alpha=0.5
            )
            # axes_2[i_axes].errorbar(
            #     X_He_in_MgSiO3, K_D,
            #     xerr=[[X_He_in_MgSiO3 - X_He_in_MgSiO3_lower], [X_He_in_MgSiO3_upper - X_He_in_MgSiO3]],
            #     yerr=[[K_D - K_D_lower], [K_D_upper - K_D]],
            #     fmt='none',  # no extra marker
            #     ecolor='red',
            #     linestyle='none',
            #     capsize=5, elinewidth=1.5, alpha=0.25,
            #     zorder=4
            # )


        elif secondary_species == "H" and H_STOICH_MODE == 1 and i_axes == 1:
            K_D, K_D_lower, K_D_upper = (1.434, 1.239, 1.674)
            X_H_in_MgSiO3, X_H_in_MgSiO3_lower, X_H_in_MgSiO3_upper = (0.42190, 0.37334, 0.46500)
            X_H_in_Fe, X_H_in_Fe_lower, X_H_in_Fe_upper = (0.60387, 0.54287, 0.66338)

            if X_in_Fe_MODE == 1:
                X_H_in_MgSiO3 = X_H_in_Fe
                X_H_in_MgSiO3_lower = X_H_in_Fe_lower
                X_H_in_MgSiO3_upper = X_H_in_Fe_upper

            axes_2[i_axes].scatter(
                X_H_in_MgSiO3, K_D,
                label = (
                    f"2P-AIMD")
                ,
                marker='s', s=200, color='red', edgecolor='black', zorder=5, alpha=0.5
            )
            # axes_2[i_axes].errorbar(
            #     X_H_in_MgSiO3, K_D,
            #     xerr=[[X_H_in_MgSiO3 - X_H_in_MgSiO3_lower], [X_H_in_MgSiO3_upper - X_H_in_MgSiO3]],
            #     yerr=[[K_D - K_D_lower], [K_D_upper - K_D]],
            #     fmt='none',  # no extra marker
            #     ecolor='red',
            #     linestyle='none',
            #     capsize=5, elinewidth=1.5, alpha=0.25, 
            #     zorder=4
            # )
        elif secondary_species == "H" and H_STOICH_MODE == 2 and i_axes == 1:
            K_D, K_D_lower, K_D_upper = (0.232, 0.173, 0.306)
            X_H2_in_MgSiO3, X_H2_in_MgSiO3_lower, X_H2_in_MgSiO3_upper = (0.13814, 0.11454, 0.16058)
            X_H2_in_Fe, X_H2_in_Fe_lower, X_H2_in_Fe_upper = (0.17803, 0.15727, 0.19806)

            if X_in_Fe_MODE == 1:
                X_H2_in_MgSiO3 = X_H2_in_Fe
                X_H2_in_MgSiO3_lower = X_H2_in_Fe_lower
                X_H2_in_MgSiO3_upper = X_H2_in_Fe_upper

            axes_2[i_axes].scatter(
                X_H2_in_MgSiO3, K_D,
                label = (
                    f"2P-AIMD")
                ,
                marker='s', s=200, color='red', edgecolor='black', zorder=5, alpha=0.5
            )
        ###############################################
        ###############################################




    # set x and y limits for all axes
    for axes in [axes_2]:
    # for axes in [axes_2]:
        for i_axes, ax in enumerate(axes):
            # if axes is not axes_2:
            #     if axes is not axes_1:
            #         ax.set_xscale("log")
            ax.set_xscale("log")
            # if axes is not axes_2:
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.5)

            # if i_axes not in (0, 5, 10, 15):
            #     ax.tick_params(labelleft=False)
            # else:
            ax.tick_params(labelleft=True)
            # if axes is axes_1:
            #     ax.set_ylabel(f"K$_D$")
            if axes is axes_2:
                if secondary_species == "He":
                    ax.set_ylabel(r"$K_{D,\;\mathrm{He}}^{\mathrm{Fe/MgSiO_3}}$", fontsize=font_size_labels)
                    ax.set_xlabel(rf"$x_{{\mathrm{{He}}}}^{{\mathrm{{MgSiO_3}}}}$", fontsize=font_size_labels)
                    if X_in_Fe_MODE == 1:
                        ax.set_xlabel(rf"$x_{{\mathrm{{He}}}}^{{\mathrm{{Fe}}}}$", fontsize=font_size_labels)
                elif secondary_species == "H":
                    if H_STOICH_MODE == 2:
                        ax.set_ylabel(r"$K_{D,\;\mathrm{H_2}}^{\mathrm{Fe/MgSiO_3}}$", fontsize=font_size_labels)
                        ax.set_xlabel(rf"$x_{{\mathrm{{H_2}}}}^{{\mathrm{{MgSiO_3}}}}$", fontsize=font_size_labels)
                        if X_in_Fe_MODE == 1:
                            ax.set_xlabel(rf"$x_{{\mathrm{{H_2}}}}^{{\mathrm{{Fe}}}}$", fontsize=font_size_labels)
                    elif H_STOICH_MODE == 1:
                        ax.set_ylabel(r"$K_{D,\;\mathrm{H}}^{\mathrm{Fe/MgSiO_3}}$", fontsize=font_size_labels)
                        ax.set_xlabel(rf"$x_{{\mathrm{{H}}}}^{{\mathrm{{MgSiO_3}}}}$", fontsize=font_size_labels)
                        if X_in_Fe_MODE == 1:
                            ax.set_xlabel(rf"$x_{{\mathrm{{H}}}}^{{\mathrm{{Fe}}}}$", fontsize=font_size_labels)
            # elif axes is axes_3:
            #     ax.set_ylabel(f"D$_{{wt}}$")
            # elif axes is axes_4:
            #     ax.set_ylabel(f"D$_{{wt}}$")
            # elif axes is axes_5:
            #     ax.set_ylabel(f"X$_{{w,{secondary_species_label}}}$")

            # hide y label + ticklabels for right column sub-plots
            if i_axes > 0:
                ax.set_ylabel("")  # remove y label
                ax.tick_params(labelleft=False)  # remove y tick labels

            if i_axes != 1:
                ax.set_xlabel("")  # remove y label


            # if i_axes >= 15:
            ax.tick_params(labelbottom=True)         # force it ON explicitly
            # If shared x hides it, unhide individual labels:
            for lbl in ax.get_xticklabels():
                lbl.set_visible(True)
            # if axes is not axes_4 and axes is not axes_5:
            
            # elif axes is axes_4:
            #     ax.set_xlabel(fr"$X_{{w,{secondary_species_label}}}^{{MgSiO_3}}$")
            # elif axes is axes_5:
            #     ax.set_xlabel(fr"$X_{{{secondary_species_label}}}^{{MgSiO_3}}$")
            # else:
            #     ax.tick_params(labelbottom=False)

            # delete sub-plots that are not used -- i_axes = 4,9,14
            # if i_axes in (4, 9, 14):
            #     ax.remove()

            # legend
            # if i_axes < 15:
            #     ax.legend(loc="lower right", fontsize=8)
            # else:
            #     ax.legend(loc="lower right", fontsize=8)
            ax.legend(loc="best", fontsize=font_size_legend)

            # font size of tick labels
            ax.tick_params(axis='x', labelsize=font_size_ticks)
            ax.tick_params(axis='y', labelsize=font_size_ticks)

            # x label font size
            ax.xaxis.label.set_size(font_size_labels)


            # x axis limits
            # if axes is not axes_3 and axes is not axes_4:
            #     ax.set_xlim(1e-4, 100/109)
            # else:
            #     ax.set_xlim(1e-4, 0.1)  # set x
            # ax.set_xlower(1e-4)
            ax.set_xlim(1e-6, 1)
            # ax.set_xlim(1e-2, 0.5)

            if secondary_species == "H":
                ax.set_ylim(1e-1, 1e2)

            if secondary_species == "He":
                ax.set_ylim(1e-3, 1e0)

            if secondary_species == "H" and H_STOICH_MODE == 2:
                # ax.set_ylim(1e-6, None)
                ax.set_ylim(2e-6, 0.9e2)



            # y minor grid lines on
            ax.minorticks_on()

            ax.grid(True, which="both", ls="--", alpha=0.5)

            

            # minor grid lines
            # ax.minorticks_on()
            # ax.grid(which='minor', linestyle=':', alpha=0.5)


    # set the title for each figure
    # fig_1.suptitle(
    #     f"K$_D$ vs X for {secondary_species}",
    #     fontsize=10
    # )
    # if secondary_species == "He":
    #     fig_2.suptitle(
    #         r"K$_D^{{He_{{sil}}\\rightleftharpoons He_{{Fe}}}} vs X_{{{secondary_species_label}}}$",
    #         fontsize=font_size_title
    #     )
    # elif secondary_species == "H":
    #     if H_STOICH_MODE == 2:
    #         fig_2.suptitle(
    #             f"K$_D^{{H_{{2,sil}}\\rightleftharpoons H_{{Fe}}}}$ vs X for {secondary_species_label}",
    #             fontsize=font_size_title
    #         )
    #     elif H_STOICH_MODE == 1:
    #         fig_2.suptitle(
    #             f"K$_D^{{H_{{sil}}\\rightleftharpoons H_{{Fe}}}}$ vs X for {secondary_species_label}",
    #             fontsize=font_size_title
    #         )
    # fig_2.suptitle(
    #     f"K$_D^{{H_{{2,sil}}\\rightleftharpoons H_{{Fe}}}}$ vs X for {secondary_species}",
    #     fontsize=10
    # )
    # fig_3.suptitle(
    #     f"D$_{{wt}}$ vs X for {secondary_species}",
    #     fontsize=10
    # )
    # fig_4.suptitle(
    #     f"D$_{{wt}}$ vs Xw for {secondary_species}",
    #     fontsize=10
    # )
    # fig_5.suptitle(
    #     f"X vs Xw for {secondary_species}",
    #     fontsize=10
    # )

    plt.tight_layout()  # leave space for suptitle

    # save the figures
    # fig_1.savefig(f"array__KD_vs_X.png", dpi=300)
    if secondary_species == "He":
        fig_2.savefig(f"paper__fit__array__KD_chosen_vs_He.png", dpi=300)
    elif secondary_species == "H":
        if H_STOICH_MODE == 2:
            fig_2.savefig(f"paper__fit__array__KD_chosen_vs_H2.png", dpi=300)
        elif H_STOICH_MODE == 1:
            fig_2.savefig(f"paper__fit__array__KD_chosen_vs_H.png", dpi=300)
    # fig_3.savefig(f"array__D_wt_vs_X.png", dpi=300)
    # fig_4.savefig(f"array__D_wt_vs_Xw.png", dpi=300)
    # fig_5.savefig(f"array__X_vs_Xw.png", dpi=300)

print("Done!")