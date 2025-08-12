
import numpy as np
import re
from ast import literal_eval
import argparse

"""
Usage:  python $HELP_SCRIPTS_TI/fit_KD_PTX.py -s H > log.fit_KD_PTX 2>&1
        python $HELP_SCRIPTS_TI/fit_KD_PTX.py -s He > log.fit_KD_PTX 2>&1
"""

ANALYSIS_MODE=1 # >0: analysis, <=0: no analysis
PLOT_MODE=0 # >0: plot, <=0: no plot


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
        y = flat_array_logKD_prime
        error_y = 1e-10 * flat_array_logKD_prime_error
        x = np.column_stack((flat_array_logX2, flat_array_P, flat_array_T))
    
    weights = 1 / (error_y ** 2 + 1e-10)  # Avoid division by zero

    


    # shape of the data
    print(f"x shape: {x.shape}, y shape: {y.shape}, weights shape: {weights.shape}")

    # exit(0)

    model = PySRRegressor(
        populations=56,
        # ^ Assuming we have 4 cores, this means 2 populations per core, so one is always running.
        population_size=224,
        # ^ Slightly larger populations, for greater diversity.
        ncycles_per_iteration=1000,
        # ^ Generations between migrations.
        niterations=10000000,  # Run forever
        # early_stop_condition=(
        #     "stop_if(loss, complexity) = loss < 1e-20 && complexity < 10"
        #     # Stop early if we find a good and simple equation
        # ),
        timeout_in_seconds=60 * 60 * 23,
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


    # x0= np.log(array_X)
    # x1 = array_P
    # x2 = array_T
    def best_fn(x0, x1, x2):
        if secondary_species == "He":
            # fn = (2.0459936 / (x1 + (x2 / -26.194601))) + (((2.6340475 - ((x2 / x1) * 0.17239113)) / (np.exp(x0) + 0.9716166)) + x0)
            # fn = (((3.4979217 / (((x2 / x1) / -0.10494591) + x1)) + ((x0 + 2.4573598) - ((x2 * 0.16685495) / (x1 - (-1.1481286 / x0))))) / np.exp(np.exp(x0 * 1.9996892)))
            # fn = (x0 + (-343.9297 / x1))
            # fn = ((-435.83066 / x1) + 0.60782576) + x0
            # fn = (((x2 * -0.16254203) / x1) + (-57.93452 / (x1 + -329.52286))) + (x0 + 2.6449335)
            fn = np.exp(x0) - (x2 * (0.07335293 / x1))
            # fn = np.exp(x0) - (((x2 * (0.089986384 - (1.5164394 / (420.57715 - x1)))) / x1) - 0.40996122)
        elif secondary_species == "H":
            # fn = (np.exp(x0) + -0.6378883) - (x1 / (x2 - (x1 * 27.900202)))
            # fn = ((-3.9062214 - (x1 / (x2 - (27.763603 * x1)))) / np.exp((np.exp((-3.9062214 - x0) * x0) / x1) + np.exp(x0 + -0.11166532))) + np.exp(9.85995 / np.log(x2))
            # fn = np.exp(9.859399 / np.log(x2)) + ((-3.9076896 - (x1 / (x2 - (27.762503 * x1)))) / np.exp((np.exp(x0 * (-3.9062726 - x0)) / x1) + np.exp(x0)))
            # fn = (-0.6201157 - (4.2224436 / ((x2**0.62346363) - x1))) + np.exp(x0)
            # fn = (((x1 * -0.0004224998) + np.exp(x0 + 1.292497)) + -0.60875523) - (4.2287445 / ((x2 ** 0.6235566) - x1))
            # fn = (np.exp(x0) + (((np.exp((np.exp(x0) * -136381.97) + -0.4070354) * x0) + (0.9994636 ** (x1 / np.exp(x0)))) - (4.025868 / ((x2 ** 0.6234068) - x1)))) + -0.6311273
            fn = (x0 / 1.5825349) / np.exp(np.exp(((((0.23073731 ** (((0.746305 - ((np.log(x1) - x0) ** (-4.958918 - x0))) / 0.22926822) * x0)) ** 0.23119897) - x0) / x1) / 0.23119897))
        return np.exp(fn)
    # eqn = (2.0459936 / (x1 + (x2 / -26.194601))) + (((2.6340475 - ((x2 / x1) * 0.17239113)) / (np.exp(x0) + 0.9716166)) + x0)


    # plot array_KD vs array_X only for phase = MgSiO3 -- df_superset
    
    # 4 separate figures for KD vs X, KD_prime vs X, D_wt vs X, and D_wt vs Xw
    # create a 4 x 5 grid of subplots where each row corresponds to a different target "Target pressure (GPa)"
    # and each column corresponds to a different "Target temperature (K)" in ascending order
    fig_1, axes_1 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
    axes_1 = axes_1.flatten()  # flatten the 2D array of axes to 1D for easier iteration
    fig_2, axes_2 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
    axes_2 = axes_2.flatten()  # flatten the 2D array of axes to 1D for easier iteration
    fig_3, axes_3 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
    axes_3 = axes_3.flatten()  # flatten the 2D array of axes to 1D for easier iteration
    fig_4, axes_4 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
    axes_4 = axes_4.flatten()  # flatten the 2D array of axes to 1D for easier iteration
    fig_5, axes_5 = plt.subplots(4, 5, figsize=(16, 16), sharex=True, sharey=True)
    axes_5 = axes_5.flatten()  # flatten the 2D array of axes to 1D for easier iteration
    
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

        array_T = row[f"array_T_{secondary_species}"]
        array_P = row[f"array_P_{secondary_species}"]

        if secondary_species == "He":
            KD_chosen = array_KD
            KD_chosen_lower = array_KD_lower
            KD_chosen_upper = array_KD_upper
        elif secondary_species == "H":
            KD_chosen = array_KD_prime
            KD_chosen_lower = array_KD_prime_lower
            KD_chosen_upper = array_KD_prime_upper
            KD_chosen_error = array_KD_prime_error
            array_X2 = array_X / (2 - array_X)  # X_H2 = X_H / (2 - X_H)
            array_X2_lower = row[f"array_X_{secondary_species}_lower"]
            array_X2_upper = row[f"array_X_{secondary_species}_upper"]
        # KD_chosen = array_KD
        # KD_chosen_lower = array_KD_lower
        # KD_chosen_upper = array_KD_upper

        ########################
        axes_2[i_axes].scatter(
            array_X, KD_chosen,
            label=f"P={row['Target pressure (GPa)']}, T={row['Target temperature (K)']}",
            **marker_opts_scatter
        )
        # axes_2[i_axes].errorbar(
        #     array_X, KD_chosen,
        #     yerr=[KD_chosen-KD_chosen_lower, KD_chosen_upper-KD_chosen],
        #     fmt='none',  # no extra marker
        #     # ecolor=cmap(norm(row["Target temperature (K)"])),  # single RGBA tuple
        #     **marker_opts_error,
        #     marker=marker_TI  # use the TI marker for errorbars
        # )
        axes_2[i_axes].fill_between(
            array_X,
            KD_chosen_lower,
            KD_chosen_upper,
            # KD_chosen-KD_chosen_error,
            # KD_chosen+KD_chosen_error,
            alpha=0.2, label='Error range',
            color=axes_2[i_axes].collections[0].get_edgecolor()  # same as the scatter points
        )

        # plot the best fit line
        if secondary_species == "He":
            x0 = np.log(array_X)
            secondary_species_label = "He"
        elif secondary_species == "H":
            x0 = np.log(array_X2)
            secondary_species_label = "H2"
        x1 = array_P
        x2 = array_T
        # print("array_P:", array_P)
        y_fit = best_fn(x0, x1, x2)
        axes_2[i_axes].plot(
            array_X, y_fit,
            linestyle='--',
            label=f"Best fit model",
            color='black', linewidth=1.5
        )
        ########################




    # set x and y limits for all axes
    for axes in [axes_1, axes_2, axes_3, axes_4, axes_5]:
    # for axes in [axes_2]:
        for i_axes, ax in enumerate(axes):
            # if axes is not axes_2:
            #     if axes is not axes_1:
            #         ax.set_xscale("log")
            ax.set_xscale("log")
            # if axes is not axes_2:
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.5)

            if i_axes not in (0, 5, 10, 15):
                ax.tick_params(labelleft=False)
            else:
                ax.tick_params(labelleft=True)
                if axes is axes_1:
                    ax.set_ylabel(f"K$_D$")
                elif axes is axes_2:
                    if secondary_species == "He":
                        ax.set_ylabel(f"K$_D^{{He}}$")
                    elif secondary_species == "H":
                        ax.set_ylabel(f"K$_D^{{H_{{2,sil}}\\rightleftharpoons H_{{Fe}}}}$")
                elif axes is axes_3:
                    ax.set_ylabel(f"D$_{{wt}}$")
                elif axes is axes_4:
                    ax.set_ylabel(f"D$_{{wt}}$")
                elif axes is axes_5:
                    ax.set_ylabel(f"X$_{{w,{secondary_species_label}}}$")

            if i_axes >= 15:
                ax.tick_params(labelbottom=True)         # force it ON explicitly
                # If shared x hides it, unhide individual labels:
                for lbl in ax.get_xticklabels():
                    lbl.set_visible(True)
                if axes is not axes_4 and axes is not axes_5:
                    ax.set_xlabel(fr"$X_{{{secondary_species_label}}}^{{MgSiO_3}}$")
                elif axes is axes_4:
                    ax.set_xlabel(fr"$X_{{w,{secondary_species_label}}}^{{MgSiO_3}}$")
                elif axes is axes_5:
                    ax.set_xlabel(fr"$X_{{{secondary_species_label}}}^{{MgSiO_3}}$")
            else:
                ax.tick_params(labelbottom=False)

            # delete sub-plots that are not used -- i_axes = 4,9,14
            if i_axes in (4, 9, 14):
                ax.remove()

            # legend
            if i_axes < 15:
                ax.legend(loc="lower right", fontsize=8)
            else:
                ax.legend(loc="lower right", fontsize=8)

            # x axis limits
            # if axes is not axes_3 and axes is not axes_4:
            #     ax.set_xlim(1e-4, 100/109)
            # else:
            #     ax.set_xlim(1e-4, 0.1)  # set x
            # ax.set_xlower(1e-4)
            # ax.set_xlim(1e-4, None)


    # set the title for each figure
    # fig_1.suptitle(
    #     f"K$_D$ vs X for {secondary_species}",
    #     fontsize=10
    # )
    if secondary_species == "He":
        fig_2.suptitle(
            f"K$_D^{{He}}$ vs X for {secondary_species_label}",
            fontsize=10
        )
    elif secondary_species == "H":
        fig_2.suptitle(
            f"K$_D^{{H_{{2,sil}}\\rightleftharpoons H_{{Fe}}}}$ vs X for {secondary_species_label}",
            fontsize=10
        )
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

    # save the figures
    # fig_1.savefig(f"array__KD_vs_X.png", dpi=300)
    fig_2.savefig(f"fit__array__KD_prime_vs_X.png", dpi=300)
    # fig_3.savefig(f"array__D_wt_vs_X.png", dpi=300)
    # fig_4.savefig(f"array__D_wt_vs_Xw.png", dpi=300)
    # fig_5.savefig(f"array__X_vs_Xw.png", dpi=300)

print("Done!")