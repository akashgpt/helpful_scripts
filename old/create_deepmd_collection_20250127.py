from __future__ import print_function
import os
import pandas as pd
import numpy as np
import subprocess
import time
import json
import glob
import sys
import shutil
import matplotlib.pyplot as plt
import logging
import sys


# import $SCRATCH environment variable
MY_SCRATCH = os.environ['SCRATCH']
MY_SCRATCH = MY_SCRATCH+"/"

MY_APPTAINER_REPO = os.environ['APPTAINER_REPO']
MY_APPTAINER_REPO = MY_APPTAINER_REPO+"/"


### mode ###
#  1: create deepmd_collection (by searching "recal") and run dp test; 
#  2: concatenate individual e,f,v files; 
#  3: just run dp test;
#  4: like mode=1 by searching for "deepmd" (PREFERRED vs 1)
# -1: both 1 and 2;
#  5: compare TRAIN and TEST folder sub-directories versus myinput.json file
#  6: anaylze distribution of errors in all_dp_test.{e_peratom,f,v_peratom}.out files -> create histograms,
#     detail rmse values of bad simulations and the respective frame numbers and other stats 
#  7: delete the frames with high errors from the respective sims in the *TRAIN and *TEST folders and recompile
mode = 2

n_parallel_analysis = 40 # no. of parallel analysis runs to do

# Fresh analysis?
fresh_analysis = False #for mode 1, 3 and 4 only

verbose = True

skip_dp_test = True #for mode 4 only

update_deepmd_w_OUTCAR = False #for mode 3 only and only after mode 3, 2 and 6 have been run once
exclude_index_file = "mapping__B_sigma_details.f.csv"

virial_conversion = False #for mode 3 only

enable_debug = False

# dp_model_version = f"v5_i57__v5_only" # ~ label for current analysis run
dp_model_version = f"v5_i71__all_960X3_fittingNN" # ~ label for current analysis run
dp_model_file_name = "v5_i71__all_960X3_fittingNN__pv_comp.pb"

percentile__A_sigma = 99.74#[95]
percentile__B_sigma = 99.994#[99.74]

f_threshold = 0.1 #ev A^-1
v_threshold = 0.1 #ev A^3
v_GPa_threshold = 0.1 #GPa
e_threshold = 10e-3 #ev

### Define the base directory where raw data is searched for ###
# primary_base_dir = "/scratch/gpfs/ag5805/qmd_data/NH3_MgSiO3/sim_data_ML/"
# primary_base_dir = MY_SCRATCH+"qmd_data/YihangDengetal2024/MgSiOH_dataset/train_test_data"
primary_base_dir = MY_SCRATCH+"qmd_data/NH3_MgSiO3/sim_data_ML/"
# primary_base_dir = MY_SCRATCH+"qmd_data/H2O_H2/sim_data"
base_dir = primary_base_dir

### Define the file path ###
# file_path = os.path.join(base_dir, "TRAIN_MLMD_parameters.txt")
# RUNID_PREFIX=get_next_line_after_keyword(file_path, "RUNID_PREFIX")
RUNID_PREFIX = "v5_"

dp_container = f"{MY_APPTAINER_REPO}deepmd-kit_latest.sif"

####################################################################################################################


### Define the destination directories -- Defaults ###
destination_dir= MY_SCRATCH+"qmd_data/MgSiOHN/"
base_destination_dir = destination_dir+"deepmd_collection"
# destination_dir_TRAIN = base_destination_dir+"_TRAIN__v5_only/"
# destination_dir_TEST = base_destination_dir+"_TEST__v5_only/"
destination_dir_TRAIN = base_destination_dir+"_TRAIN_master_v2/"
destination_dir_TEST = base_destination_dir+"_TEST_master_v2/"

setup_MLMD_dir = MY_SCRATCH+"qmd_data/MgSiOHN/setup_MLMD/"

myinput_json_dir = setup_MLMD_dir+"input_JSON/"
myinput_json_file = myinput_json_dir+"myinput.json"

dp_model_dir = setup_MLMD_dir+"latest_trained_potential/"
dp_model = dp_model_dir+dp_model_file_name

reference_analysis_runs_dir = setup_MLMD_dir+"reference_analysis_runs/"
dp_model_version_dir = os.path.join(reference_analysis_runs_dir, dp_model_version)
os.system(f"mkdir -p {dp_model_version_dir}")

# initialize the output log file
output_log_file = os.path.join(dp_model_version_dir, "log.output")
os.system(f"rm -f {output_log_file}")
os.system(f"touch {output_log_file}")

# also make a copy of dp_model in dp_model_version_dir
os.system(f"cp {dp_model} {dp_model_version_dir}")


# exclude_index_file_dir = 

# Initialize an empty list to store data
data_list = []


# erase any previous "*.active" files
os.system(f"rm -f {destination_dir_TRAIN}*.active")
os.system(f"rm -f {destination_dir_TEST}*.active")


eV_A3_to_GPa = 160.21766208 # 1 eV/A^3 = 160.21766208 GPa



####################################################################################################################
# Function to configure logging with adjustable levels
def configure_logging(output_log_file, enable_debug=False):
    """
    Configures logging to write to a log file and print to the console.

    Parameters:
        output_log_file (str): Path to the log file.
        enable_debug (bool): If True, include DEBUG outputs; otherwise, suppress DEBUG outputs.
    """
    log_level = logging.DEBUG if enable_debug else logging.INFO  # Set logging level based on enable_debug

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_log_file),  # Log to a file
            logging.StreamHandler(sys.stdout),  # Log to the console
        ]
    )

    # Redirect stdout and stderr to the logger
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
        def write(self, message):
            if message.strip():  # Avoid logging empty messages
                self.logger.log(self.level, message)
        def flush(self):  # Required for Python's I/O
            pass

    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger(), logging.ERROR)

configure_logging(output_log_file, enable_debug=enable_debug)

####################################################################################################################
####################################################################################################################
# # write the entire output to a log file, in addition to printing it to the console
# # Configure logging
# log_file = output_log_file
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_file),  # Log to a file
#         logging.StreamHandler(sys.stdout),  # Log to the console
#     ]
# )

# class LoggerWriter:
#     def __init__(self, logger, level):
#         self.logger = logger
#         self.level = level
#     def write(self, message):
#         if message.strip():  # Avoid logging empty messages
#             self.logger.log(self.level, message)
#     def flush(self):  # Required for Python's I/O
#         pass

# sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
# sys.stderr = LoggerWriter(logging.getLogger(), logging.ERROR)
####################################################################################################################
####################################################################################################################

# print all the variables
print(f"MY_SCRATCH: {MY_SCRATCH}")
print(f"MY_APPTAINER_REPO: {MY_APPTAINER_REPO}")

print("")

print(f"mode: {mode}")
print(f"n_parallel_analysis: {n_parallel_analysis}")
print(f"fresh_analysis: {fresh_analysis}")
print(f"verbose: {verbose}")
print(f"skip_dp_test: {skip_dp_test}")
print(f"virial_conversion: {virial_conversion}")
print(f"dp_model_version: {dp_model_version}")
print(f"RUNID_PREFIX: {RUNID_PREFIX}")
print(f"dp_container: {dp_container}")
print(f"primary_base_dir/base_dir: {base_dir}")

print("")

print(f"destination_dir: {destination_dir}")
print(f"destination_dir_TRAIN: {destination_dir_TRAIN}")
print(f"destination_dir_TEST: {destination_dir_TEST}")
print(f"myinput_json_file: {myinput_json_file}")
print(f"dp_model: {dp_model}")
print(f"dp_model_version_dir: {dp_model_version_dir}")


print("")








starting_dir = os.getcwd()


start_time = time.time()

####################################################################################################################
####################################################################################################################
####################################################################################################################
def get_next_line_after_keyword(file_path, keyword):
    """
    Searches for a keyword in a file and returns the line immediately after it.

    Parameters:
        file_path (str): Path to the file to be searched.
        keyword (str): The keyword to search for.

    Returns:
        str: The line after the line containing the keyword, stripped of whitespace.
        None: If the keyword is not found or there is no subsequent line.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if keyword in line:
                    if i + 1 < len(lines):
                        return lines[i + 1].strip()  # Return the next line
                    else:
                        return None  # No line after the keyword
        return None  # Keyword not found
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")

#####################################    
#####################################

# Function to check if dp_test files exist in the local directory
def dp_test_files_exist(deepmd_path_new="."):
    """
    Check if all required dp_test files exist in the specified path.
    
    Parameters:
    deepmd_path_new (str): The directory to check for dp_test files. Defaults to the current directory.
    
    Returns:
    bool: True if all required files exist, False otherwise.
    """
    # File types to check
    required_files = [
        "log.dp_test",
        "dp_test.e.out",
        "dp_test.v.out",
        "dp_test.f.out",
        "dp_test.e_peratom.out",
        "dp_test.v_peratom.out",
    ]

    # Check if all required files exist in the specified directory
    return all(os.path.exists(os.path.join(deepmd_path_new, file)) for file in required_files)

#########################
#########################


def count_active_files(directories, extension="*.active"):
    """
    Count the number of active files with a given extension in multiple directories.

    Args:
        directories (list): List of directory paths to search in.
        extension (str): File extension to search for (default is "*.active").
    
    Returns:
        int: Total number of active files across all directories.
    """
    count = 0
    for directory in directories:
        count += len(glob.glob(os.path.join(directory, extension)))
    return count



#############################
#############################


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_rmse_histogram_v0(rmse_values, filename, base_dir):
    # Define the desired percentiles
    percentiles = [1, 10, 50, 90, 99]
    percentile_values = np.percentile(rmse_values, percentiles)
    mean_value = np.mean(rmse_values)
    median_value = np.median(rmse_values)
    
    # Define colors for percentile lines based on distance from the 50th percentile
    def get_color(percentile):
        if percentile == 50:
            return "green"  # Median
        elif percentile < 50:
            return "blue"  # Lower percentiles
        else:
            return "red"  # Higher percentiles
    
    # Get min and max for x-axis limits
    x_min, x_max = np.min(rmse_values), np.max(rmse_values)
    
    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1]})
    
    # Left Panel: Linear Binning
    sns.histplot(rmse_values, bins="auto", kde=True, stat="frequency", color="blue", edgecolor="black", ax=axes[0])
    for perc, value in zip(percentiles, percentile_values):
        color = get_color(perc)
        axes[0].axvline(value, color=color, linestyle='--', label=f'{perc}th Percentile ({value:.2f})')
    axes[0].axvline(mean_value, color='purple', linestyle='-', label=f'Mean ({mean_value:.2f})')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_xlabel("RMSE (Linear Scale)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Linear Binning", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)
    
    # Right Panel: Log-Scaled Binning
    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), num=50)  # 50 bins in log scale
    sns.histplot(rmse_values, bins=log_bins, kde=False, stat="frequency", color="green", edgecolor="black", ax=axes[1])
    for perc, value in zip(percentiles, percentile_values):
        color = get_color(perc)
        axes[1].axvline(value, color=color, linestyle='--', label=f'{perc}th Percentile ({value:.2f})')
    axes[1].axvline(mean_value, color='purple', linestyle='-', label=f'Mean ({mean_value:.2f})')
    axes[1].set_xscale("log")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_xlabel("RMSE (Log Scale)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Logarithmic Binning", fontsize=14)

    # Avoid duplicate legends in the right panel
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)
    
    # Set the main title as the filename
    fig.suptitle(f"RMSE Distribution: {filename}", fontsize=16, y=1.02)

    # Save the combined figure
    output_path = os.path.join(base_dir, f"hist_rmse.{filename}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved to {output_path}")








import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_rmse_histogram_v1(rmse_values, filename, base_dir):
    # Define the desired percentiles
    # percentiles = [0.13, 2.22, 50, 97.78 ,99.87]
    rmse_values = rmse_values * 1000
    percentiles = [50, percentile__A_sigma, percentile__B_sigma]
    percentile_values = np.percentile(rmse_values, percentiles)
    print(f"Plotting histograms -- percentile_values: {percentile_values}")
    mean_value = np.mean(rmse_values)
    
    # Define colors for percentile lines based on distance from the 50th percentile
    def get_color(percentile):
        if percentile == 50:
            return "green"  # Median
        elif percentile == percentile__A_sigma:
            return "blue"  # Lower percentiles
        elif percentile == percentile__B_sigma:
            return "red"  # Higher percentiles
    
    # Get min and max for x-axis limits
    x_min, x_max = np.min(rmse_values), np.max(rmse_values)
    
    # Create the figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # Top Left: Linear Binning
    sns.histplot(rmse_values, bins="auto", kde=True, stat="frequency", color="blue", edgecolor="black", ax=axes[0, 0])
    for perc, value in zip(percentiles, percentile_values):
        color = get_color(perc)
        axes[0, 0].axvline(value, color=color, linestyle='--', label=f'{perc}th Percentile ({value:.2f})')
    axes[0, 0].axvline(mean_value, color='purple', linestyle='-', label=f'Mean ({mean_value:.2f})')
    axes[0, 0].set_xlim(x_min, x_max)
    axes[0, 0].set_xlabel("RMSE (Linear Scale)", fontsize=12)
    axes[0, 0].set_ylabel("Frequency", fontsize=12)
    axes[0, 0].set_title("Linear Binning", fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0, 0].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)
    
    # Top Right: Logarithmic Binning
    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), num=50)  # 50 bins in log scale
    sns.histplot(rmse_values, bins=log_bins, kde=False, stat="frequency", color="green", edgecolor="black", ax=axes[0, 1])
    for perc, value in zip(percentiles, percentile_values):
        color = get_color(perc)
        axes[0, 1].axvline(value, color=color, linestyle='--', label=f'{perc}th Percentile ({value:.2f})')
    axes[0, 1].axvline(mean_value, color='purple', linestyle='-', label=f'Mean ({mean_value:.2f})')
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlim(x_min, x_max)
    axes[0, 1].set_xlabel("RMSE (Log Scale)", fontsize=12)
    axes[0, 1].set_ylabel("Frequency", fontsize=12)
    axes[0, 1].set_title("Logarithmic Binning", fontsize=14)

    # Bottom Left: Linear Binning with Log Y-axis
    sns.histplot(rmse_values, bins="auto", kde=True, stat="frequency", color="blue", edgecolor="black", ax=axes[1, 0])
    for perc, value in zip(percentiles, percentile_values):
        color = get_color(perc)
        axes[1, 0].axvline(value, color=color, linestyle='--', label=f'{perc}th Percentile ({value:.2f})')
    axes[1, 0].axvline(mean_value, color='purple', linestyle='-', label=f'Mean ({mean_value:.2f})')
    axes[1, 0].set_xlim(x_min, x_max)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(1e-1,)
    axes[1, 0].set_xlabel("RMSE (Linear Scale)", fontsize=12)
    axes[1, 0].set_ylabel("Log Frequency", fontsize=12)
    axes[1, 0].set_title("Linear Binning with Log Y-axis", fontsize=14)

    # Bottom Right: Logarithmic Binning with Log Y-axis
    sns.histplot(rmse_values, bins=log_bins, kde=False, stat="frequency", color="green", edgecolor="black", ax=axes[1, 1])
    for perc, value in zip(percentiles, percentile_values):
        color = get_color(perc)
        axes[1, 1].axvline(value, color=color, linestyle='--', label=f'{perc}th Percentile ({value:.2f})')
    axes[1, 1].axvline(mean_value, color='purple', linestyle='-', label=f'Mean ({mean_value:.2f})')
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(1e-1,)
    axes[1, 1].set_xlabel("RMSE (Log Scale)", fontsize=12)
    axes[1, 1].set_ylabel("Log Frequency", fontsize=12)
    axes[1, 1].set_title("Logarithmic Binning with Log Y-axis", fontsize=14)

    # Set the main title as the filename with date
    fig.suptitle(f"RMSE Distribution (in meV | meV/A): {filename} ({time.strftime('%Y-%m-%d %H:%M:%S')})", fontsize=16, y=1.02)
    # fig.suptitle(f"RMSE Distribution: {filename}", fontsize=16, y=1.02)

    # Adjust layout and save the figure
    plt.tight_layout()
    output_path = os.path.join(base_dir, f"hist_rmse.{filename}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved to {output_path}")


from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

def plot_rmse_histogram_parallel(rmse_values, filename, base_dir):
    # Scale RMSE values
    rmse_values = rmse_values * 1000

    # Define percentiles and precompute values
    percentiles = [50, percentile__A_sigma, percentile__B_sigma]
    percentile_values = np.percentile(rmse_values, percentiles)
    mean_value = np.mean(rmse_values)

    # Precompute min/max for axes limits
    x_min, x_max = np.min(rmse_values), np.max(rmse_values)

    # Define binning strategies
    linear_bins = "auto"
    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), num=50)

    # Function to plot individual subplots
    def plot_histogram(ax, bins, xscale=None, yscale=None, title=""):
        sns.histplot(rmse_values, bins=bins, kde=True, stat="frequency", color="blue", edgecolor="black", ax=ax)
        for perc, value in zip(percentiles, percentile_values):
            color = "green" if perc == 50 else "blue" if perc == percentile__A_sigma else "red"
            ax.axvline(value, color=color, linestyle='--', label=f'{perc}th Percentile ({value:.2f})')
        ax.axvline(mean_value, color='purple', linestyle='-', label=f'Mean ({mean_value:.2f})')
        ax.set_xlim(x_min, x_max)
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
        ax.set_xlabel("RMSE", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(title, fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc="upper right", fontsize=10)

    # Define the plotting tasks
    tasks = [
        {"bins": linear_bins, "xscale": None, "yscale": None, "title": "Linear Binning"},
        {"bins": log_bins, "xscale": "log", "yscale": None, "title": "Logarithmic Binning"},
        {"bins": linear_bins, "xscale": None, "yscale": "log", "title": "Linear Binning with Log Y-axis"},
        {"bins": log_bins, "xscale": "log", "yscale": "log", "title": "Logarithmic Binning with Log Y-axis"},
    ]

    # Create the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Function to execute each plotting task
    def plot_task(idx):
        task = tasks[idx]
        plot_histogram(axes[idx], **task)

    # Use ThreadPoolExecutor for parallel plotting
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(plot_task, range(len(tasks)))

    # Set the main title
    fig.suptitle(f"RMSE Distribution (in meV | meV/A): {filename} ({time.strftime('%Y-%m-%d %H:%M:%S')})", fontsize=16, y=1.02)

    # Save the figure
    output_path = os.path.join(base_dir, f"hist_rmse.{filename}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Histogram saved to {output_path}")


####################################################################################################################
####################################################################################################################

def update_dp_model_version_dir():
    """
    Copies files to the specified dp_model_version directory within the parent directory.

    Returns:
        None
    """
    if dp_model_version:
        # Create necessary directories
        os.makedirs(dp_model_version_dir, exist_ok=True)
        os.makedirs(os.path.join(dp_model_version_dir, 'TRAIN'), exist_ok=True)
        os.makedirs(os.path.join(dp_model_version_dir, 'TEST'), exist_ok=True)
        
        # Copy files to the TRAIN directory
        os.system(f"cp -rf {destination_dir_TRAIN}/*dp_test* {os.path.join(dp_model_version_dir, 'TRAIN')}")
        os.system(f"cp -rf {destination_dir_TRAIN}/*MLMD_vs* {os.path.join(dp_model_version_dir, 'TRAIN')}")
        os.system(f"cp -rf {destination_dir_TRAIN}/*hist* {os.path.join(dp_model_version_dir, 'TRAIN')}")
        os.system(f"cp -rf {destination_dir_TRAIN}/*mapping_* {os.path.join(dp_model_version_dir, 'TRAIN')}")

        # Copy files to the TEST directory
        os.system(f"cp -rf {destination_dir_TEST}/*dp_test* {os.path.join(dp_model_version_dir, 'TEST')}")
        os.system(f"cp -rf {destination_dir_TEST}/*MLMD_vs* {os.path.join(dp_model_version_dir, 'TEST')}")
        os.system(f"cp -rf {destination_dir_TEST}/*hist* {os.path.join(dp_model_version_dir, 'TRAIN')}")
        os.system(f"cp -rf {destination_dir_TEST}/*mapping_* {os.path.join(dp_model_version_dir, 'TRAIN')}")
        
        # Copy the input JSON file
        os.system(f"cp {myinput_json_file} {dp_model_version_dir}")


####################################################################################################################
####################################################################################################################
####################################################################################################################


def search_bad_indices_for_keyword(file_path, keyword, output_file=None):
    """
    Reads a CSV file, extracts the second column, and creates a space-separated string.
    Optionally, writes the result to an output file.

    Parameters:
        file_path (str): Path to the input CSV file.
        output_file (str): Path to the output file to save the result (optional).

    Returns:
        str: A space-separated string of the second column's values.
    """
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)

    # Ensure the CSV has at least 2 columns
    if df.shape[1] < 2:
        raise ValueError("The CSV file must have at least two columns.")

    # Extract the second column where the first column contains {keyword} and convert it to a space-separated string
    result = " ".join(map(str, df[df.iloc[:, 0].str.contains(keyword)].iloc[:, 1]))
    # result = " ".join(map(str, df.iloc[:, 1]))

    # Write to output file if provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)

    return result


####################################################################################################################
####################################################################################################################
####################################################################################################################


####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################


####################################################################################################################
####################################################################################################################
####################################################################################################################



if mode==1 or mode==-1:
    # Walk through the current directory to find "recal" folders
    for root, dirs, files in os.walk(base_dir):

        # sort the directories
        dirs.sort()

        for directory in dirs:
            if directory == "deepmd":

                # Define the paths to the relevant files
                recal_path = os.path.join(root, directory) ## check for DP_TEST_INPUT_FILE
                deepmd_path = os.path.join(recal_path,"deepmd/")

                # print(f"Early Processing {deepmd_path}")

                # continue if recal_path starts with base_dir+"v5_"
                if not recal_path.startswith(base_dir+RUNID_PREFIX):
                # if not recal_path.startswith(base_dir+"v5_i34/"):
                    continue

                # go to the folder containing the deepmd folder
                os.chdir(recal_path)
                print(f"Processing {root}")

                ############################################
                ############################################

                # replace base_dir with destination_dir in deepmd_path and recal_path and rename as deepmd_path_new and recal_path_new
                deepmd_path_new_TRAIN = deepmd_path.replace(base_dir, destination_dir_TRAIN)
                recal_path_new_TRAIN = recal_path.replace(base_dir, destination_dir_TRAIN)

                # create the recal_path_new and copy everything inside recal_path there
                os.makedirs(recal_path_new_TRAIN, exist_ok=True)
                os.system(f"cp -r {recal_path}/* {recal_path_new_TRAIN}")

                # move to the new directory
                os.chdir(recal_path_new_TRAIN)

                # remove all folders in this new directory whose names are a number
                os.system("rm -rf [0-9]*")

                # remove all files except the remaining folder
                os.system("rm -f * 2>/dev/null")

                if fresh_analysis: #if true, this section runs
                    
                    # cd to deepmd_path_new and remove certain files
                    os.chdir(deepmd_path_new_TRAIN)
                    os.system("rm -rf *.out log.*")
                    
                    #remove all folders except set.000
                    command_to_clean = f'find {deepmd_path_new_TRAIN}/* -type d ! -name "set.000" -exec rm -rf {{}} +'
                    try:
                        subprocess.run(command_to_clean, shell=True, check=True)
                        # print("Successfully removed all folders except 'set.000'")
                    except subprocess.CalledProcessError as e:
                        print(f"Error while executing command: {e}")

                    # run command if set.000 exists
                    if os.path.exists("set.000"):
                        os.system(f"dp test -m {dp_model} -d dp_test -n 0 > log.dp_test 2>&1 &")

                ############################################
                ############################################

                # replace base_dir with destination_dir in deepmd_path and recal_path and rename as deepmd_path_new and recal_path_new
                deepmd_path_new_TEST = deepmd_path.replace(base_dir, destination_dir_TEST)
                recal_path_new_TEST = recal_path.replace(base_dir, destination_dir_TEST)

                # create the recal_path_new and copy everything inside recal_path there
                os.makedirs(recal_path_new_TEST, exist_ok=True)
                os.system(f"cp -r {recal_path}/* {recal_path_new_TEST}")

                # move to the new directory
                os.chdir(recal_path_new_TEST)

                # remove all folders in this new directory whose names are a number
                os.system("rm -rf [0-9]*")

                # remove all files except the remaining folder
                os.system("rm -f * 2>/dev/null")

                if fresh_analysis: #if true, this section runs
                    
                    # cd to deepmd_path_new and remove certain files
                    os.chdir(deepmd_path_new_TEST)
                    os.system("rm -rf *.out log.*")
                    
                    #remove all folders except set.000
                    command_to_clean = f'find {deepmd_path_new_TEST}/* -type d ! -name "set.001" -exec rm -rf {{}} +'
                    try:
                        subprocess.run(command_to_clean, shell=True, check=True)
                        # print("Successfully removed all folders except 'set.001'")
                    except subprocess.CalledProcessError as e:
                        print(f"Error while executing command: {e}")

                    # run command if set.001 exists
                    if os.path.exists("set.001"):
                        os.system(f"apptainer exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m {dp_model} -d dp_test -n 0 > log.dp_test 2>&1")


    # print time taken
    end_time1 = time.time()
    print("Time taken: ", end_time1-start_time)



####################################################################################################################
####################################################################################################################
####################################################################################################################





import os
import pandas as pd


def process_base_dir(base_dir, run_prefix):
    """
    Processes a given base directory to concatenate files while keeping headers.

    Parameters:
    - base_dir (str): The directory to process.
    - run_prefix (str): Prefix for the output file names (e.g., TRAIN or TEST).
    """

    # File types and their headers
    files_to_read = {
        "dp_test.e.out": None,
        "dp_test.v.out": None,
        "dp_test.f.out": None,
        "dp_test.e_peratom.out": None,
        "dp_test.v_peratom.out": None,
        "dp_test.v_GPa.out": None,
        "dp_test.v_GPa_peratom.out": None,
    }

    # Cumulative DataFrames for concatenated data
    all_dataframes = {key: pd.DataFrame() for key in files_to_read.keys()}



    # Define the keywords to search for and their corresponding columns
    keywords = [
        "Energy RMSE ",
        "Energy RMSE/Natoms",
        "Force  RMSE",
        "Virial RMSE ",
        "Virial RMSE/Natoms"
    ]
    # Define the DataFrame columns
    columns = ["runID_dir", "n_frames", "n_atoms"] + [keyword.strip() for keyword in keywords]

    # Initialize a list to hold extracted data
    log_dp_test_data = []

    log_deepmd_path = []

    counter_processed = 0



    # Walk through the base directory to find "recal" folders
    for root, dirs, files in os.walk(base_dir):

        # sort the directories
        dirs.sort()

        for directory in dirs:
            #if directory contains string "deepmd", then process
            if "deepmd" in directory:
            # if directory == "deepmd":

                print("")

                folder_path = os.path.join(root, directory)
                # deepmd_path = os.path.join(folder_path, "deepmd")
                deepmd_path = folder_path

                # if os.path.dirname(deepmd_path) contains "pre", then parent directory is os.path.dirname(deepmd_path), otherwise, parent directory is os.path.dirname(deepmd_path+"../../")
                if ".recal" in os.path.basename(os.path.dirname(deepmd_path)):
                    runID_dir = (os.path.dirname(deepmd_path))
                else:
                    runID_dir = (os.path.dirname(os.path.dirname(os.path.dirname(deepmd_path))))

                # if not folder_path.startswith(os.path.join(base_dir, "v5_i16")):
                # if not deepmd_path.startswith(os.path.join(base_dir,RUNID_PREFIX)):
                #     continue

                print(f"Processing {runID_dir}")

                if not "recal" in os.path.dirname(deepmd_path):
                    print(f"*** {os.path.dirname(deepmd_path)} does not contain 'recal'. Skipping. ***")
                    continue

                # Process files in the deepmd folder
                for filename in files_to_read.keys():
                    file_path = os.path.join(deepmd_path, filename)

                    if os.path.exists(file_path):
                        try:
                            # Read the header (first line)
                            with open(file_path, "r") as f:
                                header = f.readline().strip()
                                if files_to_read[filename] is None:
                                    files_to_read[filename] = header  # Store the header once

                            # Read the file content excluding the header
                            df = pd.read_csv(file_path, sep=r'\s+', header=None, skiprows=1)

                            # Concatenate into the corresponding DataFrame
                            all_dataframes[filename] = pd.concat(
                                [all_dataframes[filename], df], ignore_index=True
                            )

                            counter_processed += 1

                        except Exception as e:
                            if verbose:
                                print(f"~~~ Error processing file {file_path}: {e} ~~~")
                    else:
                        if verbose:
                            print(f"Warning: {file_path} not found. Skipping.")



                #################################
                # Extract values from log.dp_test
                log_file_path = os.path.join(deepmd_path, "log.dp_test")

                # Check if the log file exists
                if not os.path.exists(log_file_path):
                    if verbose:
                        print(f"Warning: {log_file_path} not found. Skipping.")
                    continue

                # Initialize a dictionary to store values for this folder
                row = {"runID_dir": runID_dir}
                # print(f"runID_dir: {runID_dir}")

                # create a separate dataframe to store deepmd_path
                row_deepmd_path = {"deepmd_path": deepmd_path}


                # Extract the number of data points
                try:
                    # Run grep and awk to extract the last element
                    command = f"grep -m 1 'number of test data' '{log_file_path}' | awk '{{print $(NF)}}'"
                    result = subprocess.check_output(command, shell=True, text=True).strip()
                    row["n_frames"] = result
                except subprocess.CalledProcessError:
                    # If the keyword is not found, set the value to None
                    row["n_frames"] = None

                # Extract the number of atoms = number of lines in the type.raw file
                try:
                    # Run wc to count the number of lines
                    command = f"wc -l < '{os.path.join(deepmd_path, 'type.raw')}'"
                    result = subprocess.check_output(command, shell=True, text=True).strip()
                    row["n_atoms"] = result
                except subprocess.CalledProcessError:
                    # If the keyword is not found, set the value to None
                    row["n_atoms"] = None

                # Extract values for each keyword
                for keyword in keywords:
                    try:
                        # Run grep and awk to extract the second last element
                        command = f"grep -m 1 '{keyword}' '{log_file_path}' | awk '{{print $(NF-1)}}'"
                        result = subprocess.check_output(command, shell=True, text=True).strip()
                        row[keyword.strip()] = result
                    except subprocess.CalledProcessError:
                        # If the keyword is not found, set the value to None
                        row[keyword.strip()] = None

                # Append the row to the data list
                log_dp_test_data.append(row)
                log_deepmd_path.append(row_deepmd_path)


    print("")
    print("")
    print("")

    # Write concatenated files with their headers
    for filename, df in all_dataframes.items():
        if not df.empty:
            output_file = os.path.join(base_dir, f"all_{filename}")
            with open(output_file, "w") as f:
                # Write the stored header
                f.write(files_to_read[filename] + "\n")
            # Append the concatenated data
            df.to_csv(output_file, sep=" ", index=False, header=False, mode="a")
            print(f"Written: {output_file}")



    # Create a DataFrame from the extracted data
    log_dp_test_df = pd.DataFrame(log_dp_test_data, columns=columns)
    log_deepmd_path_df = pd.DataFrame(log_deepmd_path)

    # Compute total frames and atoms and averages for numeric columns
    total_frames = pd.to_numeric(log_dp_test_df["n_frames"], errors="coerce").sum()

    total_atoms = pd.to_numeric(log_dp_test_df["n_atoms"], errors="coerce").sum()


    averages = {
    col: pd.to_numeric(log_dp_test_df[col], errors="coerce").mean(skipna=True)
    for col in columns[3:]  # Skip "runID_dir", "n_frames" and "n_atoms"
    }

    # Add the "Total" row to the DataFrame
    total_row = {"runID_dir": "Total", "n_frames": total_frames, "n_atoms": total_atoms, **averages}
    log_dp_test_df = pd.concat([log_dp_test_df, pd.DataFrame([total_row])], ignore_index=True)

    # Save the DataFrame to a CSV file
    output_csv = os.path.join(base_dir, "all_log.dp_test")
    log_dp_test_df.to_csv(output_csv, index=False)

    print(f"Extracted data saved to {output_csv}")


    # Create a new JSON file log_deepmd_path_df["deepmd_path"]
    OG_json_file = os.path.join(dp_model_version_dir, "myinput.json")
    NEW_json_file = os.path.join(dp_model_version_dir, f"{run_prefix}_myinput.json")

    # make a copy of the original myinput.json file as NEW_myinput.json
    os.system(f"cp {myinput_json_file} {OG_json_file}")
    os.system(f"cp {OG_json_file} {NEW_json_file}")

    # open this new JSON file and replace data["training"]["training_data"]["systems"] with log_deepmd_path_df["deepmd_path"]
    with open(NEW_json_file, "r") as f:
        data = json.load(f)
        # old_data_systems = data["training"]["training_data"]["systems"]

    # replace the old_data_systems with log_dp_test_df["runID_dir"]
    data["training"]["training_data"]["systems"] = log_deepmd_path_df["deepmd_path"].tolist()

    # write the new data to the NEW_myinput.json file
    with open(NEW_json_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Updated JSON file saved to {NEW_json_file}")






    # run python $mldp/plot_MLMD_vs_DFT_test.py -f all_dp_test
    os.chdir(base_dir)
    os.system(f"python $local_mldp/plot_MLMD_vs_DFT_test.py -f all_dp_test > log.MLMD_vs_DFT 2>&1 &")

    print("")
    print(f"Processed {counter_processed} files in {base_dir}")
    print("")
    print("")



# Process both TRAIN and TEST directories
if mode in [2, -1]:

    print("Running mode {2 or -1} ...")

    print("Processing TRAIN directory...")
    process_base_dir(destination_dir_TRAIN, run_prefix="TRAIN")

    print("Processing TEST directory...")
    process_base_dir(destination_dir_TEST, run_prefix="TEST")

    # if dp_model_version is not empty, then copy all the files to the dp_model_version directory within the parent directory
    # if dp_model_version:
    #     os.makedirs(dp_model_version_dir, exist_ok=True)
    #     os.makedirs(os.path.join(dp_model_version_dir, 'TRAIN'), exist_ok=True)
    #     os.makedirs(os.path.join(dp_model_version_dir, 'TEST'), exist_ok=True)
    #     os.system(f"cp -r {destination_dir_TRAIN}/*dp_test* {os.path.join(dp_model_version_dir, 'TRAIN')}")
    #     os.system(f"cp -r {destination_dir_TRAIN}/*MLMD_vs* {os.path.join(dp_model_version_dir, 'TRAIN')}")
    #     os.system(f"cp -r {destination_dir_TEST}/*dp_test* {os.path.join(dp_model_version_dir, 'TEST')}")
    #     os.system(f"cp -r {destination_dir_TEST}/*MLMD_vs* {os.path.join(dp_model_version_dir, 'TEST')}")

    #     #copy myinput_json_file to OG_myinput.json in 
    #     os.system(f"cp {myinput_json_file} {dp_model_version_dir}")
    update_dp_model_version_dir()


    # print time taken
    end_time1 = time.time()
    print("Time taken: ", end_time1-start_time)




####################################################################################################################
####################################################################################################################
####################################################################################################################






def dp_test_base_dir(base_dir, run_prefix):
    # Walk through the current directory to find "recal" folders
    counter_processed = 0
    for root, dirs, files in os.walk(base_dir):

        # sort the directories
        dirs.sort()

        for directory in dirs:

            #if directory contains string "deepmd", then process
            if "deepmd" in directory:
            # if directory == "deepmd":
                # Define the paths to the relevant files
                folder_path = os.path.join(root, directory)
                # deepmd_path = os.path.join(folder_path, "deepmd")
                deepmd_path = folder_path

                deepmd_path_name = os.path.basename(deepmd_path)

                # print(f"Early Processing {deepmd_path}")

                # continue if folder_path starts with base_dir+"v5_"
                # if not deepmd_path.startswith(os.path.join(base_dir,RUNID_PREFIX)):
                # # if not folder_path.startswith(base_dir+"v5_i34/"):
                #     continue

                #check if os.path.dirname(deepmd_path) contains "recal", if not, skip the loop
                if not "recal" in os.path.dirname(deepmd_path):
                    #delete os.path.dirname(deepmd_path)
                    # os.system(f"rm -rf {deepmd_path}")
                    print(f"*** {os.path.dirname(deepmd_path)} does not contain 'recal'. Skipping and deleting this deepmd folder. ***")
                    continue

                # go to the folder containing the deepmd folder
                os.chdir(deepmd_path)
                print(f"Processing {root}")
                counter_processed += 1
                # print(f"Processing {deepmd_path_name}")

                # "touch found_target_file" if deepmd_path = /scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOHN/deepmd_collection_TEST/u.h.j.j.pro-l.metad.sigma-20interval.60-80.recal/deepmd_mm4_pca
                if deepmd_path == "/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOHN/deepmd_collection_TEST/u.h.j.j.pro-l.metad.sigma-20interval.60-80.recal/deepmd_mm4_pca":
                    os.system("touch found_target_file")


                # run command if set.000/1 exists
                if run_prefix=="TRAIN":
                    search_string = "set.000"
                    destination_dir_X = destination_dir_TRAIN
                elif run_prefix=="TEST":
                    search_string = "set.001"
                    destination_dir_X = destination_dir_TEST

                if os.path.exists(search_string):
                    # try:
                    if fresh_analysis:
                        os.system("rm -f *.out log.*")
                    else:
                        os.system("rm -f *dpmd_stat*")

                    #if dp_test* files exist, skip the following command
                    if not dp_test_files_exist(deepmd_path):
                        # Directories to check for active runs
                        directories_for_active_files = [destination_dir_TRAIN, destination_dir_TEST]

                        # Wait until the number of active files is less than the parallel threshold
                        n_active = count_active_files(directories_for_active_files)
                        # print(f"Current n_active: {n_active}")
                        while n_active >= n_parallel_analysis:
                            if verbose:
                                print(f"### {n_active} active dp_test runs. Waiting for 10 seconds. ###")
                            time.sleep(10)
                            n_active = count_active_files(directories_for_active_files)

                        # Replace all "/" in deepmd_path_new with "_" for safe file naming
                        deepmd_path_new_for_filing = deepmd_path.replace("/", "_")

                        # Create the reference file
                        reference_file = os.path.join(destination_dir_X, f"{deepmd_path_new_for_filing}.active")
                        os.system(f"touch {reference_file}")

                        # update_deepmd_w_OUTCAR and if deepmd_path does not contain "u."
                        if update_deepmd_w_OUTCAR and not "u." in deepmd_path:
                            os.system("rm -r deepmd")
                            exclude_index_file_locn = os.path.join(dp_model_version_dir, run_prefix, exclude_index_file)
                            string_bad_indices = string_bad_indices = search_bad_indices_for_keyword(exclude_index_file_locn, deepmd_path_name, None)
                            os.system(f"python $local_mldp/extract_deepmd.py -d deepmd -ttr 10000 -ex {string_bad_indices}")

                        # Run the dp test command in the background
                        os.system(
                            f"apptainer exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m {dp_model} -d dp_test -n 0 > log.dp_test 2>&1 && rm {reference_file} &"
                        )
                        # os.system(
                        #     f"dp test -m {dp_model} -d dp_test -n 0 > log.dp_test 2>&1 && rm {reference_file} &"
                        # )


                        # os.system(
                        #     f"export DP_INFER_BATCH_SIZE=16384 && apptainer --debug exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m {dp_model} -d dp_test -n 0 > log.dp_test 2>&1 && rm {reference_file} &"
                        # )
                        # os.system(f"apptainer exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m {dp_model} -d dp_test -n 0 > log.dp_test 2>&1")
                    else:
                        # print(f"Processing {root}")
                        print("~~~ dp_test files already exist. Skipping. ~~~")
                    # except subprocess.CalledProcessError as e:
                    #     print(f"Error processing folder {deepmd_path}: {e}")

                    if virial_conversion:
                        # converting virial from eV to GPa
                        if not os.path.exists("dp_test.v_GPa.out") and not os.path.exists("dp_test.v_GPa_peratom.out"):
                        # if 1==1:

                            print(f"Working on virial_conversion in {deepmd_path}")#, np.shape(all_virial), np.shape(all_virial)[0], np.ndim(all_virial) )

                            all_virial = np.loadtxt('dp_test.v.out')
                            all_virial_peratom = np.loadtxt('dp_test.v_peratom.out')
                            # print(f"Reading dp_test.v_peratom.out", np.shape(all_virial_peratom), np.shape(all_virial_peratom)[0], np.ndim(all_virial_peratom) )
                            all_virial_GPa = []
                            all_virial_peratom_GPa = []

                            # number of dimensions of all_virial
                            # print(np.ndim(all_virial))

                            # load data from dp_test.v.out in pandas dataframe, while skipping the first row (save header separately)
                            # df_all_virial = pd.read_csv('dp_test.v.out', delim_whitespace=True, skiprows=1)
                            # df_all_virial_peratom = pd.read_csv('dp_test.v_peratom.out', delim_whitespace=True, skiprows=1)


                            boxes_file = search_string+'/box.npy'
                            boxes = np.load(boxes_file)
                            
                            
                            old_dim_virial = np.ndim(all_virial)
                            if old_dim_virial == 1:
                                # print(f"counter_processed: {counter_processed}")
                                # print(f"Reading {boxes_file}", len(boxes), np.shape(all_virial), np.shape(all_virial_peratom))
                                # transposing all_virial and all_virial_peratom
                                all_virial = all_virial.reshape(1, -1)
                                all_virial_peratom = all_virial_peratom.reshape(1, -1)
                                # print(f"Reading {boxes_file}", len(boxes), np.shape(all_virial), np.shape(all_virial_peratom))
                            
                            

                            if len(boxes) != np.shape(all_virial)[0] or len(boxes) != np.shape(all_virial_peratom)[0]:
                                print('!!! Error: Number of boxes does not match number of frames !!!')
                                exit()
                            # elif np.ndim(all_virial) == 1:
                            #     if len(boxes) != 1 or len(boxes) != 1:
                            #         print('!!! Error: Number of boxes does not match number of frames !!!')
                            #         exit()
                                    
                            for i in range(len(boxes)):
                                box = boxes[i]
                                a = box[:3]
                                b = box[3:6]
                                c = box[6:]
                                vol = np.dot(c,np.cross(a,b))
                                v_gpa = all_virial[i]/vol*eV_A3_to_GPa
                                all_virial_GPa.append(v_gpa)
                                v_peratom_gpa = all_virial_peratom[i]/vol*eV_A3_to_GPa
                                all_virial_peratom_GPa.append(v_peratom_gpa)

                                # if old_dim_virial == 1:
                                #     print(f"v_gpa: {v_gpa}")
                                #     print(f"np.shape(v_GPa): {np.shape(v_gpa)}")

                            #copy header from dp_test.v.out to dp_test.v_GPa.out
                            with open('dp_test.v.out', 'r') as f:
                                header = f.readline()
                            with open('dp_test.v_GPa.out', 'w') as f:
                                f.write(header)
                            with open('dp_test.v_peratom.out', 'r') as f:
                                header = f.readline()
                            with open('dp_test.v_GPa_peratom.out', 'w') as f:
                                f.write(header)

                            #append data to dp_test.v_GPa.out and dp_test.v_GPa_peratom.out while keeping the header
                            with open('dp_test.v_GPa.out', 'a') as f:
                                np.savetxt(f, all_virial_GPa)
                            with open('dp_test.v_GPa_peratom.out', 'a') as f:
                                np.savetxt(f, all_virial_peratom_GPa)


                else:
                        # print(f"Processing {root}")
                        print(f"### Warning: {search_string} not found. Skipping. ###")
                
                print("")


# Process both TRAIN and TEST directories
if mode in [3]:

    print(f"Running mode {mode} ...")
    print("")

    print("Processing TRAIN directory...")
    print("")
    dp_test_base_dir(destination_dir_TRAIN, run_prefix="TRAIN")

    print("Processing TEST directory...")
    print("")
    dp_test_base_dir(destination_dir_TEST, run_prefix="TEST")

    ################################
    # Directories to check for active runs
    directories_for_active_files = [destination_dir_TRAIN, destination_dir_TEST]

    # Wait until the number of active files is less than the parallel threshold
    n_active = count_active_files(directories_for_active_files)
    # print(f"Current n_active: {n_active}")
    while n_active > 0:
        if verbose:
            print(f"### {n_active} active dp_test runs. Waiting for all to finish. ###")
        time.sleep(10)
        n_active = count_active_files(directories_for_active_files)


    # print time taken
    end_time1 = time.time()
    print("Time taken: ", end_time1-start_time)



################################
################################
################################


def search_and_copy_dirs_2(base_dir, destination_dir_X, run_prefix):
    # Walk through the current directory to find "recal" folders
    for root, dirs, files in os.walk(base_dir):

        # sort the directories
        dirs.sort()

        for directory in dirs:
            if directory == "deepmd":
                # Define the paths to the relevant files
                folder_path = os.path.join(root, directory)
                # deepmd_path = os.path.join(folder_path, "deepmd")
                deepmd_path = folder_path
                recal_path = os.path.dirname(deepmd_path)

                deepmd_path_new = deepmd_path.replace(base_dir, destination_dir_X)
                recal_path_new = recal_path.replace(base_dir, destination_dir_X)

                # os.chdir(deepmd_path+"/../../..")
                cwd = os.path.basename((os.getcwd()))
                # # runID_basename = os.path.basename(os.path.dirname(os.path.join(deepmd_path+"../../..")))
                # print(f"Early Processing {cwd}")

                # continue if folder_path starts with base_dir+"v5_"
                if not deepmd_path.startswith(os.path.join(base_dir,RUNID_PREFIX)):
                # if not folder_path.startswith(base_dir+"v5_i34/"):
                # if not folder_path.startswith(base_dir+"n0502e"):
                    # print(f"*** {deepmd_path} does not start with '{RUNID_PREFIX}'. Skipping. ***")
                    continue

                # check if parent directory is "pre", and if yes, skip the loop
                parent_dir_basename = os.path.basename(os.path.dirname(deepmd_path))
                if parent_dir_basename == "pre":
                    # print(f"*** The parent directory is 'pre'. Skipping. ***")
                    continue

                # check if parent directory is exactly "recal", and if no, skip the loop
                if not parent_dir_basename == "recal":
                    # print(f"*** The parent directory is not 'recal'. Skipping. ***")
                    continue

                # os.chdir(deepmd_path_new)
                # # check if deepmd_path_new+"dp_test.e.out" exists, and if yes, skip the loop
                # if not dp_test_files_exist():
                #     print(f"*** {deepmd_path_new}/dp_test.*.out / log.dp_test files already exist. Skipping. ***")
                #     print("")
                #     continue
                # else:
                #     temp_str=deepmd_path_new+"/dp_test.e.out"
                #     print("doesn't exist: ", temp_str)

                # if dp_test_files_exist(deepmd_path_new):
                #     print(f"*** {deepmd_path_new}/dp_test.*.out / log.dp_test files already exist. Skipping. ***")
                #     print("")
                #     continue

                #check if directory deepmd_path_new exists
                if os.path.isdir(deepmd_path_new):
                    print(f"*** The directory '{deepmd_path_new}' exists. Skipping. ***")
                    print("")
                    continue
                else:
                    # print(f"The directory '{deepmd_path_new}' does not exist. Processing it.")
                    # go to the folder containing the deepmd folder
                    os.chdir(deepmd_path)
                    print(f"Processing {root}")

                    # create the recal_path_new and copy everything inside recal_path there
                    os.makedirs(recal_path_new, exist_ok=True)
                    os.system(f"cp -r {recal_path}/* {recal_path_new}")

                    # move to the new directory
                    os.chdir(recal_path_new)

                    # remove all folders in this new directory whose names are a number
                    os.system("rm -rf [0-9]*")

                    # remove all files except the remaining folder and file OUTCAR
                    os.system("find . -maxdepth 1 -type f ! -name 'OUTCAR' -exec rm -f {} +")
                    # os.system("rm -f * 2>/dev/null")

                    os.chdir(deepmd_path_new)


                    # run command if set.000/1 exists
                    if run_prefix=="TRAIN":
                        search_string = "set.000"
                    elif run_prefix=="TEST":
                        search_string = "set.001"

                    
                    # try:
                    if fresh_analysis:
                        os.system("rm -f *.out log.*")
                    else:
                        os.system("rm -f *dpmd_stat*")

                    #remove all folders except {search_string}
                    command_to_clean = f'find {deepmd_path_new}/* -type d ! -name "{search_string}" -exec rm -rf {{}} +'
                    try:
                        subprocess.run(command_to_clean, shell=True, check=True)
                        # print("Successfully removed all folders except 'set.000'")
                    except subprocess.CalledProcessError as e:
                        print(f"Error while executing command: {e}")

                    if not skip_dp_test:
                        if os.path.exists(search_string):
                            #if dp_test* files exist, skip the following command
                            if not dp_test_files_exist(deepmd_path_new):

                                # Directories to check for active runs
                                directories_for_active_files = [destination_dir_TRAIN, destination_dir_TEST]

                                # Wait until the number of active files is less than the parallel threshold
                                n_active = count_active_files(directories_for_active_files)
                                # print(f"Current n_active: {n_active}")
                                while n_active >= n_parallel_analysis:
                                    if verbose:
                                        print(f"### {n_active} active dp_test runs. Waiting for 10 seconds. ###")
                                    time.sleep(10)
                                    n_active = count_active_files(directories_for_active_files)

                                # Replace all "/" in deepmd_path_new with "_" for safe file naming
                                deepmd_path_new_for_filing = deepmd_path_new.replace("/", "_")

                                # Create the reference file
                                reference_file = os.path.join(destination_dir_X, f"{deepmd_path_new_for_filing}.active")
                                os.system(f"touch {reference_file}")

                                # Run the dp test command in the background
                                os.system(
                                    f"export DP_INFER_BATCH_SIZE=16384 && apptainer --debug exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m {dp_model} -d dp_test -n 0 > log.dp_test 2>&1 && rm {reference_file} &"
                                )
                            else:
                                print("~~~ dp_test files already exist. Skipping. ~~~")
                            # except subprocess.CalledProcessError as e:
                            #     print(f"Error processing folder {deepmd_path}: {e}")
                        else:
                                print(f"### Warning: {search_string} not found. Skipping. ###")
                    
                    print("")


# Process both TRAIN and TEST directories
if mode in [4]:

    print(f"Running mode {mode} ...")
    print("")

    print("Processing TRAIN directory...")
    print("")
    search_and_copy_dirs_2(primary_base_dir, destination_dir_TRAIN, run_prefix="TRAIN")

    print("Processing TEST directory...")
    print("")
    search_and_copy_dirs_2(primary_base_dir, destination_dir_TEST, run_prefix="TEST")

    ################################
    # wait while all dp_test runs are completed
    # Directories to check for active runs
    directories_for_active_files = [destination_dir_TRAIN, destination_dir_TEST]

    # Wait until the number of active files is less than the parallel threshold
    n_active = count_active_files(directories_for_active_files)
    # print(f"Current n_active: {n_active}")
    while n_active > 0:
        if verbose:
            print(f"### {n_active} active dp_test runs. Waiting for 10 seconds. ###")
        time.sleep(10)
        n_active = count_active_files(directories_for_active_files)

    # print time taken
    end_time1 = time.time()
    print("Time taken: ", end_time1-start_time)





####################################################################################################################
####################################################################################################################
####################################################################################################################



# def search_and_copy_dirs_3(base_dir, destination_dir_X, run_prefix):

#     # extract systems from training in myinput_json_file
#     with open(myinput_json_file) as f:
#         data = json.load(f)
#         data_systems = data["training"]["training_data"]["systems"]

#         # walk through data_sytems
#         for directory in data_systems:
#             deepmd_path = directory
#             recal_path = os.path.dirname(deepmd_path)

#             print(f"Processing {deepmd_path}")

#             deepmd_path_new = deepmd_path.replace(base_dir, destination_dir_X)

#             # if path contains "pre", skip the loop
#             # if "NH3_MgSiO3" in deepmd_path:
#             #     continue

#             parent_dir_basename = os.path.basename(os.path.dirname(deepmd_path))
#                 if parent_dir_basename == "pre":
#                     continue
        


    # # Walk through the current directory to find "recal" folders
    # for root, dirs, files in os.walk(base_dir):
    #     for directory in dirs:
    #         if directory == "deepmd":
    #             # Define the paths to the relevant files
    #             folder_path = os.path.join(root, directory)
    #             # deepmd_path = os.path.join(folder_path, "deepmd")
    #             deepmd_path = folder_path
    #             recal_path = os.path.dirname(deepmd_path)

    #             # os.chdir(deepmd_path+"/../../..")
    #             # cwd = os.path.basename((os.getcwd()))
    #             # # runID_basename = os.path.basename(os.path.dirname(os.path.join(deepmd_path+"../../..")))
    #             # print(f"Early Processing {cwd}")

    #             # continue if folder_path starts with base_dir+"v5_"
    #             if not deepmd_path.startswith(os.path.join(base_dir,RUNID_PREFIX)):
    #             # if not folder_path.startswith(base_dir+"v5_i34/"):
    #             # if not folder_path.startswith(base_dir+"n0502e"):
    #                 continue

    #             # check if parent directory is "pre", and if yes, skip the loop
    #             parent_dir_basename = os.path.basename(os.path.dirname(deepmd_path))
    #             if parent_dir_basename == "pre":
    #                 continue


    #             # go to the folder containing the deepmd folder
    #             os.chdir(deepmd_path)
    #             print(f"Processing {root}")

    #             deepmd_path_new = deepmd_path.replace(base_dir, destination_dir_X)
    #             recal_path_new = recal_path.replace(base_dir, destination_dir_X)

    #             # create the recal_path_new and copy everything inside recal_path there
    #             os.makedirs(recal_path_new, exist_ok=True)
    #             os.system(f"cp -r {recal_path}/* {recal_path_new}")

    #             # move to the new directory
    #             os.chdir(recal_path_new)

    #             # remove all folders in this new directory whose names are a number
    #             os.system("rm -rf [0-9]*")

    #             # remove all files except the remaining folder
    #             os.system("rm -f * 2>/dev/null")

    #             os.chdir(deepmd_path_new)


    #             # run command if set.000/1 exists
    #             if run_prefix=="TRAIN":
    #                 search_string = "set.000"
    #             elif run_prefix=="TEST":
    #                 search_string = "set.001"

                
    #             # try:
    #             if fresh_analysis:
    #                 os.system("rm -f *.out log.*")
    #             else:
    #                 os.system("rm -f *dpmd_stat*")

    #             #remove all folders except {search_string}
    #             command_to_clean = f'find {deepmd_path_new}/* -type d ! -name "{search_string}" -exec rm -rf {{}} +'
    #             try:
    #                 subprocess.run(command_to_clean, shell=True, check=True)
    #                 # print("Successfully removed all folders except 'set.000'")
    #             except subprocess.CalledProcessError as e:
    #                 print(f"Error while executing command: {e}")


    #             if os.path.exists(search_string):
    #                 #if dp_test* files exist, skip the following command
    #                 if not os.path.exists("dp_test.e.out"):
    #                     os.system(f"apptainer exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m {dp_model} -d dp_test -n 0 > log.dp_test 2>&1")
    #                 else:
    #                     print("~~~ dp_test files already exist. Skipping. ~~~")
    #                 # except subprocess.CalledProcessError as e:
    #                 #     print(f"Error processing folder {deepmd_path}: {e}")
    #             else:
    #                     print(f"### Warning: {search_string} not found. Skipping. ###")
                
    #             print("")


# # Process both TRAIN and TEST directories
# if mode in [5]:

#     print(f"Running mode {mode} ...")
#     print("")

#     print("Processing TRAIN directory...")
#     print("")
#     search_and_copy_dirs_3(primary_base_dir, destination_dir_TRAIN, run_prefix="TRAIN")

#     print("Processing TEST directory...")
#     print("")
#     search_and_copy_dirs_3(primary_base_dir, destination_dir_TEST, run_prefix="TEST")

#     # print time taken
#     end_time1 = time.time()
#     print("Time taken: ", end_time1-start_time)




####################################################################################################################
####################################################################################################################
####################################################################################################################



def search_and_copy_dirs_3(destination_dir, run_prefix):

    runID_dir_basename_in_json = []
    # runID_dir_in_json = []

    # extract systems from training in myinput_json_file
    with open(myinput_json_file) as f:
        data = json.load(f)
        data_systems = data["training"]["training_data"]["systems"]

        # walk through data_sytems
        for directory in data_systems:
            deepmd_path = directory
            recal_path = os.path.dirname(deepmd_path)

            

            if ".recal" in os.path.basename(os.path.dirname(deepmd_path)):
                #split the deepmd_path before and beginning "u."
                runID_dir = (os.path.dirname(deepmd_path))
                # runID_dir_basename = os.path.basename(runID_dir)
                base_dir = os.path.dirname(runID_dir)
            # elif "a0" in os.path.basename(os.path.dirname(deepmd_path))":
            #     runID_dir = (os.path.dirname(os.path.dirname(os.path.dirname(deepmd_path))))
            else:
                runID_dir = (os.path.dirname(os.path.dirname(os.path.dirname(deepmd_path))))
                # divide the string runID_dir into two parts beginning v5_i and the rest
                # runID_dir_basename = 
                base_dir = os.path.dirname(runID_dir)


            deepmd_path_new = deepmd_path.replace(base_dir+"/", destination_dir)
            runID_dir_basename = runID_dir.replace(base_dir+"/", "")

            print(f"Processing {runID_dir}")
            # print(f"New deepmd path: {deepmd_path_new}")
            print("")

            # make an array of runID_dir_basename
            runID_dir_basename_in_json.append(runID_dir_basename)

    # create an array of all directory names in destination_dir
    runID_dir_basename_at_destination = [d for d in os.listdir(destination_dir) if os.path.isdir(os.path.join(destination_dir, d))]

    # Compare and print differences
    print("\nDirectories in JSON but not in destination:")
    missing_in_destination = list(set(runID_dir_basename_in_json) - set(runID_dir_basename_at_destination))
    for missing_dir in missing_in_destination:
        print(missing_dir)

    print("\nDirectories in destination but not in JSON:")
    extra_in_destination = list(set(runID_dir_basename_at_destination) - set(runID_dir_basename_in_json))
    for extra_dir in extra_in_destination:
        print(extra_dir)




# # Process both TRAIN and TEST directories
if mode in [5]:

    print(f"Running mode {mode} ...")
    print("")

    print("Processing TRAIN directory...")
    print("")
    search_and_copy_dirs_3(destination_dir_TRAIN, run_prefix="TRAIN")

    # print("Processing TEST directory...")
    # print("")
    # search_and_copy_dirs_3(destination_dir_TEST, run_prefix="TEST")

    # print time taken
    end_time1 = time.time()
    print("Time taken: ", end_time1-start_time)    


####################################################################################################################
####################################################################################################################
####################################################################################################################












def analyze_base_dir(base_dir, run_prefix):

    # File types and their headers
    files_to_read_for_analysis = {
        "all_dp_test.e_peratom.out": None,
        "all_dp_test.f.out": None,
        # "all_dp_test.v_peratom.out": None,
        # "all_dp_test.e_peratom.out": None,
        # "all_dp_test.v_peratom.out": None,
        # "all_dp_test.v_GPa.out": None,
        # "all_dp_test.v_GPa_peratom.out": None,
    }


    # read all_log.dp_test in dp_model_version_dir/run_prefix
    all_log_dp_test = os.path.join(dp_model_version_dir, run_prefix, "all_log.dp_test")
    if os.path.exists(all_log_dp_test):
        # read all parameters in all_log.dp_test in a pandas dataframe
        all_log_dp_test_df = pd.read_csv(all_log_dp_test)
        # ignore last row
        all_log_dp_test_df = all_log_dp_test_df[:-1]# last row is the "total" row

        # print(all_log_dp_test_df.head())
        print("Just read following columns from all_log.dp_test",all_log_dp_test_df.columns)
        # print(all_log_dp_test_df.shape)
        # print(all_log_dp_test_df.dtypes)
        # print(all_log_dp_test_df.describe())
        # print(all_log_dp_test_df.info())
        # print(all_log_dp_test_df["runID_dir"])



        # exit()

    # Process files in the deepmd folder
    for filename in files_to_read_for_analysis.keys():
        file_path = os.path.join(base_dir, filename)

        if os.path.exists(file_path):
            try:
                # Read the header (first line)
                with open(file_path, "r") as f:
                    header = f.readline().strip()
                    if files_to_read_for_analysis[filename] is None:
                        files_to_read_for_analysis[filename] = header  # Store the header once


                print("")
                print("")
                print("")
                print("##########################################")
                print("##########################################")
                print(f"Processing {filename} ...")
                print("##########################################")
                print("##########################################")

                # Read the file content excluding the header
                df = pd.read_csv(file_path, sep=r'\s+', header=None, skiprows=1)

                # # print first five rows of df
                # print(f"First five rows of {filename}:")
                # print(df.head())

                if filename == "all_dp_test.f.out":
                    file_code = "f"

                    # take first three columns as the x,y,z dimensions of the original values and the next three columns as the predicted values, and the take rmse of the two for each row
                    # rmse_values = rmse(df.iloc[:,0:3], df.iloc[:,3:6])
                    t_diff = df.iloc[:,0:3].values - df.iloc[:,3:6].values
                    t_sum = np.sum(t_diff**2, axis=1)
                    t_sqrt = np.sqrt(t_sum)
                    rmse_values = (t_sqrt)
                    data_magnitude = np.sqrt(np.sum(df.iloc[:,0:3].values**2, axis=1))
                    pred_magnitude = np.sqrt(np.sum(df.iloc[:,3:6].values**2, axis=1))
                # includes v
                elif "v" in filename:
                    file_code = "v_peratom"

                    # rmse_values = rmse(df.iloc[:,0:9], df.iloc[:,9:])
                    t_diff = df.iloc[:,0:9].values - df.iloc[:,9:].values
                    t_sum = np.sum(t_diff**2, axis=1)
                    t_sqrt = np.sqrt(t_sum)
                    rmse_values = (t_sqrt)
                    data_magnitude = np.sqrt(np.sum(df.iloc[:,0:9].values**2, axis=1))
                    pred_magnitude = np.sqrt(np.sum(df.iloc[:,9:].values**2, axis=1))
                else:
                    file_code = "e_peratom"

                    # rmse_values = rmse(df.iloc[:,0], df.iloc[:,1])
                    t_diff = df.iloc[:,0].values - df.iloc[:,1].values
                    t_sum = (t_diff**2)
                    t_sqrt = np.sqrt(t_sum)
                    rmse_values = (t_sqrt)
                    data_magnitude = np.abs(df.iloc[:,0].values)
                    pred_magnitude = np.abs(df.iloc[:,1].values)

                # make rmse_values a pandas series
                rmse_values = pd.Series(rmse_values)
                data_magnitude = pd.Series(data_magnitude)
                pred_magnitude = pd.Series(pred_magnitude)

                # print first five rows of rmse_values
                # print(f"First five rows of {filename}:")
                # print(rmse_values.head())

                # write the rmse_values to a new file with rmse appended to the filename
                output_file = os.path.join(base_dir, f"rmse.{filename}")
                with open(output_file, "w") as f:
                    f.write(header + " rmse\n")
                # Append the concatenated data
                rmse_values.to_csv(output_file, sep=" ", index=False, mode="a")

                print(f"Written: {output_file}")




                # plotting the histogram for visualization
                # plot_rmse_histogram(rmse_values, filename, base_dir)






                # calculate percentiles at 2 sigma and 3 sigma
                # percentile__A_sigma = [95]
                # percentile__B_sigma = [99.74]
                percentile__A_sigma_values = np.percentile(rmse_values, percentile__A_sigma)
                percentile__B_sigma_values = np.percentile(rmse_values, percentile__B_sigma)

                # convert percentile to float
                percentile__A_sigma_values = float(percentile__A_sigma_values)
                percentile__B_sigma_values = float(percentile__B_sigma_values)

                # shape of percentile__A_sigma_values and percentile__B_sigma_values
                print(f"A-sigma: {percentile__A_sigma}")
                print(f"B-sigma: {percentile__B_sigma}")
                print(f"Percentile A-sigma values: {percentile__A_sigma_values}")
                print(f"Percentile B-sigma values: {percentile__B_sigma_values}")
                # print(f"Shape of percentile 2-sigma values: {np.shape(percentile__A_sigma_values)}")
                # print(f"Shape of percentile 3-sigma values: {np.shape(percentile__B_sigma_values)}")

                # if file_code == "f":
                # all_log_dp_test_df has columns ['runID_dir', 'n_frames', 'n_atoms', 'Energy RMSE', 'Energy RMSE/Natoms','Force  RMSE', 'Virial RMSE', 'Virial RMSE/Natoms']
                # each (n_frames * n_atoms) entries in rmse_values correspond to one runID_dir. And all rmse_values are sorted in the same order as runID_dir in all_log_dp_test_df
                # first, check if the total number of entries in rmse_values is equal to the sum of n_frames*n_atoms in all_log_dp_test_df (added up for each runID_dir one by one)
                # then find the runIDs corresponding to indices__A_sigma and indices__B_sigma

                # Identify indices of RMSE values beyond thresholds
                indices__A_sigma = np.where((rmse_values > percentile__A_sigma_values))[0]
                indices__B_sigma = np.where((rmse_values > percentile__B_sigma_values))[0]

                # Initialize variables
                current_index = 0
                counter_processed = 0
                mapping__A_sigma = []
                mapping__B_sigma = []
                mapping__A_sigma_details = []
                mapping__B_sigma_details = []

                # Create a single set for fast membership testing
                indices__A_sigma_set = set(indices__A_sigma)
                #shape
                # print(f"Shape of indices__A_sigma_set: {len(indices__A_sigma_set)}")
                # print(f"Shape of rmse_values: {np.shape(rmse_values)}")

                indices__B_sigma_set = set(indices__B_sigma)

                if file_code == "f":

                    # Calculate segment sizes
                    segment_sizes = all_log_dp_test_df['n_frames'] * all_log_dp_test_df['n_atoms']

                    # Vectorized iteration through DataFrame
                    segment_end_indices = np.cumsum(segment_sizes)  # End indices
                    segment_start_indices = segment_end_indices - segment_sizes  # Start indices

                    # # print a table of segment_start_indices and segment_end_indices in two columns
                    # print(pd.DataFrame({'segment_start_indices': segment_start_indices, 'segment_end_indices': segment_end_indices, 'n_frames': all_log_dp_test_df['n_frames'], 'n_atoms': all_log_dp_test_df['n_atoms']}))
                    # # difference between (n+1)th and (n)th element of segment_start_indices and segment_end_indices respectively
                    # # new_segment_start_indices = np.insert(segment_start_indices, 0,0)
                    # new_segment_end_indices = np.insert(segment_end_indices,0,0)
                    # new_segment_start_indices = segment_start_indices#[:-1]
                    # new_segment_end_indices = new_segment_end_indices[:-1]
                    # diff_indices = new_segment_start_indices - new_segment_end_indices
                    # diff_indices_v2 = abs((all_log_dp_test_df['n_atoms'] * all_log_dp_test_df['n_frames'] - abs(segment_start_indices - segment_end_indices)))
                    # temp_pd = pd.DataFrame({'segment_start_indices': segment_start_indices, 'segment_end_indices': segment_end_indices, 'n_frames': all_log_dp_test_df['n_frames'], 'n_atoms': all_log_dp_test_df['n_atoms'], 'n_a * n_f': all_log_dp_test_df['n_atoms'] * all_log_dp_test_df['n_frames'], '|segment_s - segment_e|': abs(segment_start_indices - segment_end_indices), '|n_a*n_f - |segment_s - segment_e||': abs((all_log_dp_test_df['n_atoms'] * all_log_dp_test_df['n_frames'] - abs(segment_start_indices - segment_end_indices))), 'diff_indices': diff_indices})
                    # print(temp_pd)
                    # # write temp_pd to file
                    # temp_pd.to_csv("temp_pd.csv")
                    # exit()
                    # Validate total entries
                    # Calculate total_entries as the element-wise product of 'n_frames' and 'n_atoms', summed across all runID_dirs
                    total_entries = (all_log_dp_test_df['n_frames'] * all_log_dp_test_df['n_atoms']).sum()
                else:
                    # Calculate segment sizes
                    segment_sizes = all_log_dp_test_df['n_frames']# * all_log_dp_test_df['n_atoms']

                    # Vectorized iteration through DataFrame
                    segment_end_indices = np.cumsum(segment_sizes)  # End indices
                    segment_start_indices = segment_end_indices - segment_sizes  # Start indices

                    # print(pd.DataFrame({'segment_start_indices': segment_start_indices, 'segment_end_indices': segment_end_indices, 'n_frames': all_log_dp_test_df['n_frames'], 'n_atoms': all_log_dp_test_df['n_atoms']}))
                    # # difference between (n+1)th and (n)th element of segment_start_indices and segment_end_indices respectively
                    # # new_segment_start_indices = np.insert(segment_start_indices, 0,0)
                    # new_segment_end_indices = np.insert(segment_end_indices,0,0)
                    # new_segment_start_indices = segment_start_indices#[:-1]
                    # new_segment_end_indices = new_segment_end_indices[:-1]
                    # diff_indices = new_segment_start_indices - new_segment_end_indices
                    # temp_pd = pd.DataFrame({'segment_start_indices': segment_start_indices, 'segment_end_indices': segment_end_indices, 'n_frames': all_log_dp_test_df['n_frames'], 'n_atoms': all_log_dp_test_df['n_atoms'], 'diff_indices': diff_indices})
                    # print(temp_pd)
                    # # write temp_pd to file
                    # temp_pd.to_csv("temp_pd.e.csv")
                    # exit()
                    

                    # Validate total entries
                    # Calculate total_entries as the element-wise product of 'n_atoms', summed across all runID_dirs
                    total_entries = all_log_dp_test_df['n_frames'].sum()


                print(f"Total entries in rmse_values: {len(rmse_values)}")
                print(f"Total entries expected: {total_entries}")

                if len(rmse_values) != total_entries:
                    raise ValueError("Mismatch between rmse_values count and total n_frames * n_atoms in metadata!")

                for idx, (start, end, row) in enumerate(zip(segment_start_indices, segment_end_indices, all_log_dp_test_df.itertuples())):
                    counter_processed += 1
                    # print at every 1/10th of the total number of rows
                    if counter_processed % (len(all_log_dp_test_df) // 10) == 0:
                        print(f"Processing {counter_processed} of {len(all_log_dp_test_df)}")

                    # Efficient range generation for indices
                    segment_indices_set = set(range(start, end))
                    # shape
                    # print(f"Shape of segment_indices_set: {len(segment_indices_set)}")

                    # A-sigma
                    run_indices__A_sigma = list(segment_indices_set.intersection(indices__A_sigma_set))
                    num_run_indices__A_sigma = len(run_indices__A_sigma)
                    rmse_run_indices__A_sigma = rmse_values[run_indices__A_sigma] # rmse of the corresponding indices
                    mean_rmse_run_indices__A_sigma = np.mean(rmse_run_indices__A_sigma) if num_run_indices__A_sigma > 0 else 0.0
                    mapping__A_sigma.append({
                        'runID_dir': row.runID_dir,
                        'n_frames': row.n_frames,
                        'n_atoms': row.n_atoms,
                        'n_bad_indices': num_run_indices__A_sigma,
                        'percntg_bad_indices': 100*num_run_indices__A_sigma / (row.n_frames * row.n_atoms),
                        'mean_rmse_bad_indices': mean_rmse_run_indices__A_sigma,
                        'mean_rmse_all_indices': np.mean(rmse_values[start:end])
                    })

                    # Rescale run_indices__A_sigma
                    rescaled_run_indices__A_sigma = (np.array(run_indices__A_sigma) - start) + 1 # recal folders start from 1
                    if file_code == "f":
                        # Every n_atoms entries in rescaled_run_indices__A_sigma correspond to one frame
                        frame_numbers__A_sigma = np.ceil(rescaled_run_indices__A_sigma / row.n_atoms).astype(int)
                    else:
                        frame_numbers__A_sigma = rescaled_run_indices__A_sigma.astype(int)

                    # Unique frame numbers and their counts
                    unique_frame_numbers__A_sigma, counts = np.unique(frame_numbers__A_sigma, return_counts=True)

                    # rmse of the corresponding indices with unique_frame_numbers__A_sigma -- only the bad ones
                    rmse_unique_frame_numbers__A_sigma = np.array([np.mean(rmse_run_indices__A_sigma[frame_numbers__A_sigma == frame_number]) for frame_number in unique_frame_numbers__A_sigma])

                    # Create DataFrame and write to CSV
                    mapping__A_sigma_details_df = pd.DataFrame({
                        'runID_dir': row.runID_dir,
                        'frame_number': unique_frame_numbers__A_sigma,
                        'num_occurrences': counts,
                        'mean_bad_rmse_frame': rmse_unique_frame_numbers__A_sigma
                    })

                    # Prepare output directory
                    # in runID_dir, replace "/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOHN/deepmd_collection_" with "" in runID_dir, then choose everything after the first "/", and then replace all "/" with "_"
                    # modified_runID_dir = row.runID_dir.replace("/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOHN/deepmd_collection", "").split("/", 1)[1].replace("/", "_")
                    output_file = os.path.join(base_dir, f"mapping__A_sigma_details.{file_code}.csv")
                    # create a new file if counter_processed is 1, else append to the existing file
                    if counter_processed == 1:
                        mapping__A_sigma_details_df.to_csv(output_file, index=False)
                    else:
                        # add 2 empty lines
                        # with open(output_file, "a") as f:
                        #     f.write("\n\n")
                        mapping__A_sigma_details_df.to_csv(output_file, index=False, mode="a",header=False)
                    # os.makedirs(output_dir, exist_ok=True)
                    # output_path = os.path.join(output_dir, f"{modified_runID_dir}.{file_code}.csv")
                    # mapping__A_sigma_details_df.to_csv(output_path, index=False)


                    # B-sigma
                    run_indices__B_sigma = list(segment_indices_set.intersection(indices__B_sigma_set))
                    num_run_indices__B_sigma = len(run_indices__B_sigma)
                    rmse_run_indices__B_sigma = rmse_values[run_indices__B_sigma] # rmse of the corresponding indices
                    mean_rmse_run_indices__B_sigma = np.mean(rmse_run_indices__B_sigma) if num_run_indices__B_sigma > 0 else 0.0
                    mapping__B_sigma.append({
                        'runID_dir': row.runID_dir,
                        'n_frames': row.n_frames,
                        'n_atoms': row.n_atoms,
                        'n_bad_indices': num_run_indices__B_sigma,
                        'percntg_bad_indices': 100*num_run_indices__B_sigma / (row.n_frames * row.n_atoms),
                        'mean_rmse_bad_indices': mean_rmse_run_indices__B_sigma,
                        'mean_rmse_all_indices': np.mean(rmse_values[start:end])
                    })
                    # Rescale run_indices__B_sigma
                    rescaled_run_indices__B_sigma = (np.array(run_indices__B_sigma) - start) + 1

                    if file_code == "f":
                        # Every n_atoms entries in rescaled_run_indices__B_sigma correspond to one frame
                        frame_numbers__B_sigma = np.ceil(rescaled_run_indices__B_sigma / row.n_atoms).astype(int)
                    else:
                        frame_numbers__B_sigma = rescaled_run_indices__B_sigma.astype(int)

                    # Unique frame numbers and their counts
                    unique_frame_numbers__B_sigma, counts = np.unique(frame_numbers__B_sigma, return_counts=True)

                    # rmse of all indices with unique_frame_numbers__B_sigma -- only the bad ones
                    rmse_unique_frame_numbers__B_sigma = np.array([np.mean(rmse_run_indices__B_sigma[frame_numbers__B_sigma == frame_number]) for frame_number in unique_frame_numbers__B_sigma])

                    # Create DataFrame and write to CSV
                    mapping__B_sigma_details_df = pd.DataFrame({
                        'runID_dir': row.runID_dir,
                        'frame_number': unique_frame_numbers__B_sigma,
                        'num_occurrences': counts,
                        'mean_bad_rmse_frame': rmse_unique_frame_numbers__B_sigma
                    })
                    output_file = os.path.join(base_dir, f"mapping__B_sigma_details.{file_code}.csv")
                    # create a new file if counter_processed is 1, else append to the existing file
                    if counter_processed == 1:
                        mapping__B_sigma_details_df.to_csv(output_file, index=False)
                    else:
                        # add 2 empty lines
                        # with open(output_file, "a") as f:
                            # f.write("\n\n")
                        # append to the existing file without header
                        mapping__B_sigma_details_df.to_csv(output_file, index=False, mode="a",header=False)

                    # # output parth once
                    # if counter_processed == 1:
                    #     print(f"Output path for output_dir: {output_dir}")
                    #     print(f"Output path for mapping__A_sigma_details: {output_path}")
                    #     print(f"Output path for mapping__B_sigma_details: {output_path}")
                    #     exit()

                # Convert to DataFrames for easy viewing
                mapping__A_sigma_df = pd.DataFrame(mapping__A_sigma)
                mapping__B_sigma_df = pd.DataFrame(mapping__B_sigma)
                # mapping__A_sigma_details_df = pd.DataFrame(mapping__A_sigma_details)
                # mapping__B_sigma_details_df = pd.DataFrame(mapping__B_sigma_details)

                # print("A-Sigma Mapping:")
                # print(mapping__A_sigma_df)

                # print("\nB-Sigma Mapping:")
                # print(mapping__B_sigma_df)

                # Save the mapping to a CSV file
                mapping__A_sigma_df.to_csv(os.path.join(base_dir, f"mapping__A_sigma.{file_code}.csv"), index=False)
                mapping__B_sigma_df.to_csv(os.path.join(base_dir, f"mapping__B_sigma.{file_code}.csv"), index=False)

                # # Save the details to a CSV file
                # mapping__A_sigma_details_df.to_csv(os.path.join(base_dir, f"mapping__A_sigma_details.{file_code}.csv"), index=False)
                # mapping__B_sigma_details_df.to_csv(os.path.join(base_dir, f"mapping__B_sigma_details.{file_code}.csv"), index=False)



                # counter_processed += 1

            except Exception as e:
                if verbose:
                    print(f"~~~ Error processing file {file_path}: {e} ~~~")
        else:
            if verbose:
                print(f"Warning: {file_path} not found. Skipping.")




# Process both TRAIN and TEST directories
if mode in [6]:

    print(f"Running mode {mode} ...")

    print("Processing TRAIN directory...")
    analyze_base_dir(destination_dir_TRAIN, run_prefix="TRAIN")

    # print("Processing TEST directory...")
    # analyze_base_dir(destination_dir_TEST, run_prefix="TEST")

    update_dp_model_version_dir()

    # print time taken
    end_time1 = time.time()
    print("Time taken: ", end_time1-start_time)
















####################################################################################################################
os.chdir(starting_dir)
print("")
print("Done.")
print("")
####################################################################################################################