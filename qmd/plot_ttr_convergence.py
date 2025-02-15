# from all recal_*/log.dp_test.master folders, grab the number at the end of the last line with train_test_ratio, and plot it against line number 6 (which is the "Energy RMSE/Natoms")

import glob
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Find all log files
log_files = glob.glob("recal_*/log.dp_test.master")

# Lists to store extracted values
train_test_ratios = []
train_energy_rmse_values = []
test_energy_rmse_values = []
train_force_rmse_values = []
test_force_rmse_values = []

# Step 2: Extract required values from each file
for file in log_files:
    with open(file, "r") as f:
        lines = f.readlines()
    
    # Ensure the file has at least 6 lines
    if len(lines) < 6:
        print(f"Skipping {file}: Not enough lines")
        continue

    # Extract Energy RMSE/Natoms from line 6 (index 5)
    try:
        train_energy_rmse = float(lines[5].strip().split()[-1])
        test_energy_rmse = float(lines[13].strip().split()[-1])
        train_force_rmse = float(lines[7].strip().split()[-1])
        test_force_rmse = float(lines[15].strip().split()[-1])
    except ValueError:
        print(f"Skipping {file}: Could not extract Energy RMSE/Natoms")
        continue

    # Extract the last number from the last line containing "train_test_ratio"
    train_test_ratio = None
    for line in reversed(lines):  # Search from bottom up
        if "train_test_ratio" in line:
            try:
                train_test_ratio = float(line.strip().split()[-1])
            except ValueError:
                print(f"Skipping {file}: Could not extract train_test_ratio")
            break

    # Ensure both values were found before adding to lists
    if train_test_ratio is not None:
        train_test_ratios.append(train_test_ratio)
        train_energy_rmse_values.append(train_energy_rmse)
        test_energy_rmse_values.append(test_energy_rmse)
        train_force_rmse_values.append(train_force_rmse)
        test_force_rmse_values.append(test_force_rmse)

# Step 3: Plot the extracted values
if train_test_ratios and train_energy_rmse_values and test_energy_rmse_values:

    # sort the values
    train_test_ratios, train_energy_rmse_values, test_energy_rmse_values, train_force_rmse_values, test_force_rmse_values = zip(*sorted(zip(train_test_ratios, train_energy_rmse_values, test_energy_rmse_values, train_force_rmse_values, test_force_rmse_values)))

    # # Add a secondary x-axis to show n_set_train corresponding to train_test_ratio
    n_set_total = 500  # Total dataset size
    log_x_axis = False
    train_test_ratios_second_axis = [0.1, 0.3, 0.5, 0.7, 1]#,2]
    # train_test_ratios_second_axis = [0.1, 0.3, 0.5, 0.7, 1.0,2]
    x_axis_min = min(train_test_ratios)*0.9
    x_axis_max = np.min([max(train_test_ratios), max(train_test_ratios_second_axis)])*1.1

    # Compute n_set_train for each train_test_ratio
    n_set_train_values = np.array([n_set_total / (1 + 1 / ttr) if ttr != 0 else 0 for ttr in train_test_ratios_second_axis])
    n_set_train_percentage = 100*n_set_train_values/n_set_total
    # print(n_set_train_percentage)




    plt.figure(figsize=(8, 6))
    plt.scatter(train_test_ratios, train_energy_rmse_values, color='blue', alpha=0.5,s=50)
    plt.scatter(train_test_ratios, test_energy_rmse_values, color='red', alpha=0.5,s=50)
    plt.plot(train_test_ratios, train_energy_rmse_values, color='blue', alpha=0.5, ls='--')
    plt.plot(train_test_ratios, test_energy_rmse_values, color='red', alpha=0.5, ls='--')
    plt.xlabel("Train-Test Ratio")
    plt.ylabel("Energy RMSE/Natoms (eV; circles)")
    plt.grid(True)
    plt.legend(["Train", "Test"])
    plt.xticks(train_test_ratios_second_axis)
    plt.xlim(x_axis_min, x_axis_max)


    # plot force on the same graph, but right y-axis and as square markers with no fill
    plt.twinx()
    plt.scatter(train_test_ratios, train_force_rmse_values, color='blue', alpha=0.15,s=25, marker='s')#, facecolors='none')
    plt.scatter(train_test_ratios, test_force_rmse_values, color='red', alpha=0.15,s=25, marker='s')#, facecolors='none')
    plt.plot(train_test_ratios, train_force_rmse_values, color='blue', alpha=0.15, ls=':')
    plt.plot(train_test_ratios, test_force_rmse_values, color='red', alpha=0.15, ls=':')
    plt.ylabel("Force RMSE/Atom (eV/Angstrom; squares)")



    # x axis log scale
    if log_x_axis:
        plt.xscale('log')

    if not log_x_axis:
    # Create a secondary x-axis
        ax2 = plt.gca().twiny()

        # Ensure the secondary x-axis has the same limits
        ax2.set_xlim(x_axis_min, x_axis_max)
        ax2.set_ylim(plt.ylim())
        # print(plt.xlim())

        # Set ticks and corresponding labels
        ax2.set_xticks((train_test_ratios_second_axis))
        # ax2.set_xticklabels([f"{int(n_train)}" for n_train in n_set_train_values])
        ax2.set_xticklabels([f"{int(n_train)}" for n_train in n_set_train_percentage])

        # x axis log scale
        # plt.xscale('log')

        # # # Set label for the secondary x-axis
        # ax2.set_xlabel("Number of Training Samples")
        ax2.set_xlabel("Percentage of Total Samples used in Traning")


    plt.title("Energy | Force vs. Train-Test Ratio")

    # more space between title and plot
    plt.suptitle("")




    plt.show()
    plt.savefig("ttr_convergence.png")
else:
    print("No valid data found to plot.")