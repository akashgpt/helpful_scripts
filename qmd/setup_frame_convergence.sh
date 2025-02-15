#!/bin/bash

# To ultimately test the convergence wrt the number of frames needed for training so that you can reproduce the entire dataset
# This script will:
# 1. Divide the folders into train and test sets
# 2. Train the model
# 3. Freeze the model
# 4. Evaluate the model on the train and test sets
# 5. Write the results to log.dp_test.master and log.dp_test.csv
#
# Usage: setup_frame_convergence.sh <train_test_ratio>
# Example: setup_frame_convergence.sh 0.2
# This will use 20% of the folders for training and 80% for testing
#
# use plot_ttr_convergence.py to plot the convergence of the model with respect to the number of frames used for training

# use for ttr, i.e. train-test-ratio
train_test_ratio=$1 # e.g., setup_frame_convergence.sh 0.2, where 0.2 is the train-test-ratio

# error if no train_test_ratio
if [ -z "$train_test_ratio" ]; then
    echo "Please provide the test-to-train ratio as an argument."
    exit 1
fi

echo "Parent directory: $(pwd)"

mkdir -p set_train
mkdir -p set_test

rm -rf dp_test_* dp_test.* log* OUTCAR rerun deepmd

# count number of folders that start with a digit
n_folders=$(ls -d [0-9]* 2>/dev/null | wc -l)

# divide these folders into train and test sets
n_set_train=$(awk -v n="$n_folders" -v t="$train_test_ratio" 'BEGIN {print int(n * t / (t + 1) + 0.5)}')
n_set_test=$((n_folders - n_set_train))
echo "Number of folders: $n_folders"
echo "Number of folders in train set: $n_set_train"
echo "Number of folders in test set: $n_set_test"

# randomly select folders for train set
train_folders=($(ls -d [0-9]* 2>/dev/null | shuf | head -n $n_set_train))
for folder in "${train_folders[@]}"; do
    mv "$folder" set_train
    # echo "Moved to set_train: $folder"
done

# rest of the folders are for test set
test_folders=($(ls -d [0-9]* 2>/dev/null))
for folder in "${test_folders[@]}"; do
    mv "$folder" set_test
    # echo "Moved to set_test: $folder"
done


# starting training
cd set_train
module purge && l_deepmd_cpu && rm -rf OUTCAR deepmd && python $mldp/merge_out.py -o OUTCAR && python $mldp/extract_deepmd.py -f OUTCAR -d deepmd -ttr 100000
echo "Training deepmd set created"
cd ../set_test
module purge && l_deepmd_cpu && rm -rf OUTCAR deepmd && python $mldp/merge_out.py -o OUTCAR && python $mldp/extract_deepmd.py -f OUTCAR -d deepmd -ttr 100000
echo "Test deepmd set created"
echo ""

cd ..
rm -rf train
cp -r ../recal/train .
cd train
rm -f slurm*
sb train_1h.apptr.sh

# wait for "wall time" to appear in the slurm* file -- indicates the job finished
while [ -z "$(ls slurm* 2>/dev/null)" ]; do sleep 10; done  # Wait for file to appear
echo "Training started"
echo 
while ! grep -q "wall time:" slurm* 2>/dev/null; do sleep 10; done  # Wait for keyword
echo "Training finished"
echo ""

cd model-compression
rm -f slurm*
sb freeze_cpu_1h.apptr.sh

# wait for "final graph" to appear in the slurm* file -- indicates the job finished
while [ -z "$(ls slurm* 2>/dev/null)" ]; do sleep 10; done
# while ! grep -q "stage 2: freeze the model" slurm* 2>/dev/null; do sleep 10; done
# sleep 120  # wait for 2 minutes for the model to be frozen
# wait until "final graph." occurs twice in the slurm file
while [ $(grep -c "final graph." slurm*) -lt 2 ]; do sleep 10; done
echo "Model compression/freeze finished"
echo ""


cd ../.. # in the main directory


# evaluate test and train errors
cd set_train
rm -f vasp_vs_nn.png
apptainer exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m ../train/model-compression/pv_comp.pb -d dp_test -n 0 > log.dp_test 2>&1
module purge && l_deepmd_cpu && python $mldp/model_dev/analysis.py -tf . -mp dp_test -rf . -euc 10 -fuc 100 -flc 0.8 -elc 0.02
echo "Train error evaluation finished"
echo ""

cd ../set_test
# cd set_test && module purge && l_deepmd_cpu && rm -rf OUTCAR deepmd && python $mldp/merge_out.py -o OUTCAR && python $mldp/extract_deepmd.py -f OUTCAR -d deepmd -ttr 100000
rm -f vasp_vs_nn.png
apptainer exec $APPTAINER_REPO/deepmd-kit_latest.sif dp test -m ../train/model-compression/pv_comp.pb -d dp_test -n 0 > log.dp_test 2>&1
module purge && l_deepmd_cpu && python $mldp/model_dev/analysis.py -tf . -mp dp_test -rf . -euc 10 -fuc 100 -flc 0.8 -elc 0.02
echo "Test error evaluation finished"
echo ""

cd ..
# extract "Energy RMSE/Natoms", "Force  RMSE" and "Virial RMSE/Natoms" from the log files and print in log.dp_test.master
echo "Energy RMSE/Natoms, Force RMSE, and Virial RMSE/Natoms for train and test sets" > log.dp_test.master
echo "train_test_ratio: $train_test_ratio" >> log.dp_test.master
echo "" >> log.dp_test.master
echo "Train set ($n_set_train)" >> log.dp_test.master
echo "Energy RMSE/Natoms" >> log.dp_test.master
grep -m 1 "Energy RMSE/Natoms" set_train/log.dp_test | awk '{print $8}' >> log.dp_test.master
echo "Force RMSE" >> log.dp_test.master
grep -m 1 "Force  RMSE" set_train/log.dp_test | awk '{print $8}' >> log.dp_test.master
echo "Virial RMSE/Natoms" >> log.dp_test.master
grep -m 1 "Virial RMSE/Natoms" set_train/log.dp_test | awk '{print $8}' >> log.dp_test.master

echo "" >> log.dp_test.master
echo "Test set ($n_set_test)" >> log.dp_test.master
echo "Energy RMSE/Natoms" >> log.dp_test.master
grep -m 1 "Energy RMSE/Natoms" set_test/log.dp_test | awk '{print $8}' >> log.dp_test.master
echo "Force RMSE" >> log.dp_test.master
grep -m 1 "Force  RMSE" set_test/log.dp_test | awk '{print $8}' >> log.dp_test.master
echo "Virial RMSE/Natoms" >> log.dp_test.master
grep -m 1 "Virial RMSE/Natoms" set_test/log.dp_test | awk '{print $8}' >> log.dp_test.master

# write a csv file with the same information, with headers as "Set", "Energy RMSE/Natoms", "Force  RMSE" and "Virial RMSE/Natoms"; first row is for train set and second row is for test set
echo "Set,Energy RMSE/Natoms,Force  RMSE,Virial RMSE/Natoms" > log.dp_test.csv
echo "Train," >> log.dp_test.csv
grep -m 1 "Energy RMSE/Natoms" set_train/log.dp_test | awk '{print $8}' | tr '\n' ',' >> log.dp_test.csv
grep -m 1 "Force  RMSE" set_train/log.dp_test | awk '{print $8}' | tr '\n' ',' >> log.dp_test.csv
grep -m 1 "Virial RMSE/Natoms" set_train/log.dp_test | awk '{print $8}' >> log.dp_test.csv

echo "Test," >> log.dp_test.csv
grep -m 1 "Energy RMSE/Natoms" set_test/log.dp_test | awk '{print $8}' | tr '\n' ',' >> log.dp_test.csv
grep -m 1 "Force  RMSE" set_test/log.dp_test | awk '{print $8}' | tr '\n' ',' >> log.dp_test.csv
grep -m 1 "Virial RMSE/Natoms" set_test/log.dp_test | awk '{print $8}' >> log.dp_test.csv


echo "Results written to log.dp_test.master and log.dp_test.csv"