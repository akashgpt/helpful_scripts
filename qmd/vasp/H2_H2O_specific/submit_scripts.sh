# for id_ = 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200
# run: cd *${id_}eV && sb RUN_VASP.sh && cd ..
# for id_ in 300 400 500 600 700 800 900 1000
# do
#   cd *${id_}eV
#   # sbatch RUN_VASP.sh
#   source data_4_analysis.sh
#   cd ..
# done

# for all directories in the current directory, run the following command: cd $dir && source data_4_analysis.sh && cd ..
# cp data_4_analysis.sh ..
# cd ..
# find . -type d | xargs -I {} cp data_4_analysis.sh {}

# for dir in */; do
#   # Check if the directory is not named "run_scripts/"
#   if [ "$dir" != "run_scripts/" ]; then
#     # Change to the directory, run the script, then return to the parent directory
#     cd "$dir" || continue
#     source data_4_analysis.sh
#     cd ..
#   fi
# done


# first SCAN run: cp *500eV/POTCAR . && mkdir PBEsol SCAN && mv *sol_* PBEsol && cd SCAN && cp -r ../PBEsol/*500eV . && cp ../../../submit_scripts.sh . && code */INCAR
# H_h runs then: cp *500eV/POTCAR . && mkdir -p PBEsol SCAN && mv *sol_* PBEsol && cd SCAN && cp -r ../../POTCAR_w_H/SCAN/*500eV . && cp ../../../submit_scripts.sh . && cp ../POTCAR */POTCAR && code */INCAR


# replace _H_ with _H_h_ in the name of folders in the current directory
# for f in *H*
# do
#   mv -- "$f" "${f//_H_/_H_h_}"
# done

# find all folders called "PBEsol" or "SCAN" withing the current directory and run the following commands: python $mldp/post_recal_rerun.py -ip all -v -ss $MY_MLMD_SCRIPTS/reference_input_files/VASP_inputs/MgSiOHN/shortQ_DELLA_RECAL/sub_vasp_xtra.sh > log.recal_test



# for Gibbs_calc dir

# cp data_4_analysis.sh ..
# cd ..
# find . -type d | xargs -I {} cp data_4_analysis.sh {}

# for dir in */; do
#   # Check if the directory is not named "run_scripts/"
#   if [ "$dir" != "VACFRepository/" ]; then
#     # Change to the directory, run the script, then return to the parent directory
#     cd "$dir" || continue
#     echo "Working on $dir"
#     source data_4_analysis.sh
#     cd ..
#   fi
# done



# for crystalline_or_not__mixture dir

for dir in */; do
  # Check if the directory is not named "run_scripts/"
  if [ "$dir" != "VACFRepository/" ]; then
    # Change to the directory, run the script, then return to the parent directory
    cd "$dir" || continue
    echo "Working on $dir"

    cp /scratch/gpfs/BURROWS/akashgpt/qmd_data/H2O_H2/sim_data_convergence/run_scripts/hist_calc.py .
    python hist_calc.py &

    cd ..

  fi
done