#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=dp_test_dpa2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0:30:00

# Run `dp --pt test` on the three trained DPA-2 models and the se_e2_a PT
# baseline. Uses one of the training systems as a held-out test (5 frames)
# to settle whether the lcurve-reported rmse_f=3.63 is real.
#
# We use a system that the model has seen during training as the test
# probe — the question is not generalization, but: does the model produce
# physically reasonable forces at all? If rmse_f is still ~3.6 here, the
# training is broken (suspected: nsel too small).

set -e
module purge
module load PrgEnv-gnu
module load gcc-native/13.2
module load cray-mpich
module load cudatoolkit/25.3_12.8
module load fftw/3.3.10-gcc13.3.1
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate ALCHEMY_env__PT
export PYTHONNOUSERSITE=1

TBENCH=/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench
TESTSYS=/work/nvme/bguf/akashgpt/qmd_data/MgSiOH__R2SCAN/deepmd_collection_TRAIN/u.h.j.j.pro-l.liquid_vapor.water1.0g.merge.recal/deepmd
NFRAMES=10

OUT=/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/shared/dp_test_results.txt
: > "$OUT"
{
  echo "Test system: $TESTSYS"
  echo "N frames:    $NFRAMES"
  echo "Date:        $(date)"
  echo
} | tee -a "$OUT"

for v in variant_train_dpa2_PT variant_train_dpa2_PT_big variant_train_dpa2_PT_big_strip variant_train_se_e2_a_PT; do
  d="$TBENCH/$v"
  [ -d "$d" ] || { echo "MISSING: $d" | tee -a "$OUT"; continue; }
  echo "=== $v ===" | tee -a "$OUT"
  cd "$d"
  CKPT=$(awk '{print $NF; exit}' checkpoint 2>/dev/null | tr -d '"')
  [ -z "$CKPT" ] && CKPT=model.ckpt.pt
  echo "checkpoint: $CKPT" | tee -a "$OUT"
  dp --pt test -m "$CKPT" -s "$TESTSYS" -n "$NFRAMES" 2>&1 | tee "$d/dp_test.log" | grep -E "RMSE|rmse|Number of|MAE" | tee -a "$OUT"
  echo | tee -a "$OUT"
done

echo "=== Done [$(date)] ===" | tee -a "$OUT"
