#!/bin/bash
##############################################################################
# GPU-agnostic DeePMD training script for NCSA Delta
#
# Usage:  sbatch train_gpu_agnostic.apptr.sh [input.json]
#   - Defaults to myinput.json if no argument given
#   - Set GPU_PARTITION env var before sbatch to override (default: gpuA100x4)
#     e.g.: GPU_PARTITION=gpuH200x8 sbatch train_gpu_agnostic.apptr.sh
#
# Benchmark results (8000 batches, DeePMD-kit v3.1.3, se_e2_a, float64):
#   H200  (gpuH200x8):  0.0118 s/batch — 101 s total
#   A100  (gpuA100x4):  0.0242 s/batch — 200 s total
#   A40   (gpuA40x4):   0.127  s/batch — ~17 min projected
#
# Author: akashgpt
##############################################################################

#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=dp_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=0:30:00

set -euo pipefail

export DP_INFER_BATCH_SIZE=32768
export APPTAINER_REPO="${APPTAINER_REPO:-/work/nvme/bguf/akashgpt/softwares/APPTAINER_REPO}"

INPUT="${1:-myinput.json}"

if [[ ! -f "$INPUT" ]]; then
    echo "Input file not found: $INPUT" >&2
    exit 1
fi

image="${APPTAINER_REPO}/deepmd-kit_latest.sif"
if [[ ! -f "${image}" ]]; then
    echo "Apptainer image not found: ${image}" >&2
    exit 1
fi

bind_root=/work/nvme
if [[ ! -d "${bind_root}" ]]; then
    echo "Bind root not found: ${bind_root}" >&2
    exit 1
fi

srun apptainer exec --nv --bind "${bind_root}:${bind_root}" "${image}" dp train "$INPUT"
