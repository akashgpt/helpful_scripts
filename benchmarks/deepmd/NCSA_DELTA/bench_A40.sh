#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=bench_A40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=0:10:00
#SBATCH --output=slurm-A40-%j.out

set -euo pipefail

export DP_INFER_BATCH_SIZE=32768
export APPTAINER_REPO="${APPTAINER_REPO:-/work/nvme/bguf/akashgpt/softwares/APPTAINER_REPO}"

if [[ -z "${APPTAINER_REPO:-}" ]]; then
    echo "APPTAINER_REPO is not set" >&2
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

srun apptainer exec --nv --bind "${bind_root}:${bind_root}" "${image}" dp train --skip-neighbor-stat myinput.json
