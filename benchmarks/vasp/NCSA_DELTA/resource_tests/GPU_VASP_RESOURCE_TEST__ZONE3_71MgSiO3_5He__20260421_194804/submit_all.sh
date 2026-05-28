#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
(cd g01_r01_c01_kpar1_algoN && sbatch RUN_VASP_gpu.sh)
(cd g01_r01_c08_kpar1_algoN && sbatch RUN_VASP_gpu.sh)
(cd g02_r02_c01_kpar1_algoN && sbatch RUN_VASP_gpu.sh)
(cd g02_r02_c01_kpar2_algoN && sbatch RUN_VASP_gpu.sh)
(cd g02_r02_c08_kpar2_algoN && sbatch RUN_VASP_gpu.sh)
(cd g04_r04_c01_kpar1_algoN && sbatch RUN_VASP_gpu.sh)
(cd g04_r04_c01_kpar4_algoN && sbatch RUN_VASP_gpu.sh)
(cd g02_r02_c01_kpar2_algoExact && sbatch RUN_VASP_gpu.sh)
