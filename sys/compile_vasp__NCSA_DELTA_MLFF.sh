#!/bin/bash
set -euo pipefail

# Build VASP 6.6.0 on NCSA Delta RH9 using the Delta-specific MLFF-aware
# makefile.include. This keeps the build steps explicit and reproducible.

readonly VASP_ROOT="/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0"
readonly DELTA_INCLUDE="${VASP_ROOT}/makefile.include__NCSA_DELTA_MLFF"
readonly ACTIVE_INCLUDE="${VASP_ROOT}/makefile.include"
readonly BUILD_PREFIX="build_delta_mlff_rh9_gnu"
readonly VASP_STD="${VASP_ROOT}/bin/vasp_std"
readonly VASP_STD_BACKUP="${VASP_ROOT}/bin/vasp_std__pre_NCSA_DELTA_MLFF"
readonly VASP_STD_DELTA="${VASP_ROOT}/bin/vasp_std__NCSA_DELTA_MLFF"

cd "${VASP_ROOT}"

module reset
module load PrgEnv-gnu cray-hdf5-parallel
ulimit -s unlimited

install -m 0644 "${DELTA_INCLUDE}" "${ACTIVE_INCLUDE}"

if [[ -f "${VASP_STD}" && ! -f "${VASP_STD_BACKUP}" ]]; then
    cp -p "${VASP_STD}" "${VASP_STD_BACKUP}"
fi

make PREFIX="${BUILD_PREFIX}" DEPS=1 -j 16 std
cp -p "${VASP_STD}" "${VASP_STD_DELTA}"
chmod 750 "${VASP_STD}" "${VASP_STD_DELTA}"

echo "Built ${VASP_STD_DELTA}"
