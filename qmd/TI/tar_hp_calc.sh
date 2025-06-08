#!/usr/bin/env bash
# Finds all hp_calculations/ directories under the current directory,
# auto-detects compressor (zstd, pigz, or gzip),
# and creates one compressed tarball per hp_calculations/ folder in its parent directory.
# Usage:
#   chmod +x tar_hp_calc.sh
#   nohup $HELP_SCRIPTS_TI/tar_hp_calc.sh > log.tar_hp_calc 2>&1 &

set -euo pipefail

echo "Starting hp_calculations archiving process at $(date)"
echo "Current directory: $(pwd)"
echo ""

# 1) Detect available compressor and set extension
tools=(zstd pigz gzip)
for t in "${tools[@]}"; do
    if command -v "$t" >/dev/null 2>&1; then
        case "$t" in
            zstd)
                COMP="zstd -T0 -5"; EXT="tar.zst";;
            pigz)
                COMP="pigz -p$(nproc)"; EXT="tar.gz";;
            gzip)
                COMP="gzip";            EXT="tar.gz";;
        esac
        break
    fi
done
if [[ -z "${COMP:-}" ]]; then
    echo "Error: No compressor found (need zstd, pigz or gzip)." >&2
    exit 1
fi

echo "Using compressor: $COMP"
echo "Output extension: .$EXT"

# Export for parallel jobs
export COMP EXT

# 2) Find hp_calculations directories and compress in parallel
CONCURRENCY=4  # tweak this to available I/O / CPU

echo "Searching and archiving hp_calculations directories..."
find . -type d -name 'hp_calculations' -print0 \
    | parallel -0 -j "$CONCURRENCY" --env COMP,EXT --will-cite \
        'dir="{}"; \
        parent=$(dirname "${dir}"); \
        out="${parent}/hp_calculations.${EXT}"; \
        echo "Archiving ${dir} → ${out}"; \
        tar -C "${dir}" --use-compress-program="${COMP}" -cvf "${out}" .'

echo ""
echo "Archiving process completed at $(date)"
