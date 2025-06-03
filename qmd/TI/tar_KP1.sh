#!/usr/bin/env bash
# merge_and_compress_KP1.sh
# Finds all KP1/ directories under the current directory,
# auto-detects compressor (zstd, pigz, or gzip),
# and creates one compressed tarball per KP1/ folder in its parent directory.
# Usage:
#   chmod +x merge_and_compress_KP1.sh
#   ./merge_and_compress_KP1.sh

set -euo pipefail

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

# 2) Find KP1 directories and compress in parallel
CONCURRENCY=4  # tweak this to available I/O / CPU

echo "Searching for KP1 directories..."
find . -type d -name 'KP1' -print0 \
    | parallel -0 -j "$CONCURRENCY" --env COMP,EXT --will-cite \
        'dir="{}"; \
         parent=$(dirname "${dir}"); \
         out="${parent}/KP1.${EXT}"; \
         echo "Archiving ${dir} → ${out}"; \
         # exclude any existing archive file
         tar -C "${dir}" --use-compress-program="${COMP}" -cvf "${out}" .'

echo "All KP1 folders have been archived."
