#!/bin/bash
set -euo pipefail
printf '%-34s %-8s %-12s %-12s %-12s %-12s\n' test status jobid oom eddiag max_mem_mib
for d in */; do
	[ -f "$d/sub_vasp_gpu.sh" ] || continue
	status=pending
	[ -f "$d/done_RUN_VASP" ] && status=done
	[ -f "$d/failed_RUN_VASP" ] && status=failed
	[ -f "$d/running_RUN_VASP" ] && status=running
	jobid=$(awk '/Job ID:/ {print $3; exit}' "$d/log.run_sim" 2>/dev/null || true)
	oom=$(grep -c 'CUDA_ERROR_OUT_OF_MEMORY\|cuMemAlloc returned error 2' "$d/log.run_sim" 2>/dev/null || true)
	eddiag=$(grep -c 'ERROR EDDIAG\|ZHEEV failed' "$d/log.run_sim" 2>/dev/null || true)
	maxmem=$(awk -F, '/^[0-9]+,/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); if ($4+0>m) m=$4+0} END {print m+0}' "$d/gpu_memory_trace.csv" 2>/dev/null || true)
	printf '%-34s %-8s %-12s %-12s %-12s %-12s\n' "${d%/}" "$status" "${jobid:-NA}" "$oom" "$eddiag" "${maxmem:-0}"
done
