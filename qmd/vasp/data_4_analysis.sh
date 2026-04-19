#!/bin/bash

#############################################################
# Summary:
#   Wrapper entrypoint for VASP analysis.
#   Auto-detects standard vs MLFF OUTCAR format and dispatches
#   to the matching implementation.
#
# Usage: source data_4_analysis.sh [OUTCAR]
#############################################################

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

resolve_analysis_script() {
	local script_name="$1"
	local candidate_path resolved_path

	candidate_path="$script_dir/$script_name"
	if [[ -f "$candidate_path" ]]; then
		printf '%s\n' "$candidate_path"
		return 0
	fi

	resolved_path=$(command -v "$script_name" 2>/dev/null || true)
	if [[ -n "$resolved_path" && -f "$resolved_path" ]]; then
		printf '%s\n' "$resolved_path"
		return 0
	fi

	return 1
}

run_data_4_analysis_autodetect() {
	local outcar_path="${1:-OUTCAR}"
	local analysis_mode target_script script_path

	if [[ ! -f "$outcar_path" ]]; then
		echo "Error: OUTCAR not found: $outcar_path"
		return 1
	fi

	if grep -aq "free  energy ML TOTEN\|MLFF:" "$outcar_path"; then
		analysis_mode="mlff"
		target_script="data_4_analysis__MLFF.sh"
	else
		analysis_mode="standard"
		target_script="data_4_analysis__standard.sh"
	fi

	if ! script_path=$(resolve_analysis_script "$target_script"); then
		echo "Error: could not locate $target_script beside this wrapper or on PATH"
		return 1
	fi

	echo "Auto-detected analysis mode: $analysis_mode"
	echo "Running $target_script using: $script_path"
	source "$script_path" "$outcar_path"
}

run_data_4_analysis_autodetect "$@"
return_code=$?
return "$return_code" 2>/dev/null || exit "$return_code"