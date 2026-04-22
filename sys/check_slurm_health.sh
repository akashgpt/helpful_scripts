#!/bin/bash

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_TIMEOUT_SECONDS=12
readonly DEFAULT_OUTPUT_LINES=12

TIMEOUT_SECONDS="${DEFAULT_TIMEOUT_SECONDS}"
OUTPUT_LINES="${DEFAULT_OUTPUT_LINES}"
CHECK_USER="${USER:-}"

# print_usage
#
# Prints command-line usage information.
#
# Args:
# 	None.
# Returns:
# 	None.
print_usage() {
	cat <<EOF
Usage:
	${SCRIPT_NAME} [options]

Description:
	Quickly checks whether Slurm controller/accounting commands are returning
	promptly. This is useful for detecting scheduler/controller hangs such as:
	"slurm_load_jobs error: Unable to contact slurm controller".

Options:
	--timeout N       Seconds to wait for each Slurm command. Default: ${DEFAULT_TIMEOUT_SECONDS}
	--lines N         Max output lines to show per command. Default: ${DEFAULT_OUTPUT_LINES}
	--user NAME       User for squeue/sacct checks. Default: current user
	--help            Show this help message.

Examples:
	bash \$HELP_SCRIPTS/sys/${SCRIPT_NAME}
	bash \$HELP_SCRIPTS/sys/${SCRIPT_NAME} --timeout 8
	bash \$HELP_SCRIPTS/sys/${SCRIPT_NAME} --user "\$USER" --lines 25
EOF
}

# require_command
#
# Verifies that a required executable is available.
#
# Args:
# 	command_name: Executable to check.
# Returns:
# 	None. Exits if the executable is missing.
require_command() {
	local command_name="$1"

	if ! command -v "${command_name}" >/dev/null 2>&1; then
		echo "Error: required command '${command_name}' was not found." >&2
		exit 1
	fi
}

# is_positive_integer
#
# Checks whether a value is a positive integer.
#
# Args:
# 	candidate_value: Value to validate.
# Returns:
# 	0 if the value is a positive integer; 1 otherwise.
is_positive_integer() {
	local candidate_value="$1"

	[[ "${candidate_value}" =~ ^[1-9][0-9]*$ ]]
}

# parse_args
#
# Parses command-line arguments.
#
# Args:
# 	All command-line arguments.
# Returns:
# 	None.
parse_args() {
	while [[ "$#" -gt 0 ]]; do
		case "$1" in
			--timeout)
				TIMEOUT_SECONDS="$2"
				shift 2
				;;
			--lines)
				OUTPUT_LINES="$2"
				shift 2
				;;
			--user)
				CHECK_USER="$2"
				shift 2
				;;
			--help|-h)
				print_usage
				exit 0
				;;
			*)
				echo "Error: unknown argument '$1'." >&2
				print_usage >&2
				exit 1
				;;
		esac
	done

	if ! is_positive_integer "${TIMEOUT_SECONDS}"; then
		echo "Error: --timeout must be a positive integer." >&2
		exit 1
	fi

	if ! is_positive_integer "${OUTPUT_LINES}"; then
		echo "Error: --lines must be a positive integer." >&2
		exit 1
	fi

	if [[ -z "${CHECK_USER}" ]]; then
		echo "Error: could not determine user; pass --user NAME." >&2
		exit 1
	fi
}

# print_output_excerpt
#
# Prints a bounded excerpt from a command output file.
#
# Args:
# 	output_file: Path to a file containing command output.
# Returns:
# 	None.
print_output_excerpt() {
	local output_file="$1"

	if [[ ! -s "${output_file}" ]]; then
		echo "(no output)"
		return 0
	fi

	sed -n "1,${OUTPUT_LINES}p" "${output_file}"
}

# run_probe
#
# Runs one Slurm command with a timeout and prints whether it returned cleanly.
#
# Args:
# 	label: Human-readable probe name.
# 	command...: Command and arguments to run.
# Returns:
# 	0 if the probe succeeded; 1 otherwise.
run_probe() {
	local label="$1"
	shift
	local output_file=""
	local start_epoch=0
	local end_epoch=0
	local elapsed_seconds=0
	local exit_code=0

	output_file="$(mktemp "/tmp/${SCRIPT_NAME}.${label// /_}.XXXXXX")"
	start_epoch="$(date +%s)"

	set +e
	timeout "${TIMEOUT_SECONDS}s" "$@" >"${output_file}" 2>&1
	exit_code="$?"
	set -e

	end_epoch="$(date +%s)"
	elapsed_seconds=$(( end_epoch - start_epoch ))

	printf "\n[%s]\n" "${label}"
	printf "command: %s\n" "$*"

	if [[ "${exit_code}" -eq 0 ]]; then
		printf "status : OK (%ss)\n" "${elapsed_seconds}"
		print_output_excerpt "${output_file}"
		rm -f "${output_file}"
		return 0
	fi

	if [[ "${exit_code}" -eq 124 ]]; then
		printf "status : TIMEOUT after %ss\n" "${TIMEOUT_SECONDS}"
	else
		printf "status : FAILED exit=%s (%ss)\n" "${exit_code}" "${elapsed_seconds}"
	fi
	print_output_excerpt "${output_file}"
	rm -f "${output_file}"
	return 1
}

# main
#
# Entrypoint for the Slurm health check.
#
# Args:
# 	All command-line arguments.
# Returns:
# 	0 if all probes passed; 1 if any probe failed or timed out.
main() {
	local failed_count=0
	local today=""

	parse_args "$@"
	require_command timeout
	require_command squeue
	require_command sinfo
	require_command sacct
	require_command scontrol

	today="$(date +%F)"

	echo "Slurm health check"
	echo "time   : $(date '+%a %b %d %T %Z %Y')"
	echo "user   : ${CHECK_USER}"
	echo "timeout: ${TIMEOUT_SECONDS}s per probe"

	run_probe "scontrol ping" scontrol ping || failed_count=$(( failed_count + 1 ))
	run_probe "squeue jobs" squeue -u "${CHECK_USER}" -h -o "%i|%.80j|%T|%R" || failed_count=$(( failed_count + 1 ))
	run_probe "sinfo partitions" sinfo -h -o "%P|%a|%D|%t" || failed_count=$(( failed_count + 1 ))
	run_probe "sacct today" sacct -u "${CHECK_USER}" -S "${today}" --format=JobIDRaw,JobName,State -n -P || failed_count=$(( failed_count + 1 ))

	printf "\n[summary]\n"
	if [[ "${failed_count}" -eq 0 ]]; then
		echo "Slurm appears responsive."
		return 0
	fi

	echo "Slurm appears unhealthy: ${failed_count} probe(s) failed or timed out."
	return 1
}

main "$@"
