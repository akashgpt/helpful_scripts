#!/bin/bash

set -euo pipefail

DEFAULT_SIZE_MB=1024
DEFAULT_ATTEMPTS=3
TEMP_FILE_PREFIX=".disk_speed_test"
TEMP_FILES=()
KEEP_TEST_FILE=0

# print_usage
#
# Prints command-line usage information.
#
# Args:
#   None.
# Returns:
#   None.
print_usage() {
  cat <<'EOF'
Usage:
  test_disk_speed.sh [--size-mb SIZE_MB] [--attempts N] [--keep-file] PATH [PATH ...]

Description:
  Benchmarks rough sequential write and read throughput for one or more
  filesystem locations by creating a temporary test file in each target path.

Options:
  --size-mb SIZE_MB   Size of the temporary test file in MiB. Default: 1024
  --attempts N        Number of benchmark attempts per path. Default: 3
  --keep-file         Keep the temporary benchmark file instead of deleting it.
  --help              Show this help message.

Examples:
  bash $HELP_SCRIPTS/sys/test_disk_speed.sh /work/hdd/bguf/akashgpt
  bash $HELP_SCRIPTS/sys/test_disk_speed.sh --attempts 5 --size-mb 4096 /work/nvme/bguf/akashgpt /u/$USER
EOF
}

# require_command
#
# Verifies that a required command is available.
#
# Args:
#   command_name (string): Executable name to check.
# Returns:
#   0 if the command exists; exits with code 1 otherwise.
require_command() {
  local command_name="$1"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Error: required command '$command_name' was not found." >&2
    exit 1
  fi
}

# is_positive_integer
#
# Checks whether a value is a positive integer.
#
# Args:
#   candidate_value (string): Value to validate.
# Returns:
#   0 if the value is a positive integer; 1 otherwise.
is_positive_integer() {
  local candidate_value="$1"

  [[ "$candidate_value" =~ ^[1-9][0-9]*$ ]]
}

# cleanup_temp_files
#
# Removes temporary benchmark files unless the user asked to keep them.
#
# Args:
#   None.
# Returns:
#   None.
cleanup_temp_files() {
  local temp_file_path=""

  if [ "$KEEP_TEST_FILE" -eq 1 ]; then
    return 0
  fi

  for temp_file_path in "${TEMP_FILES[@]:-}"; do
    if [ -n "$temp_file_path" ] && [ -e "$temp_file_path" ]; then
      rm -f "$temp_file_path"
    fi
  done
}

# create_temp_file
#
# Creates an empty temporary file inside the requested directory.
#
# Args:
#   target_dir (string): Directory where the benchmark file will be created.
# Returns:
#   Prints the temporary file path on success.
create_temp_file() {
  local target_dir="$1"
  local temp_file_path=""

  temp_file_path=$(mktemp -p "$target_dir" "${TEMP_FILE_PREFIX}_${USER}_XXXXXX")
  TEMP_FILES+=("$temp_file_path")
  printf "%s\n" "$temp_file_path"
}

# print_filesystem_summary
#
# Prints a one-line filesystem summary for the requested path.
#
# Args:
#   target_path (string): Path whose filesystem should be described.
# Returns:
#   0 on success.
print_filesystem_summary() {
  local target_path="$1"

  df -hT "$target_path" | awk 'NR==1 || NR==2 {print}'
}

# measure_write_speed
#
# Measures sequential write throughput for a temporary benchmark file.
#
# Args:
#   target_file (string): File to write.
#   size_mb (int): Number of MiB to write.
# Returns:
#   Prints two tab-separated fields: mode and elapsed_seconds.
measure_write_speed() {
  local target_file="$1"
  local size_mb="$2"
  local elapsed_seconds=""
  local write_mode="direct"

  if elapsed_seconds=$({ /usr/bin/time -f '%e' dd if=/dev/zero of="$target_file" bs=1M count="$size_mb" oflag=direct status=none >/dev/null; } 2>&1); then
    :
  else
    write_mode="buffered+fdatasync"
    elapsed_seconds=$({ /usr/bin/time -f '%e' dd if=/dev/zero of="$target_file" bs=1M count="$size_mb" conv=fdatasync status=none >/dev/null; } 2>&1)
  fi

  printf "%s\t%s\n" "$write_mode" "$elapsed_seconds"
}

# measure_read_speed
#
# Measures sequential read throughput for a temporary benchmark file.
#
# Args:
#   target_file (string): File to read.
# Returns:
#   Prints two tab-separated fields: mode and elapsed_seconds.
measure_read_speed() {
  local target_file="$1"
  local elapsed_seconds=""
  local read_mode="direct"

  if elapsed_seconds=$({ /usr/bin/time -f '%e' dd if="$target_file" of=/dev/null bs=1M iflag=direct status=none; } 2>&1); then
    :
  else
    read_mode="buffered"
    elapsed_seconds=$({ /usr/bin/time -f '%e' dd if="$target_file" of=/dev/null bs=1M status=none; } 2>&1)
  fi

  printf "%s\t%s\n" "$read_mode" "$elapsed_seconds"
}

# print_speed_line
#
# Formats a measured throughput line in MiB/s.
#
# Args:
#   label (string): Human-readable label such as write or read.
#   size_mb (int): Number of MiB transferred.
#   mode (string): I/O mode used for the measurement.
#   elapsed_seconds (string): Elapsed wall time in seconds.
# Returns:
#   0 on success.
print_speed_line() {
  local label="$1"
  local size_mb="$2"
  local mode="$3"
  local elapsed_seconds="$4"

  awk -v label="$label" -v size_mb="$size_mb" -v mode="$mode" -v elapsed_seconds="$elapsed_seconds" '
    BEGIN {
      printf("%-6s %.1f MiB/s (%s, %s s)\n", label ":", size_mb / elapsed_seconds, mode, elapsed_seconds)
    }
  '
}

# summarize_modes
#
# Collapses repeated I/O mode labels into a compact comma-separated summary.
#
# Args:
#   mode_lines (string): Newline-separated list of mode labels.
# Returns:
#   Prints a comma-separated mode summary.
summarize_modes() {
  local mode_lines="$1"

  awk '
    NF > 0 && !seen[$0]++ {
      if (count > 0) {
        printf(",")
      }
      printf("%s", $0)
      count++
    }
  ' <<< "$mode_lines"
}

# calculate_speed_stats
#
# Computes mean and sample standard deviation for MiB/s across attempts.
#
# Args:
#   size_mb (int): Number of MiB transferred in each attempt.
#   elapsed_seconds_lines (string): Newline-separated elapsed times in seconds.
# Returns:
#   Prints two tab-separated fields: mean_mib_per_sec and std_mib_per_sec.
calculate_speed_stats() {
  local size_mb="$1"
  local elapsed_seconds_lines="$2"

  awk -v size_mb="$size_mb" '
    NF > 0 {
      speed = size_mb / $1
      count += 1
      sum += speed
      sumsq += speed * speed
    }
    END {
      if (count == 0) {
        printf("0.0\t0.0\n")
        exit 0
      }

      mean = sum / count
      if (count > 1) {
        variance = (sumsq - (sum * sum / count)) / (count - 1)
        if (variance < 0) {
          variance = 0
        }
        stddev = sqrt(variance)
      } else {
        stddev = 0
      }

      printf("%.1f\t%.1f\n", mean, stddev)
    }
  ' <<< "$elapsed_seconds_lines"
}

# print_summary_line
#
# Prints a summary line with mean and standard deviation for MiB/s.
#
# Args:
#   label (string): Human-readable label such as write or read.
#   mean_speed (string): Mean throughput in MiB/s.
#   stddev_speed (string): Sample standard deviation in MiB/s.
#   mode_summary (string): Comma-separated I/O mode summary.
# Returns:
#   0 on success.
print_summary_line() {
  local label="$1"
  local mean_speed="$2"
  local stddev_speed="$3"
  local mode_summary="$4"

  printf "%-12s mean=%s MiB/s, std=%s MiB/s (modes: %s)\n" "$label" "$mean_speed" "$stddev_speed" "$mode_summary"
}

# benchmark_path
#
# Runs a sequential write/read benchmark for one target directory.
#
# Args:
#   target_path (string): Directory to benchmark.
#   size_mb (int): Number of MiB to write and read.
#   attempts (int): Number of repeated benchmark attempts.
# Returns:
#   0 on success; 1 if the path is invalid or not writable.
benchmark_path() {
  local target_path="$1"
  local size_mb="$2"
  local attempts="$3"
  local resolved_path=""
  local temp_file_path=""
  local write_result=""
  local read_result=""
  local write_mode=""
  local write_seconds=""
  local read_mode=""
  local read_seconds=""
  local write_seconds_lines=""
  local read_seconds_lines=""
  local write_mode_lines=""
  local read_mode_lines=""
  local write_stats=""
  local read_stats=""
  local write_mean=""
  local write_stddev=""
  local read_mean=""
  local read_stddev=""
  local write_mode_summary=""
  local read_mode_summary=""
  local attempt_index=0

  if [ ! -d "$target_path" ]; then
    echo "Error: '$target_path' is not a directory." >&2
    return 1
  fi

  if [ ! -w "$target_path" ]; then
    echo "Error: '$target_path' is not writable." >&2
    return 1
  fi

  resolved_path=$(realpath "$target_path")
  temp_file_path=$(create_temp_file "$resolved_path")

  echo "============================================================"
  echo "Benchmark target: $resolved_path"
  echo "Test size: ${size_mb} MiB"
  echo "Attempts: ${attempts}"
  print_filesystem_summary "$resolved_path"

  for (( attempt_index = 1; attempt_index <= attempts; attempt_index++ )); do
    write_result=$(measure_write_speed "$temp_file_path" "$size_mb")
    write_mode=$(printf "%s" "$write_result" | cut -f1)
    write_seconds=$(printf "%s" "$write_result" | cut -f2)
    write_seconds_lines+="${write_seconds}"$'\n'
    write_mode_lines+="${write_mode}"$'\n'
    print_speed_line "write#${attempt_index}" "$size_mb" "$write_mode" "$write_seconds"

    read_result=$(measure_read_speed "$temp_file_path")
    read_mode=$(printf "%s" "$read_result" | cut -f1)
    read_seconds=$(printf "%s" "$read_result" | cut -f2)
    read_seconds_lines+="${read_seconds}"$'\n'
    read_mode_lines+="${read_mode}"$'\n'
    print_speed_line "read#${attempt_index}" "$size_mb" "$read_mode" "$read_seconds"
  done

  write_stats=$(calculate_speed_stats "$size_mb" "$write_seconds_lines")
  write_mean=$(printf "%s" "$write_stats" | cut -f1)
  write_stddev=$(printf "%s" "$write_stats" | cut -f2)
  write_mode_summary=$(summarize_modes "$write_mode_lines")

  read_stats=$(calculate_speed_stats "$size_mb" "$read_seconds_lines")
  read_mean=$(printf "%s" "$read_stats" | cut -f1)
  read_stddev=$(printf "%s" "$read_stats" | cut -f2)
  read_mode_summary=$(summarize_modes "$read_mode_lines")

  print_summary_line "write summary" "$write_mean" "$write_stddev" "$write_mode_summary"
  print_summary_line "read summary" "$read_mean" "$read_stddev" "$read_mode_summary"

  if [ "$KEEP_TEST_FILE" -eq 1 ]; then
    echo "Kept test file: $temp_file_path"
  else
    rm -f "$temp_file_path"
  fi

  echo ""
}

# main
#
# Parses command-line arguments and benchmarks each requested path.
#
# Args:
#   "$@" (string[]): Command-line arguments.
# Returns:
#   0 on success; exits with code 1 on invalid input.
main() {
  local size_mb="$DEFAULT_SIZE_MB"
  local attempts="$DEFAULT_ATTEMPTS"
  local target_paths=()

  require_command "awk"
  require_command "dd"
  require_command "df"
  require_command "mktemp"
  require_command "realpath"
  require_command "/usr/bin/time"

  while [ $# -gt 0 ]; do
    case "$1" in
      --size-mb)
        if [ $# -lt 2 ]; then
          echo "Error: --size-mb requires an integer value." >&2
          exit 1
        fi
        if ! is_positive_integer "$2"; then
          echo "Error: --size-mb expects a positive integer, got '$2'." >&2
          exit 1
        fi
        size_mb="$2"
        shift 2
        ;;
      --attempts)
        if [ $# -lt 2 ]; then
          echo "Error: --attempts requires an integer value." >&2
          exit 1
        fi
        if ! is_positive_integer "$2"; then
          echo "Error: --attempts expects a positive integer, got '$2'." >&2
          exit 1
        fi
        attempts="$2"
        shift 2
        ;;
      --keep-file)
        KEEP_TEST_FILE=1
        shift
        ;;
      --help|-h)
        print_usage
        exit 0
        ;;
      --*)
        echo "Error: unknown option '$1'." >&2
        exit 1
        ;;
      *)
        target_paths+=("$1")
        shift
        ;;
    esac
  done

  if [ "${#target_paths[@]}" -eq 0 ]; then
    print_usage
    exit 1
  fi

  trap cleanup_temp_files EXIT

  local target_path=""
  for target_path in "${target_paths[@]}"; do
    benchmark_path "$target_path" "$size_mb" "$attempts"
  done
}

main "$@"
