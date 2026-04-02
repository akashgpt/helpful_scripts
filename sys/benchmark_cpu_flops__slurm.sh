#!/bin/bash

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_PARTITION="cpu"
readonly DEFAULT_NODES=1
readonly DEFAULT_REPEATS=5
readonly DEFAULT_WARMUP_SECONDS=5
readonly DEFAULT_BENCHMARK_SECONDS=20
readonly DEFAULT_TIME_LIMIT="00:10:00"
readonly DEFAULT_JOB_NAME="cpu_flops_bench"

JOB_ID=""
TEMP_DIR=""
KEEP_ARTIFACTS=0

# select_shared_temp_root
#
# Chooses a shared writable directory for benchmark artifacts so both the login
# node and compute nodes can access the generated files.
#
# Args:
#   None.
# Returns:
#   Prints the selected shared directory path.
select_shared_temp_root() {
	local candidate_dir=""

	for candidate_dir in "${SCRATCH:-}" "${SCRATCH_2:-}" "$PWD"; do
		if [[ -n "${candidate_dir}" && -d "${candidate_dir}" && -w "${candidate_dir}" ]]; then
			printf "%s\n" "${candidate_dir}"
			return 0
		fi
	done

	echo "Error: could not find a shared writable directory for benchmark artifacts." >&2
	exit 1
}

# print_usage
#
# Prints command-line usage information.
#
# Args:
#   None.
# Returns:
#   None.
print_usage() {
	cat <<EOF
Usage:
	${SCRIPT_NAME} [options]

Description:
	Submits a Slurm job that reserves full CPU nodes, compiles a small OpenMP
	floating-point benchmark on the allocated node(s), runs it on all allocated
	cores, waits for completion, and prints average throughput per core and per
	node.

Options:
	--partition NAME         Slurm partition to use. Default: ${DEFAULT_PARTITION}
	--account NAME           Slurm account to charge.
	--nodes N                Number of nodes to benchmark. Default: ${DEFAULT_NODES}
	--repeats N              Number of timed repeats per node. Default: ${DEFAULT_REPEATS}
	--warmup-seconds N       Warm-up time per node in seconds. Default: ${DEFAULT_WARMUP_SECONDS}
	--benchmark-seconds N    Timed benchmark duration in seconds. Default: ${DEFAULT_BENCHMARK_SECONDS}
	--time LIMIT             Slurm wall-time limit. Default: ${DEFAULT_TIME_LIMIT}
	--job-name NAME          Slurm job name. Default: ${DEFAULT_JOB_NAME}
	--keep-artifacts         Keep the temporary benchmark directory even on success.
	--help                   Show this help message.

Examples:
	bash \$HELP_SCRIPTS/sys/${SCRIPT_NAME}
	bash \$HELP_SCRIPTS/sys/${SCRIPT_NAME} --partition cpu --account myacct
	bash \$HELP_SCRIPTS/sys/${SCRIPT_NAME} --nodes 2 --repeats 3 --benchmark-seconds 30
EOF
}

# require_command
#
# Verifies that a required command is available.
#
# Args:
#   command_name: Executable to check.
# Returns:
#   0 if available; exits otherwise.
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
#   candidate_value: Value to validate.
# Returns:
#   0 if the value is valid; 1 otherwise.
is_positive_integer() {
	local candidate_value="$1"

	[[ "$candidate_value" =~ ^[1-9][0-9]*$ ]]
}

# cleanup_temp_dir
#
# Removes the temporary benchmark directory when cleanup is allowed.
#
# Args:
#   exit_code: Script exit code.
# Returns:
#   None.
cleanup_temp_dir() {
	local exit_code="$1"

	if [[ "${KEEP_ARTIFACTS}" -eq 1 ]]; then
		return 0
	fi

	if [[ "${exit_code}" -ne 0 ]]; then
		echo "Preserving benchmark artifacts for debugging: ${TEMP_DIR}" >&2
		return 0
	fi

	if [[ -n "${TEMP_DIR}" && -d "${TEMP_DIR}" ]]; then
		rm -rf "${TEMP_DIR}"
	fi
}

# detect_cores_per_node
#
# Detects the CPU count advertised for the requested Slurm partition.
#
# Args:
#   partition_name: Slurm partition name.
# Returns:
#   Prints the detected core count.
detect_cores_per_node() {
	local partition_name="$1"
	local detected_cores=""

	detected_cores=$(sinfo -N -h -p "$partition_name" -o "%c" | awk '
		NF > 0 && $1 > max {
			max = $1
		}
		END {
			if (max > 0) {
				print max
			}
		}
	')

	if [[ -z "${detected_cores}" ]]; then
		echo "Error: could not detect CPU count for Slurm partition '${partition_name}'." >&2
		exit 1
	fi

	printf "%s\n" "${detected_cores}"
}

# print_submission_summary
#
# Prints a concise summary of the completed Slurm job.
#
# Args:
#   partition_name: Slurm partition used.
#   node_count: Number of nodes requested.
#   cores_per_node: Number of cores used on each node.
#   artifact_dir: Temporary benchmark directory.
# Returns:
#   None.
print_submission_summary() {
	local partition_name="$1"
	local node_count="$2"
	local cores_per_node="$3"
	local artifact_dir="$4"

	echo "Job ${JOB_ID} completed."
	echo "Partition: ${partition_name}"
	echo "Nodes: ${node_count}"
	echo "Cores used per node: ${cores_per_node}"
	echo "Artifacts: ${artifact_dir}"
}

# print_final_report
#
# Prints the completed benchmark summary and the Slurm output path.
#
# Args:
#   summary_file: Path to the generated summary file.
#   stdout_file: Path to the Slurm stdout file.
# Returns:
#   None.
print_final_report() {
	local summary_file="$1"
	local stdout_file="$2"

	if [[ ! -f "${summary_file}" ]]; then
		echo "Error: benchmark summary file was not created: ${summary_file}" >&2
		exit 1
	fi

	echo
	echo "Floating-point benchmark summary"
	echo "================================"
	cat "${summary_file}"
	echo
	echo "Slurm stdout: ${stdout_file}"
}

partition="${DEFAULT_PARTITION}"
account=""
nodes="${DEFAULT_NODES}"
repeats="${DEFAULT_REPEATS}"
warmup_seconds="${DEFAULT_WARMUP_SECONDS}"
benchmark_seconds="${DEFAULT_BENCHMARK_SECONDS}"
time_limit="${DEFAULT_TIME_LIMIT}"
job_name="${DEFAULT_JOB_NAME}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--partition)
			partition="$2"
			shift 2
			;;
		--account)
			account="$2"
			shift 2
			;;
		--nodes)
			nodes="$2"
			shift 2
			;;
		--repeats)
			repeats="$2"
			shift 2
			;;
		--warmup-seconds)
			warmup_seconds="$2"
			shift 2
			;;
		--benchmark-seconds)
			benchmark_seconds="$2"
			shift 2
			;;
		--time)
			time_limit="$2"
			shift 2
			;;
		--job-name)
			job_name="$2"
			shift 2
			;;
		--keep-artifacts)
			KEEP_ARTIFACTS=1
			shift
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

for numeric_value in "$nodes" "$repeats" "$warmup_seconds" "$benchmark_seconds"; do
	if ! is_positive_integer "$numeric_value"; then
		echo "Error: numeric options must be positive integers." >&2
		exit 1
	fi
done

require_command sbatch
require_command sinfo
require_command awk

shared_temp_root=$(select_shared_temp_root)
TEMP_DIR=$(mktemp -d "${shared_temp_root}/cpu_flops_bench_${USER}_XXXXXX")
trap 'cleanup_temp_dir $?' EXIT

source_path="${TEMP_DIR}/flops_kernel.c"
binary_path="${TEMP_DIR}/flops_kernel"
results_path="${TEMP_DIR}/raw_results.txt"
summary_path="${TEMP_DIR}/benchmark_summary.txt"
batch_script_path="${TEMP_DIR}/run_batch.sh"
stdout_path="${TEMP_DIR}/slurm-%j.out"

cores_per_node=$(detect_cores_per_node "$partition")

cat > "${source_path}" <<'EOF'
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

enum {
	WORKING_SET = 8192,
	FLOPS_PER_ELEMENT = 8
};

static double *allocate_aligned_buffer(const size_t element_count) {
	void *buffer = NULL;

	if (posix_memalign(&buffer, 64, element_count * sizeof(double)) != 0) {
		return NULL;
	}

	return (double *)buffer;
}

int main(int argc, char **argv) {
	const int warmup_seconds = (argc > 1) ? atoi(argv[1]) : 5;
	const int benchmark_seconds = (argc > 2) ? atoi(argv[2]) : 20;
	const int repeats = (argc > 3) ? atoi(argv[3]) : 5;
	const int thread_count = omp_get_max_threads();
	const size_t total_elements = (size_t)thread_count * (size_t)WORKING_SET;
	double *restrict a = NULL;
	double *restrict b = NULL;
	double *restrict c = NULL;
	char hostname[256] = {0};

	if (thread_count < 1 || warmup_seconds < 1 || benchmark_seconds < 1 || repeats < 1) {
		fprintf(stderr, "Invalid benchmark arguments.\n");
		return 1;
	}

	if (gethostname(hostname, sizeof(hostname) - 1) != 0) {
		snprintf(hostname, sizeof(hostname), "unknown-host");
	}

	a = allocate_aligned_buffer(total_elements);
	b = allocate_aligned_buffer(total_elements);
	c = allocate_aligned_buffer(total_elements);
	if (a == NULL || b == NULL || c == NULL) {
		fprintf(stderr, "Failed to allocate benchmark buffers.\n");
		free(a);
		free(b);
		free(c);
		return 1;
	}

	#pragma omp parallel
	{
		const int thread_id = omp_get_thread_num();
		const size_t start = (size_t)thread_id * (size_t)WORKING_SET;

		for (size_t index = 0; index < (size_t)WORKING_SET; ++index) {
			a[start + index] = 1.0 + 0.00001 * (double)(index + 1 + thread_id);
			b[start + index] = 2.0 + 0.00002 * (double)(index + 1 + thread_id);
			c[start + index] = 3.0 + 0.00003 * (double)(index + 1 + thread_id);
		}
	}

	for (int repeat_index = -1; repeat_index < repeats; ++repeat_index) {
		const double target_seconds = (repeat_index < 0) ? (double)warmup_seconds : (double)benchmark_seconds;
		double elapsed_seconds = 0.0;
		double start_time = omp_get_wtime();
		double repeat_checksum = 0.0;
		uint64_t iteration_count = 0;

		do {
			#pragma omp parallel reduction(+:repeat_checksum)
			{
				const int thread_id = omp_get_thread_num();
				const size_t start = (size_t)thread_id * (size_t)WORKING_SET;
				const double alpha = 1.000000119;
				const double beta = 0.999999881;
				const double gamma = 0.999999761;
				const double delta = 1.000000239;

				for (size_t index = 0; index < (size_t)WORKING_SET; ++index) {
					const size_t offset = start + index;
					const double next_a = alpha * a[offset] + beta * b[offset];
					const double next_b = gamma * b[offset] + delta * c[offset];
					c[offset] = next_a + next_b;
					a[offset] = next_a;
					b[offset] = next_b;
					repeat_checksum += c[offset];
				}
			}

			iteration_count += 1;
			elapsed_seconds = omp_get_wtime() - start_time;
		} while (elapsed_seconds < target_seconds);

		if (repeat_index >= 0) {
			const double total_flops =
				(double)iteration_count *
				(double)thread_count *
				(double)WORKING_SET *
				(double)FLOPS_PER_ELEMENT;
			const double node_gflops = total_flops / elapsed_seconds / 1.0e9;
			const double core_gflops = node_gflops / (double)thread_count;

			printf(
				"HOST %s REPEAT %d THREADS %d ELAPSED %.6f ITERATIONS %llu NODE_GFLOPS %.6f CORE_GFLOPS %.6f CHECKSUM %.6f\n",
				hostname,
				repeat_index + 1,
				thread_count,
				elapsed_seconds,
				(unsigned long long)iteration_count,
				node_gflops,
				core_gflops,
				repeat_checksum
			);
			fflush(stdout);
		}
	}

	free(a);
	free(b);
	free(c);
	return 0;
}
EOF

cat > "${batch_script_path}" <<EOF
#!/bin/bash

set -euo pipefail

module reset
module load PrgEnv-gnu

export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-\${SLURM_CPUS_ON_NODE:-1}}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

compiler=""
for candidate in cc gcc clang; do
	if command -v "\${candidate}" >/dev/null 2>&1; then
		compiler="\${candidate}"
		break
	fi
done

if [[ -z "\${compiler}" ]]; then
	echo "No C compiler found on compute node." >&2
	exit 1
fi

"\${compiler}" -O3 -fopenmp -ffast-math -o "${binary_path}" "${source_path}"

srun \
	--nodes="\${SLURM_JOB_NUM_NODES}" \
	--ntasks="\${SLURM_JOB_NUM_NODES}" \
	--ntasks-per-node=1 \
	--cpus-per-task="\${SLURM_CPUS_PER_TASK}" \
	--cpu-bind=cores \
	"${binary_path}" "${warmup_seconds}" "${benchmark_seconds}" "${repeats}" | tee "${results_path}"

{
	echo "Job ID: \${SLURM_JOB_ID}"
	echo "Partition: \${SLURM_JOB_PARTITION}"
	echo "Node list: \${SLURM_NODELIST}"
	echo "Node count: \${SLURM_JOB_NUM_NODES}"
	echo "Cores used per node: \${SLURM_CPUS_PER_TASK}"
	echo "Timed repeats per node: ${repeats}"
	echo "Warm-up seconds: ${warmup_seconds}"
	echo "Benchmark seconds per repeat: ${benchmark_seconds}"
	echo
	echo "Per-sample results:"
} > "${summary_path}"

awk '
	\$1 == "HOST" {
		host = \$2
		repeat_index = \$4
		thread_count = \$6
		elapsed_seconds = \$8
		iteration_count = \$10
		node_gflops = \$12
		core_gflops = \$14

		sample_count += 1
		total_node_gflops += node_gflops
		total_core_gflops += core_gflops
		total_threads += thread_count

		if (!(host in seen_host)) {
			seen_host[host] = 1
			host_order[++host_order_count] = host
		}

		host_sample_count[host] += 1
		host_node_gflops[host] += node_gflops
		host_core_gflops[host] += core_gflops

		printf("  %s repeat %s: %.3f GFLOP/s per node, %.3f GFLOP/s per core, %.3f s, %s iterations\n",
			host,
			repeat_index,
			node_gflops,
			core_gflops,
			elapsed_seconds,
			iteration_count
		)
	}
	END {
		if (sample_count == 0) {
			print "No benchmark samples were captured."
			exit 1
		}

		print ""
		print "Per-node averages:"
		for (index = 1; index <= host_order_count; ++index) {
			host = host_order[index]
			printf("  %s: %.3f GFLOP/s per node, %.3f GFLOP/s per core across %d repeats\n",
				host,
				host_node_gflops[host] / host_sample_count[host],
				host_core_gflops[host] / host_sample_count[host],
				host_sample_count[host]
			)
		}

		print ""
		printf("Cluster-wide average: %.3f GFLOP/s per node\n", total_node_gflops / sample_count)
		printf("Cluster-wide average: %.3f GFLOP/s per core\n", total_core_gflops / sample_count)
		printf("Average cores used per node: %.1f\n", total_threads / sample_count)
		printf("Samples collected: %d\n", sample_count)
	}
' "${results_path}" >> "${summary_path}"
EOF

chmod 700 "${batch_script_path}"

sbatch_args=(
	--parsable
	--wait
	--job-name "$job_name"
	--partition "$partition"
	--nodes "$nodes"
	--ntasks "$nodes"
	--ntasks-per-node 1
	--cpus-per-task "$cores_per_node"
	--exclusive
	--time "$time_limit"
	--output "$stdout_path"
)

if [[ -n "$account" ]]; then
	sbatch_args+=(--account "$account")
fi

echo "Submitting floating-point benchmark to partition '${partition}' using ${nodes} full node(s) and ${cores_per_node} core(s) per node."
submission_output=$(sbatch "${sbatch_args[@]}" "$batch_script_path")
JOB_ID="${submission_output%%;*}"

resolved_stdout_path="${stdout_path//%j/${JOB_ID}}"

print_submission_summary "$partition" "$nodes" "$cores_per_node" "$TEMP_DIR"
print_final_report "$summary_path" "$resolved_stdout_path"