#!/bin/bash

####################################################################################################
# Summary:
#   This script is meant to be sourced in the .bashrc file to set up useful aliases and environment variables.
#   It is meant to be used in the Princeton clusters (DELLA, TIGER, STELLAR)
#   and on NCSA_DELTA at NCSA.
#   The script sets up the following:
#   1. Detects the cluster name and sets the SCRATCH directory accordingly.
#   2. Sets up aliases for squeue commands.
#   3. Enables color support for ls and sets up useful ls/grep aliases.
#   4. Defines myfind function to find files within the current directory.
#   5. Defines hog and hog_gpu functions to check the top CPU/GPU users.
#   6. Sets up environment and conda aliases.
#   7. Exports useful folder paths.
#   8. Creates local copy of useful directories from the projects/ directory.
#   9. Runs rsync operations to update the local copy of the directories.
#   On NCSA_DELTA, PRIMARY_PROJECTS_FOLDER is expected to host active project content such as
#   misc_libraries and run_scripts/ALCHEMY__in_use, while other trees may still be optional.
#
# Usage: source myshortcuts.sh [verbose]
#   verbose: 0 (default) or 1
#   Example: source myshortcuts.sh 1
#
# Author: akashgpt
####################################################################################################


verbose=${1:-0}

# printf "Setting up myshortcuts.sh with verbose=$verbose\n"

# echo location of this file that is being sourced
# echo "Sourced from: ${BASH_SOURCE[0]}"

# if the location of this file is in scratch, then set verbose=0
if [[ ${BASH_SOURCE[0]} == *"scratch"* ]]; then
  verbose=0
fi


# Tag time when this script is sourced
echo "Sourcing myshortcuts.sh at $(date)"

##########################################
# BLOCK 1: Detect cluster name
##########################################
start_block1=$(date +%s)
# check name of current cluster
if [[ $(hostname) == *"della"* ]]; then
  export CLUSTER="DELLA"
elif [[ $(hostname) == *"tiger"* ]]; then
  export CLUSTER="TIGER"
elif [[ $(hostname) == *"stellar"* ]]; then
  export CLUSTER="STELLAR"
elif [[ $(hostname) == *"delta"* ]]; then
  export CLUSTER="NCSA_DELTA" # NCSA (via ACCESS)
fi

if [ "$CLUSTER" == "DELLA" ] || [ "$CLUSTER" == "TIGER" ] || [ "$CLUSTER" == "STELLAR" ]; then
  export SCRATCH="/scratch/gpfs/BURROWS/akashgpt"
elif [ "$CLUSTER" == "NCSA_DELTA" ]; then
  export SCRATCH="/work/nvme/bguf/akashgpt"
  export SCRATCH_2="/work/hdd/bguf/akashgpt"
else
  echo "WARNING: Cluster name detection failed. CLUSTER variable not set. SCRATCH variable not set."
fi

end_block1=$(date +%s)
elapsed_block1=$(( end_block1 - start_block1 ))
if [ $elapsed_block1 -gt 10 ]; then
  echo "WARNING: Cluster name detection (block 1) took $elapsed_block1 seconds!"
fi


##########################################
# SHARED HELPERS
##########################################

# append_to_env_var_if_dir
#
# Appends a directory to a colon-separated environment variable when the
# directory exists and is not already present.
#
# Args:
#   var_name: Name of the environment variable to update.
#   target_dir: Directory path to append.
# Returns:
#   0 if the variable is updated or left unchanged.
append_to_env_var_if_dir() {
  local var_name="$1"
  local target_dir="$2"
  local current_value=""

  if [ ! -d "$target_dir" ]; then
    return 0
  fi

  current_value="${!var_name}"

  case ":$current_value:" in
    *":$target_dir:"*)
      return 0
      ;;
  esac

  if [ -n "$current_value" ]; then
    export "$var_name=$current_value:$target_dir"
  else
    export "$var_name=$target_dir"
  fi
}

# ensure_local_copy_dir
#
# Creates a local mirror directory only when the matching source directory
# exists in the primary project tree.
#
# Args:
#   source_dir: Source directory to mirror.
#   target_dir: Local destination directory to create.
#   label: Human-readable label used in verbose messages.
# Returns:
#   0 if the directory is created or skipped safely.
ensure_local_copy_dir() {
  local source_dir="$1"
  local target_dir="$2"
  local label="$3"

  if [ -d "$source_dir" ]; then
    if ! mkdir -p "$target_dir" >/dev/null 2>&1; then
      if [ "$verbose" -eq 1 ]; then
        echo "Skipping local copy [$label]: unable to create $target_dir"
      fi
    fi
    return 0
  fi

  if [ "$verbose" -eq 1 ]; then
    echo "Skipping local copy [$label]: source not found at $source_dir"
  fi
}

# resolve_NCSA_DELTA_account
#
# Resolves the default NCSA_DELTA account name for CPU or GPU usage reports.
#
# Args:
#   requested_account: Explicit account passed by the user, if any.
#   resource_type: One of cpu or gpu.
# Returns:
#   0 and prints the chosen account when available; 1 otherwise.
resolve_NCSA_DELTA_account() {
  local requested_account="$1"
  local resource_type="$2"

  if [ -n "$requested_account" ]; then
    printf "%s\n" "$requested_account"
    return 0
  fi

  if [ "$resource_type" == "cpu" ] && [ -n "$NCSA_DELTA_CPU_ACCOUNT" ]; then
    printf "%s\n" "$NCSA_DELTA_CPU_ACCOUNT"
    return 0
  fi

  if [ "$resource_type" == "gpu" ] && [ -n "$NCSA_DELTA_GPU_ACCOUNT" ]; then
    printf "%s\n" "$NCSA_DELTA_GPU_ACCOUNT"
    return 0
  fi

  if [ -n "$NCSA_DELTA_ACCOUNT" ]; then
    printf "%s\n" "$NCSA_DELTA_ACCOUNT"
    return 0
  fi

  return 1
}

# run_hog_report
#
# Runs a Slurm usage summary with cluster-aware account defaults.
#
# Args:
#   days: Number of days to include.
#   requested_account: Explicit account filter, all, or empty for defaulting.
#   topcount: Number of rows to show.
#   tres_name: Slurm TRES name such as cpu or gres/gpu.
#   resource_type: One of cpu or gpu.
# Returns:
#   Exit status from sreport.
run_hog_report() {
  local days="${1:-30}"
  local requested_account="$2"
  local topcount="${3:-20}"
  local tres_name="$4"
  local resource_type="$5"
  local account=""
  local start_date=""

  start_date=$(date -d"${days} days ago" +%D)

  if [ "$CLUSTER" == "DELLA" ] || [ "$CLUSTER" == "TIGER" ] || [ "$CLUSTER" == "STELLAR" ]; then
    account="${requested_account:-astro}"
  elif [ "$CLUSTER" == "NCSA_DELTA" ]; then
    if resolve_NCSA_DELTA_account "$requested_account" "$resource_type" >/dev/null; then
      account=$(resolve_NCSA_DELTA_account "$requested_account" "$resource_type")
    else
      if [ -z "$requested_account" ]; then
        echo "WARNING: No NCSA_DELTA default account configured for ${resource_type} usage. Set NCSA_DELTA_${resource_type^^}_ACCOUNT or NCSA_DELTA_ACCOUNT, or pass an explicit account. Running unfiltered sreport."
      fi
      account=""
    fi
  else
    account="$requested_account"
  fi

  if [ "$account" == "all" ] || [ -z "$account" ]; then
    sreport user top start=${start_date} end=now TopCount=${topcount} -t hourper --tres="${tres_name}"
  else
    sreport user top start=${start_date} end=now TopCount=${topcount} accounts=${account} -t hourper --tres="${tres_name}"
  fi
}

# NCSA_DELTA_conda_init
#
# Bootstraps the shared NCSA Delta Miniforge install directly so that
# interactive activation works without a prior conda init.
#
# Args:
#   None.
# Returns:
#   0 if conda is ready to use; 1 otherwise.
NCSA_DELTA_conda_init() {
  local miniforge_root="/sw/rh9.4/python/miniforge3"
  local miniforge_bin="${miniforge_root}/bin"
  local conda_init_script="${miniforge_root}/etc/profile.d/conda.sh"

  if [ ! -d "$miniforge_bin" ]; then
    echo "ERROR: Shared NCSA_DELTA Miniforge install not found at $miniforge_root."
    return 1
  fi

  if [ -n "${CONDA_EXE:-}" ] && command -v conda >/dev/null 2>&1; then
    return 0
  fi

  case ":$PATH:" in
    *":$miniforge_bin:"*)
      ;;
    *)
      export PATH="$miniforge_bin:$PATH"
      ;;
  esac

  export CONDA_EXE="${miniforge_bin}/conda"
  export CONDA_PYTHON_EXE="${miniforge_bin}/python"

  if [ ! -f "$conda_init_script" ]; then
    echo "ERROR: Conda shell hook not found at $conda_init_script."
    return 1
  fi

  # shellcheck disable=SC1091
  . "$conda_init_script"

  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is still unavailable after sourcing $conda_init_script."
    return 1
  fi
}

##########################################
# BLOCK 2: squeue aliases
#
# List: sqp, sqpmy, sqpmy_eta, sq
#     jobpath (jobpath <jobid1> [jobid2 ...])
##########################################
start_block2=$(date +%s)
# if verbose; then
if [ $verbose -eq 1 ]; then
  echo "Setting up aliases at $(date) ..."
fi
# squeue
alias sqp='squeue -o "%.18i %Q %.9P %.9q %.8j %.8u %.10a %.2t %.10M %.10L %.6C %.12b %R"' #priority rating
# alias sqpmy='squeue -o "%.18i %Q %.9q %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | grep $USER' #priority rating
function sqpmy {
  printf "%12s %9s %9s %20s %8s %8s %2s %10s %10s %6s %6s %12s %s\n" \
    "JOBID" "PRIORITY" "QOS" "NAME" "USER" "ACCOUNT" "ST" "TIME" "TIME_LIMIT" "NODES" "CPUS" "GRES/GPUS" "REASON"
  # Running/suspended jobs: %S is the actual job start time
  # squeue -u "$USER" -h -t R,S -o "%.12i %.8Q %.9q %.20j %.8u %.8a %.2t %.10M %.10L %.6D %.6C %.25S %R" 2>/dev/null
  squeue -u "$USER" -h -t R,S -o "%.12i %.9Q %.9q %.20j %.8u %.8a %.2t %.10M %.10L %.6D %.6C %.12b %R" 2>/dev/null
  # Pending jobs: --start asks the scheduler for a backfill estimate of START_TIME
  # squeue --start -u "$USER" -h -o "%.12i %.8Q %.9q %.20j %.8u %.8a %.2t %.10M %.10L %.6D %.6C %.25S %R" 2>/dev/null
  squeue -u "$USER" -h -t PD   -o "%.12i %.9Q %.9q %.20j %.8u %.8a %.2t %.10M %.10L %.6D %.6C %.12b %R" 2>/dev/null
}

# Show the current user's jobs with expected start time and pending reason.
# Includes all sqpmy columns (PRIORITY, QOS, ACCOUNT) plus NODES, START_TIME.
function sqpmy_eta {
  printf "%12s %9s %9s %20s %8s %8s %2s %10s %10s %6s %6s %12s %20s %s\n" \
    "JOBID" "PRIORITY" "QOS" "NAME" "USER" "ACCOUNT" "ST" "TIME" "TIME_LIMIT" "NODES" "CPUS" "GRES/GPUS" "START_TIME" "REASON"
  # Running/suspended jobs: %S is the actual job start time
  squeue -u "$USER" -h -t R,S -o "%.12i %.9Q %.9q %.20j %.8u %.8a %.2t %.10M %.10L %.6D %.6C %.12b %.20S %R" 2>/dev/null
  # Pending jobs: %S shows backfill-estimated start time (N/A if not yet computed by scheduler)
  squeue -u "$USER" -h -t PD   -o "%.12i %.9Q %.9q %.20j %.8u %.8a %.2t %.10M %.10L %.6D %.6C %.12b %.20S %R" 2>/dev/null
}

alias sq='squeue -u $USER -o "%.18i %.9P %.12j %.8u %.2t %.10M %.6D %.8C %.10l"'

function busyness() {
  cpu_jobs_running=$(sqp | grep cpu | grep " R " | wc -l)
  gpu_jobs_running=$(sqp | grep gpu | grep " R " | wc -l)
  cpu_jobs_pending=$(sqp | grep cpu | grep " PD " | wc -l)
  gpu_jobs_pending=$(sqp | grep gpu | grep " PD " | wc -l)

  cpu_busyness_ratio=$(echo "scale=2; $cpu_jobs_running / ($cpu_jobs_pending)" | bc)
  gpu_busyness_ratio=$(echo "scale=2; $gpu_jobs_running / ($gpu_jobs_pending)" | bc)
  echo "CPU busyness: $cpu_busyness_ratio ($cpu_jobs_running running, $cpu_jobs_pending pending)"
  echo "GPU busyness: $gpu_busyness_ratio ($gpu_jobs_running running, $gpu_jobs_pending pending)"
}

# Show parent directories of given Slurm job IDs
jobpath() {
  if [ $# -eq 0 ]; then
    echo "Usage: jobpath <jobid1> [jobid2 ...]"
    return 1
  fi
  for id in "$@"; do
    workdir=$(scontrol show job "$id" 2>/dev/null | awk -F= '/WorkDir/{print $2}')
    if [ -n "$workdir" ]; then
      echo "$id → $workdir"
    else
      echo "$id → [Job not found or WorkDir unavailable]"
    fi
  done
}

end_block2=$(date +%s)
elapsed_block2=$(( end_block2 - start_block2 ))
if [ $elapsed_block2 -gt 10 ]; then
  echo "WARNING: Defining squeue aliases (block 2) took $elapsed_block2 seconds!"
fi






##########################################
# BLOCK 3: Color support & other ls/grep aliases
#
# List: ll, la, l, ls, lsf, llt, llf, lls, 
#       du, cq, 
#       mk, cl, .., ..., pwd
##########################################
start_block3=$(date +%s)
# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto -v'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

alias ll='ls -vclF -h'
alias la='ls -vA'
alias l='ls -vCF'
alias ls='ls --color=auto -v'
alias lsf='ls --color=auto -p -v' # only files + sort by time
alias llt='ls --color=auto -vhlctur' # sort by time
alias llf='ls --color=auto -hlctur -p -v' # only files + sort by time
alias lls='ls --color=auto -vlhcS' # sort by size
alias ll='ls --color=auto -lhc -v'
alias du='du -csh'

if [ "$CLUSTER" == "DELLA" ] || [ "$CLUSTER" == "TIGER" ] || [ "$CLUSTER" == "STELLAR" ]; then
  alias checkquota='checkquota'
elif [ "$CLUSTER" == "NCSA_DELTA" ]; then
  alias checkquota='quota'
fi

mk() { mkdir -p "$1" && cd "$1"; }
cl() { cd "$1" && ll; } # uses alias defined above
alias ..='cd .. && ll'
alias ...='cd ../.. && ll'
alias pwd='pwd -P'


end_block3=$(date +%s)
elapsed_block3=$(( end_block3 - start_block3 ))
if [ $elapsed_block3 -gt 10 ]; then
  echo "WARNING: Color/ls aliases block (block 3) took $elapsed_block3 seconds!"
fi


##########################################
# BLOCK 4: myfind & hog/hog_gpu functions
#
# List: myfind [directory] filename_pattern
#       hog [days] [account] [topcount]
#       hog_gpu [days] [account] [topcount]
#       hog_summary [days] [useraccount]
##########################################
start_block4=$(date +%s)

# to find files within the current directory; myfind 1061560[2-3] == myfind 10615602 && myfind 10615603
myfind() {
  if [ "$#" -eq 1 ]; then
    find . -iname \*"$1"\* 2>/dev/null
  else
    [ -d "$1" ] && find "$1" -iname \*"$2"\* 2>/dev/null
  fi
}

alias myfindscript='ps aux | grep'


hog() {
  local days="${1:-30}"
  local account="${2:-}"
  local topcount="${3:-20}"

  run_hog_report "$days" "$account" "$topcount" "cpu" "cpu"
}

hog_gpu() {
  local days="${1:-30}"
  local account="${2:-}"
  local topcount="${3:-20}"

  run_hog_report "$days" "$account" "$topcount" "gres/gpu" "gpu"
}


hog_summary() {
  # Default values
  local days=${1:-30}
  local useraccount=${2:-$USER}
  local start_date=""
  
  # Calculate the start date based on the provided number of days
  start_date=$(date -d"${days} days ago" +%D)

  printf "\n"
  echo "========================================================"
  echo "Usage Summary for $useraccount since $start_date (last $days days)"
  echo "========================================================"

  sacct -X -u $USER --starttime=$start_date --format=JobID%30,AllocCPUS,ElapsedRAW \
  | awk 'NR>2 {sec += $2*$3} END { printf "\nTotal CPU-hours: %.2f\n\n", sec/3600 }'

  sacct -X --starttime=$start_date -u $USER -o NNodes,ElapsedRaw -P \
  | awk -F'|' '{ sum += $1 * ($2/3600) } END { printf "Total node-hours: %.2f\n\n", sum }'
}


hog_OG() {
  start_date=$(date -d"30 days ago" +%D);
  account=$(sshare | grep $USER | awk '{print $1}' | head -n 1);
  sreport user top start=${start_date} end=now TopCount=100 accounts=${account} -t hourper --tres=cpu;
}


# Function to list and summarize all running non-SLURM scripts/programs for the current user
# Usage: myscripts [-v] [filter]
#   -v      : verbose, show full process list (default: summary only)
#   filter  : optional grep pattern to filter results (e.g. "CROSS_CLUSTER")
myscripts() {
  local verbose_flag=0
  local filter=""

  # Parse arguments
  for arg in "$@"; do
    if [ "$arg" == "-v" ]; then
      verbose_flag=1
    else
      filter="$arg"
    fi
  done

  # Get all user processes that are scripts/programs (exclude SLURM daemons, grep, and IDE internals)
  local proc_list
  proc_list=$(ps -u "$(whoami)" -o pid,ppid,etime,cmd --sort=cmd \
    | grep -E '\.sh\b|\.py\b|\.pl\b|\.R\b|\.jl\b' \
    | grep -v -E 'grep|shellIntegration|vscode')

  # Apply optional filter
  if [ -n "$filter" ]; then
    proc_list=$(echo "$proc_list" | grep "$filter")
  fi

  if [ -z "$proc_list" ]; then
    echo "No running scripts found."
    return 0
  fi

  local total
  total=$(echo "$proc_list" | wc -l)

  echo "====================================="
  echo " Running scripts for $(whoami): $total"
  echo "====================================="

  # Summary: count by script name
  echo ""
  echo "  Script                              Count"
  echo "  -----------------------------------+------"
  echo "$proc_list" \
    | grep -oP '[^\s/]+\.(sh|py|pl|R|jl)\b' \
    | sort | uniq -c | sort -rn \
    | awk '{printf "  %-37s %s\n", $2, $1}'

  # Verbose: full process list with parent directory (cwd)
  if [ $verbose_flag -eq 1 ]; then
    echo ""
    echo "  PID      PPID     ELAPSED   PARENT_DIR                                                CMD"
    echo "  ------   ------   --------  --------------------------------------------------------  ---"
    echo "$proc_list" | while read -r line; do
      local pid=$(echo "$line" | awk '{print $1}')
      local ppid=$(echo "$line" | awk '{print $2}')
      local etime=$(echo "$line" | awk '{print $3}')
      local cmd=$(echo "$line" | awk '{print substr($0, index($0,$4))}')
      local cwd=$(readlink -f /proc/"$pid"/cwd 2>/dev/null || echo "N/A")
      printf "  %-8s %-8s %-9s %-56s  %s\n" "$pid" "$ppid" "$etime" "$cwd" "$cmd"
    done
  fi

  echo ""
}




end_block4=$(date +%s)
elapsed_block4=$(( end_block4 - start_block4 ))
if [ $elapsed_block4 -gt 10 ]; then
  echo "WARNING: hog/hog_gpu definition block (block 4) took $elapsed_block4 seconds!"
fi



##########################################
# BLOCK 5: myjobs function
#
# List: myjobs [core_flag]
##########################################
start_block5=$(date +%s)
if [ "$CLUSTER" == "DELLA" ] || [ "$CLUSTER" == "TIGER" ] || [ "$CLUSTER" == "STELLAR" ]; then
  myjobs() {
    # core_flag=${1:-0}
    
    # Default values
    jobs_running=$(squeue -u $USER -h -t R -o "%.18i %.9P %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | wc -l)
    jobs_pending=$(squeue -u $USER -h -t PD -o "%.18i %.9P %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | wc -l)
    total_jobs=$(($jobs_running + $jobs_pending))

    echo "####################"
    echo "Jobs for user $USER"
    echo "####################"
    echo "Total jobs: $total_jobs"

    cores_list=$(sqp | grep ag5805 | grep " R " | awk '{print $10}')

    # Sum all instances of total_cores
    total_cores_sum=0
    for core in $cores_list; do
        total_cores_sum=$((total_cores_sum + core))
    done

    echo "Running jobs: $jobs_running, procs $total_cores_sum";
    echo "Pending jobs: $jobs_pending"
  }
  end_block5=$(date +%s)
  elapsed_block5=$(( end_block5 - start_block5 ))
  if [ $elapsed_block5 -gt 10 ]; then
    echo "WARNING: myjobs function block (block 5) took $elapsed_block5 seconds!"
  fi
elif [ "$CLUSTER" == "NCSA_DELTA" ]; then
  myjobs() {
  local jobs_running jobs_pending total_jobs total_cores_sum

  jobs_running=$(squeue -u "$USER" -h -t R | wc -l)
  jobs_pending=$(squeue -u "$USER" -h -t PD | wc -l)
  total_jobs=$((jobs_running + jobs_pending))
  total_cores_sum=$(squeue -u "$USER" -h -t R -o "%C" | awk '{sum += $1} END {print sum+0}')

  echo "####################"
  echo "Jobs for user $USER"
  echo "####################"
  echo "Total jobs: $total_jobs"
  echo "Running jobs: $jobs_running, procs $total_cores_sum"
  echo "Pending jobs: $jobs_pending"
}
fi


##########################################
# BLOCK 6: environment & conda aliases
#
# List: modl, sb, js, jspending
#        conda_a, conda_d, l_base, l_hpc, l_deepmd_gpu, l_deepmd_cpu, l_dp_plmd, l_mda, l_asap
#        git_merge_main, git_update_dev, git_update_main, git_last_update
##########################################
start_block6=$(date +%s)
if [ $verbose -eq 1 ]; then
  echo "Setting up environment and conda aliases at $(date) ..."
fi
# setting up environment in cluster
alias modl='module load'
alias sb='sbatch'

if [ "$CLUSTER" == "DELLA" ] || [ "$CLUSTER" == "TIGER" ] || [ "$CLUSTER" == "STELLAR" ]; then
  alias js='jobstats'
elif [ "$CLUSTER" == "NCSA_DELTA" ]; then
  js() { seff "$@"; }
fi
alias jspending='scontrol show job'

# Clear any stale conda aliases from an older sourced version of this file
# before defining the conda helpers below. This avoids interactive alias
# expansion breaking function definitions when .bashrc is sourced again.
unalias conda_a 2>/dev/null || true
unalias conda_d 2>/dev/null || true

# conda related alias
if [ "$CLUSTER" == "DELLA" ] || [ "$CLUSTER" == "TIGER" ] || [ "$CLUSTER" == "STELLAR" ]; then
  # conda_a
  #
  # Activates a conda environment on Princeton clusters.
  #
  # Args:
  #   env_name: Optional conda environment name. Defaults to conda's default activation.
  # Returns:
  #   Exit status from conda activate.
  conda_a() {
    if [ $# -eq 0 ]; then
      conda activate
    else
      conda activate "$@"
    fi
  }

  # conda_d
  #
  # Deactivates the current conda environment on Princeton clusters.
  #
  # Args:
  #   None.
  # Returns:
  #   Exit status from conda deactivate.
  conda_d() {
    conda deactivate
  }
  alias l_base='module load anaconda3/2025.12; conda activate base'
  alias l_hpc='module load anaconda3/2025.12; conda activate hpc-tools'
  # alias l_dpdev='module load anaconda3/2021.5; conda activate dpdev'
  alias l_planet_evo='module load anaconda3/2021.5; conda activate planet_evo'
  alias l_chhota_apple='module load anaconda3/2024.6; conda activate chhota_apple'
  # alias l_deepmd_cpu='module load anaconda3/2021.5; conda activate deepmd_cpu'

  alias l_deepmd_cpu='module load anaconda3/2025.12; conda activate deepmd_cpu' #DELLA/STELLAR; deepmd-kit 3.1.2 -- deepmd_cpu is different from deepmd-cpu

  if [[ $CLUSTER == "DELLA" ]]; then
    alias l_deepmd_gpu='module load anaconda3/2021.5; conda activate deepmd_gpu' #DELLA; deepmd-kit 2.1.1
    alias l_deepmd-cpu='module load anaconda3/2025.12; conda activate deepmd-cpu' #DELLA; deepmd-kit 2.2.10
  elif [[ $CLUSTER == "TIGER" ]]; then
    alias l_deepmd='module load anaconda3/2024.6; conda activate deepmd' #TIGER
    # alias l_deepmd_cpu='module load anaconda3/2024.6; conda activate deepmd_cpu' #TIGER
  elif [[ $CLUSTER == "STELLAR" ]]; then
    alias l_deepmd='module load anaconda3/2024.6; conda activate deepmd' #STELLAR
  fi

  # alias l_dp2='module load anaconda3/2021.5; conda activate dp2.2.7; export PLUMED_KERNEL=$CONDA_PREFIX/lib/libplumedKernel.so; LAMMPS_PLUGIN_PATH=$CONDA_PREFIX/lib/deepmd_lmp; patchelf --add-rpath $CONDA_PREFIX/lib dpplugin.so'
  alias l_mda='module load anaconda3/2025.12; conda activate mda_env' #TIGER | STELLAR | DELLA
  alias l_asap='module load anaconda3/2024.6; conda activate asap'
  alias l_pysr='module load anaconda3/2025.12; conda activate pysr_env' # TIGER

  if [[ $CLUSTER == "STELLAR" ]]; then
    alias l_qmda='module load anaconda3/2025.12; conda activate qmda' #STELLAR
    alias l_dp_plmd='module load anaconda3/2025.12; conda activate dp_plmd_stellar' #STELLAR | DELLA
  elif [[ $CLUSTER == "DELLA" ]]; then
    alias l_qmda='module load anaconda3/2025.12; conda activate qmda' #DELLA
    alias l_dp_plmd='module load anaconda3/2025.12; conda activate dp_plmd_della' #STELLAR | DELLA
  fi 
  # alias l_dp_plmd='module load anaconda3/2024.2; conda activate dp_plmd' #DELLA; deepmd-kit 2.2.12-dev
  # alias l_dpdev='module load anaconda3/2024.6; conda activate dpdev' #DELLA; deepmd-kit 2.2.12-dev

  alias l_ase='module load anaconda3/2025.12; conda activate ase_env'

elif [ "$CLUSTER" == "NCSA_DELTA" ]; then
  # conda_a
  #
  # Activates a NCSA_DELTA conda environment after loading Miniforge and the shell hook.
  #
  # Args:
  #   env_name: Optional conda environment name. Defaults to conda's default activation.
  # Returns:
  #   Exit status from conda activate.
  conda_a() {
    NCSA_DELTA_conda_init || return 1

    if [ $# -eq 0 ]; then
      conda activate
    else
      conda activate "$@"
    fi
  }
  alias l_base='conda_a base'
  alias l_dp_plmd='conda_a dp_plmd_ncsa_delta'
  alias l_hpc='conda_a hpc-tools'

  # conda_d
  #
  # Deactivates the current NCSA_DELTA conda environment after initializing the shell hook.
  #
  # Args:
  #   None.
  # Returns:
  #   Exit status from conda deactivate.
  conda_d() {
    NCSA_DELTA_conda_init || return 1
    conda deactivate
  }
fi

# git aliases
alias git_merge_main="git switch main; git merge dev; git push origin main; git switch dev"

git_update_dev() {
  date_time=$(date +"%Y-%m-%d %T")
  # Default commit message with current date and time if no argument is provided
  message_commit=${1:-"update: $date_time"}

  git switch dev
  git add .
  git commit -m "$message_commit"
  git push origin dev
}

git_update_main() {
  date_time=$(date +"%Y-%m-%d %T")
  # Default commit message with current date and time if no argument is provided
  message_commit=${1:-"update: $date_time"}

  git switch main
  git add .
  git commit -m "$message_commit"
  git push origin main
}

git_last_update() {
  local target_path repo_root abs_target rel_target current_folder created_line updated_line repo_name repo_display_name origin_url
  target_path="${1:-.}"

  if ! abs_target=$(realpath "$target_path" 2>/dev/null); then
    echo "Error: path not found: $target_path"
    return 1
  fi

  if ! repo_root=$(git -C "$abs_target" rev-parse --show-toplevel 2>/dev/null); then
    echo "Error: not inside a git repository: $abs_target"
    return 1
  fi

  rel_target=$(realpath --relative-to="$repo_root" "$abs_target")
  current_folder=$(basename "$abs_target")

  origin_url=$(git -C "$repo_root" remote get-url origin 2>/dev/null)
  if [ -n "$origin_url" ]; then
    repo_name=$(basename "$origin_url")
    repo_display_name="${repo_name%.git}"
  else
    repo_name=$(basename "$repo_root")
    repo_display_name="$repo_name"
  fi

  updated_line=$(git -C "$repo_root" log -1 --date=iso --pretty=format:'%ad  %s' -- "$rel_target")
  created_line=$(git -C "$repo_root" log --reverse --date=iso --pretty=format:'%ad  %s' -- "$rel_target" | head -n 1)

  if [ -z "$updated_line" ]; then
    echo "Error: no git history found for: $rel_target"
    return 1
  fi

  if [ -z "$created_line" ]; then
    created_line="$updated_line"
  fi

  echo ""
  if [ -n "$origin_url" ]; then
    repo_name=$(basename "$origin_url")
    repo_display_name="${repo_name%.git}"
    echo "Parent Git Repo: $repo_display_name"
  else
    repo_name=$(basename "$repo_root")
    repo_display_name="$repo_name"
    echo "Parent Git Repo: $repo_display_name (repo dir name; no origin URL found)"
  fi
  # echo "Parent Git Repo: $repo_display_name"
  echo "Git Folder Root: $repo_root"
  echo "Current Folder: $rel_target"
  echo "Created: $created_line"
  echo "Updated: $updated_line"
  echo ""
}

end_block6=$(date +%s)
elapsed_block6=$(( end_block6 - start_block6 ))
if [ $elapsed_block6 -gt 10 ]; then
  echo "WARNING: environment/conda aliases (block 6) took $elapsed_block6 seconds!"
fi



##########################################
# BLOCK 7: Exporting folder paths
##########################################
start_block7=$(date +%s)
if [ $verbose -eq 1 ]; then
  echo "Exporting folder paths at $(date) ..."
fi
# Defining useful folder paths
export AG_BURROWS="/projects/BURROWS/akashgpt"
export LOCAL_AG_BURROWS="$SCRATCH/local_copy__projects/BURROWS/akashgpt"
export AG_JIEDENG="/projects/JIEDENG/akashgpt"
export AG_bguf="/projects/bguf/akashgpt"
export LOCAL_AG_bguf="$SCRATCH/local_copy__projects/bguf/akashgpt"
export AG_TIGERDATA="/tigerdata/burrows/planet_evo/akashgpt"
export AG_TIGERDATA_2="/tigerdata/jiedeng/exoplanet/akashgpt"

if [ "$CLUSTER" == "DELLA" ] || [ "$CLUSTER" == "TIGER" ] || [ "$CLUSTER" == "STELLAR" ]; then
  export PRIMARY_PROJECTS_FOLDER=$AG_BURROWS
  export LOCAL_PRIMARY_PROJECTS_FOLDER=$LOCAL_AG_BURROWS
elif [ "$CLUSTER" == "NCSA_DELTA" ]; then
  export PRIMARY_PROJECTS_FOLDER=$AG_bguf
  export LOCAL_PRIMARY_PROJECTS_FOLDER=$LOCAL_AG_bguf
fi

export BACKUP_DIR="$SCRATCH/akashgpt_ucla_desktop_backup_20231231"
export VASP_ANALYSIS="$AG_BACKUP/Academics/Research/VASP/analysis_codes"
export VASP_DATA="$SCRATCH/qmd_data"
export mldp="$PRIMARY_PROJECTS_FOLDER/misc_libraries/scripts_Jie/mldp"
export LOCAL_mldp="$LOCAL_PRIMARY_PROJECTS_FOLDER/misc_libraries/scripts_Jie/mldp"
export JIE_SCRIPTS_DIR="$PRIMARY_PROJECTS_FOLDER/misc_libraries/scripts_Jie"
export LOCAL_JIE_SCRIPTS_DIR="$LOCAL_PRIMARY_PROJECTS_FOLDER/misc_libraries/scripts_Jie"
export LARS_SCRIPTS_DIR="$PRIMARY_PROJECTS_FOLDER/misc_libraries/Box_Lars"
export LOCAL_LARS_SCRIPTS_DIR="$LOCAL_PRIMARY_PROJECTS_FOLDER/misc_libraries/Box_Lars"
export MY_MLMD_SCRIPTS="$PRIMARY_PROJECTS_FOLDER/run_scripts/MLMD_scripts"
export APPTAINER_REPO="$SCRATCH/softwares/APPTAINER_REPO"
export ALCHEMY__dev="$PRIMARY_PROJECTS_FOLDER/run_scripts/ALCHEMY__dev"
export ALCHEMY__dev__HEALTH="$PRIMARY_PROJECTS_FOLDER/run_scripts/ALCHEMY__dev/TRAIN_MLMD_scripts/ANALYSIS/HEALTH_CHECK_UP"
export ALCHEMY__dev__MLDP="$PRIMARY_PROJECTS_FOLDER/run_scripts/ALCHEMY__dev/TRAIN_MLMD_scripts/ANALYSIS/mldp"
export LOCAL__ALCHEMY__main="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/ALCHEMY__in_use"
export ALCHEMY__main="$PRIMARY_PROJECTS_FOLDER/run_scripts/ALCHEMY__in_use"
export ALCHEMY__main__HEALTH="$PRIMARY_PROJECTS_FOLDER/run_scripts/ALCHEMY__in_use/TRAIN_MLMD_scripts/ANALYSIS/HEALTH_CHECK_UP"
export ALCHEMY__main__MLDP="$PRIMARY_PROJECTS_FOLDER/run_scripts/ALCHEMY__in_use/TRAIN_MLMD_scripts/ANALYSIS/mldp"
export LOCAL__ALCHEMY__main__HEALTH="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/ALCHEMY__in_use/TRAIN_MLMD_scripts/ANALYSIS/HEALTH_CHECK_UP"
export LOCAL__ALCHEMY__main__MLDP="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/ALCHEMY__in_use/TRAIN_MLMD_scripts/ANALYSIS/mldp"
export PLANET_EVO__main="$PRIMARY_PROJECTS_FOLDER/run_scripts/planet_evo_x_qmd"
export CONDA_SECONDARY_DIR="$SCRATCH/softwares/conda_envs_dir_secondary"

export HELP_SCRIPTS="$PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts"
export LOCAL_HELP_SCRIPTS="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts"
export HELP_SCRIPTS_qmd="$PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd"
export LOCAL_HELP_SCRIPTS_qmd="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd"
export HELP_SCRIPTS_vasp="$PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd/vasp"
export LOCAL_HELP_SCRIPTS_vasp="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd/vasp"
export HELP_SCRIPTS_plmd="$PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd/plmd"
export LOCAL_HELP_SCRIPTS_plmd="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd/plmd"
export HELP_SCRIPTS_TI="$PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd/TI"
export LOCAL_HELP_SCRIPTS_TI="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd/TI"
export HELP_SCRIPTS_ALCHEMY="$PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd/ALCHEMY"
export LOCAL_HELP_SCRIPTS_ALCHEMY="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/qmd/ALCHEMY"
export HELP_SCRIPTS_general="$PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/general"
export LOCAL_HELP_SCRIPTS_general="$LOCAL_PRIMARY_PROJECTS_FOLDER/run_scripts/helpful_scripts/general"
export HELP_SCRIPTS_BENCHMARKS="$HELP_SCRIPTS/benchmarks"



# Some ALCHEMY related aliases
alias ALCHEMY_status='$ALCHEMY__main__HEALTH/ALCHEMY_status.sh'
alias ALCHEMY_timing='$ALCHEMY__main__HEALTH/ALCHEMY_timing.sh'
alias ALCHEMY_performance='python $ALCHEMY__main__HEALTH/ALCHEMY_performance.py'




# export MDNN="/scratch/gpfs/ag5805/qmd_data/NH3_MgSiO3/sim_data_ML"

##########
# Adding to different default paths
##########
append_to_env_var_if_dir "PATH" "$LOCAL_PRIMARY_PROJECTS_FOLDER/misc_libraries"
append_to_env_var_if_dir "PATH" "$LOCAL_LARS_SCRIPTS_DIR"
append_to_env_var_if_dir "PATH" "$LOCAL_JIE_SCRIPTS_DIR"
append_to_env_var_if_dir "PATH" "$LOCAL_mldp"
append_to_env_var_if_dir "PATH" "$HOME/local/bin" # for patchelf


append_to_env_var_if_dir "PYTHONPATH" "$LOCAL_PRIMARY_PROJECTS_FOLDER/misc_libraries"
append_to_env_var_if_dir "PYTHONPATH" "$HELP_SCRIPTS"
append_to_env_var_if_dir "PYTHONPATH" "$HELP_SCRIPTS/general"
# export PYTHONPATH=$LOCAL_HELP_SCRIPTS/general:$PYTHONPATH
# export PATH=$PATH:$LARS_SCRIPTS_DIR
# export PATH=$PATH:$JIE_SCRIPTS_DIR
# export PATH=$PATH:$mldp


# APPTAINER
export APPTAINER_REPO=$SCRATCH/softwares/APPTAINER_REPO
export APPTAINER_CACHEDIR=$SCRATCH/APPTAINER_CACHE
export APPTAINER_TMPDIR=/tmp


# increasing commands remembered in history
HISTSIZE=50000 # number of commands to remember in history in memory
HISTFILESIZE=100000 # number of lines in history file
end_block7=$(date +%s)
elapsed_block7=$(( end_block7 - start_block7 ))
if [ $elapsed_block7 -gt 10 ]; then
  echo "WARNING: Exporting folder paths (block 7) took $elapsed_block7 seconds!"
fi








##########################################
# BLOCK 8: Creating local copy directories
##########################################
start_block8=$(date +%s)
if [ $verbose -eq 1 ]; then
  echo "Creating local copy of useful directories at $(date) ..."
fi

# Setting up local copies for directories that currently exist in the chosen
# PRIMARY_PROJECTS_FOLDER. This keeps active NCSA_DELTA trees synced while skipping
# optional trees that are not present yet.
DIR1="misc_libraries/scripts_Jie"
DIR2="misc_libraries/Box_Lars"
DIR3="misc_libraries/vatic-master"
DIR4="misc_libraries/XDATCAR_toolkit"
# DIR5="run_scripts/MLMD_scripts/mol_systems/MgSiOHN"
DIR6="run_scripts/helpful_scripts"
DIR7="run_scripts/ALCHEMY__in_use"

FILE1="myshortcuts.sh"
FILE2=".bashrc"
FILE3=".condarc"

ensure_local_copy_dir "$PRIMARY_PROJECTS_FOLDER/$DIR1" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR1" "$DIR1"
ensure_local_copy_dir "$PRIMARY_PROJECTS_FOLDER/$DIR2" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR2" "$DIR2"
ensure_local_copy_dir "$PRIMARY_PROJECTS_FOLDER/$DIR3" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR3" "$DIR3"
ensure_local_copy_dir "$PRIMARY_PROJECTS_FOLDER/$DIR4" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR4" "$DIR4"
# mkdir -p $LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR5
ensure_local_copy_dir "$PRIMARY_PROJECTS_FOLDER/$DIR6" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR6" "$DIR6"
ensure_local_copy_dir "$PRIMARY_PROJECTS_FOLDER/$DIR7" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR7" "$DIR7"

# Refresh local project paths after creating any missing local-copy directories.
append_to_env_var_if_dir "PATH" "$LOCAL_PRIMARY_PROJECTS_FOLDER/misc_libraries"
append_to_env_var_if_dir "PATH" "$LOCAL_LARS_SCRIPTS_DIR"
append_to_env_var_if_dir "PATH" "$LOCAL_JIE_SCRIPTS_DIR"
append_to_env_var_if_dir "PATH" "$LOCAL_mldp"
append_to_env_var_if_dir "PYTHONPATH" "$LOCAL_PRIMARY_PROJECTS_FOLDER/misc_libraries"
end_block8=$(date +%s)
elapsed_block8=$(( end_block8 - start_block8 ))
if [ $elapsed_block8 -gt 10 ]; then
  echo "WARNING: Creating local copy directories (block 8) took $elapsed_block8 seconds!"
fi


##########################################
# BLOCK 9: rsync operations
##########################################
# if DELLA, don't run
# if [[ $CLUSTER == "DELLA" ]]; then
#   echo "DELLA cluster detected. Skipping rsync operations."
#   return
# fi
start_block9=$(date +%s)
if [ $verbose -eq 1 ]; then
  echo "Running rsync operations at $(date) ..."
fi

warn_rsync_seconds=10

run_rsync_timed() {
  local label="$1"
  shift
  local start_rsync end_rsync elapsed_rsync
  start_rsync=$(date +%s)
  "$@" > /dev/null 2>&1
  end_rsync=$(date +%s)
  elapsed_rsync=$(( end_rsync - start_rsync ))
  if [ "$elapsed_rsync" -gt "$warn_rsync_seconds" ]; then
    echo "WARNING: rsync [$label] took $elapsed_rsync seconds!"
  fi
}

# run_rsync_timed_if_source_exists
#
# Runs an rsync command only when the source path exists.
#
# Args:
#   label: Human-readable label for timing and verbose messages.
#   source_path: Source file or directory that must exist before rsync runs.
#   ...: Full rsync command and arguments.
# Returns:
#   0 if rsync runs or is skipped safely.
run_rsync_timed_if_source_exists() {
  local label="$1"
  local source_path="$2"
  shift 2

  if [ ! -e "$source_path" ]; then
    if [ "$verbose" -eq 1 ]; then
      echo "Skipping rsync [$label]: source not found at $source_path"
    fi
    return 0
  fi

  run_rsync_timed "$label" "$@"
}

# # only update new or recently updated files in the local copy of the BURROWS and JIEDENG directory
# rsync -av --update --progress $PRIMARY_PROJECTS_FOLDER/* $SCRATCH/local_copy__projects/BURROWS/akashgpt/ --exclude='/projects/BURROWS/akashgpt/run_scripts/MLMD_scripts/iteration_CROSS_CLUSTER' --exclude='/projects/BURROWS/akashgpt/VASP_POTPAW' --exclude='run_scripts/MLMD_scripts/mol_systems/MgSiOHN/deepmd_collection_TRAIN'  --exclude='run_scripts/MLMD_scripts/mol_systems/MgSiOHN/deepmd_collection_TEST'
run_rsync_timed_if_source_exists "$DIR1" "$PRIMARY_PROJECTS_FOLDER/$DIR1" rsync -av --update --progress --delete "$PRIMARY_PROJECTS_FOLDER/$DIR1/" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR1/"
run_rsync_timed_if_source_exists "$DIR2" "$PRIMARY_PROJECTS_FOLDER/$DIR2" rsync -av --update --progress --delete "$PRIMARY_PROJECTS_FOLDER/$DIR2/" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR2/"
run_rsync_timed_if_source_exists "$DIR3" "$PRIMARY_PROJECTS_FOLDER/$DIR3" rsync -av --update --progress --delete "$PRIMARY_PROJECTS_FOLDER/$DIR3/" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR3/"
run_rsync_timed_if_source_exists "$DIR4" "$PRIMARY_PROJECTS_FOLDER/$DIR4" rsync -av --update --progress --delete "$PRIMARY_PROJECTS_FOLDER/$DIR4/" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR4/"
# rsync -av --update --progress --delete  --exclude='$PRIMARY_PROJECTS_FOLDER/$DIR5/deepmd_collection_TRAIN' --exclude='$PRIMARY_PROJECTS_FOLDER/$DIR5/deepmd_collection_TEST' --exclude='deepmd_collection_TRAIN' --exclude='deepmd_collection_TEST' $PRIMARY_PROJECTS_FOLDER/$DIR5/*  $LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR5 > /dev/null 2>&1
run_rsync_timed_if_source_exists "$DIR6" "$PRIMARY_PROJECTS_FOLDER/$DIR6" rsync -av --update --progress --delete "$PRIMARY_PROJECTS_FOLDER/$DIR6/" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR6/"
run_rsync_timed_if_source_exists "$DIR7" "$PRIMARY_PROJECTS_FOLDER/$DIR7" rsync -av --update --progress --delete --exclude='iteration_CROSS_CLUSTER' "$PRIMARY_PROJECTS_FOLDER/$DIR7/" "$LOCAL_PRIMARY_PROJECTS_FOLDER/$DIR7/"

# rsync -av --update --progress --delete $PRIMARY_PROJECTS_FOLDER/$FILE1  $LOCAL_PRIMARY_PROJECTS_FOLDER/$FILE1 > /dev/null 2>&1
# rsync -av --update --progress --delete $PRIMARY_PROJECTS_FOLDER/$FILE1  $HELP_SCRIPTS/sys/$FILE1 > /dev/null 2>&1
run_rsync_timed_if_source_exists "${CLUSTER}${FILE2}" "$HOME/$FILE2" rsync -av --update --progress --delete "$HOME/$FILE2" "$HELP_SCRIPTS/sys/${CLUSTER}${FILE2}"
run_rsync_timed_if_source_exists "${CLUSTER}${FILE3}" "$HOME/$FILE3" rsync -av --update --progress --delete "$HOME/$FILE3" "$HELP_SCRIPTS/sys/${CLUSTER}${FILE3}"
# rsync -av --update --progress $PRIMARY_PROJECTS_FOLDER/VASP_POTPAW/* $SCRATCH/local_copy__projects/BURROWS/VASP_POTPAW
end_block9=$(date +%s)
elapsed_block9=$(( end_block9 - start_block9 ))
if [ $elapsed_block9 -gt 10 ]; then
  echo "WARNING: rsync operations (block 9) took $elapsed_block9 seconds!"
fi

##########################################
if [ "$CLUSTER" == "NCSA_DELTA" ]; then
  # module reset >/dev/null 2>&1
  echo 
elif [ "$CLUSTER" == "DELLA" ] || [ "$CLUSTER" == "TIGER" ] || [ "$CLUSTER" == "STELLAR" ]; then
  module purge >/dev/null 2>&1
fi
##########################################

# Tag time when this script ends
echo "Done with myshortcuts.sh at $(date)"
echo ""

# Create a tag file with .bashrc to indicate that the script has been sourced
rm -f $HOME/.tag.myshortcuts.sh
touch $HOME/.tag.myshortcuts.sh
