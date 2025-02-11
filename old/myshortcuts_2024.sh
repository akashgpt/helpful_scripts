#!/bin/bash

##########
# aliases
##########

# check name of current cluster
if [[ $(hostname) == *"della"* ]]; then
  export CLUSTER="DELLA"
  export SCRATCH="/scratch/gpfs/$USER"
elif [[ $(hostname) == *"tigercpu"* ]]; then
  export CLUSTER="TIGER"
  export SCRATCH="/scratch/gpfs/$USER"
elif [[ $(hostname) == *"tiger3"* ]]; then
  export CLUSTER="TIGER3"
  export SCRATCH="/scratch/gpfs/BURROWS/akashgpt"
elif [[ $(hostname) == *"stellar"* ]]; then
  export CLUSTER="STELLAR"
  export SCRATCH="/scratch/gpfs/$USER"
fi

# squeue
alias sqp='squeue -o "%.18i %Q %.9q %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | more' #priority rating
alias sqpmy='squeue -o "%.18i %Q %.9q %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | grep $USER | more' #priority rating
alias sq='squeue -u $USER -o "%.18i %.9P %.12j %.8u %.2t %.10M %.6D %.8C %.10l"
'

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

alias ll='ls -clF -h'
alias la='ls -A'
alias l='ls -CF'
alias ls='ls --color=auto'
alias lsf='ls --color=auto -p | grep -v /' # only files + sort by time
alias llt='ls --color=auto -hlctur' # sort by time
alias llf='ls --color=auto -hlctur -p | grep -v /' # only files + sort by time
alias lls='ls --color=auto -lhcS' # sort by size
alias ll='ls --color=auto -lhc'
alias du='du -csh'
alias cq='checkquota'

mk() { mkdir -p "$1" && cd "$1"; }
cl() { cd "$1" && ll; } # uses alias defined above
alias ..='cd .. && ll'
alias ...='cd ../.. && ll'
alias pwd='pwd -P'


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
  # Default values
  days=${1:-30}
  account=${2:-astro}
  topcount=${3:-20}

  # Calculate the start date based on the provided number of days
  start_date=$(date -d"${days} days ago" +%D)

  # Construct the sreport command based on whether 'all' is specified
  if [ "$account" == "all" ]; then
    sreport user topusage start=${start_date} end=now TopCount=${topcount} -t hourper --tres=cpu
  else
    sreport user topusage start=${start_date} end=now TopCount=${topcount} accounts=${account} -t hourper --tres=cpu
  fi
}

hog_gpu() {
  # Default values
  days=${1:-30}
  account=${2:-astro}
  topcount=${3:-20}
  
  # Calculate the start date based on the provided number of days
  start_date=$(date -d"${days} days ago" +%D)

  # Construct the sreport command based on whether 'all' is specified
  if [ "$account" == "all" ]; then
    sreport user topusage start=${start_date} end=now TopCount=${topcount} -t hourper --tres=gres/gpu;
  else
    sreport user topusage start=${start_date} end=now TopCount=${topcount} accounts=${account} -t hourper --tres=gres/gpu;
  fi
}


myjobs() {
  core_flag=${1:-0}
  
  # Default values
  jobs_running=$(squeue -u $USER -h -t R -o "%.18i %.9P %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | wc -l)
  jobs_pending=$(squeue -u $USER -h -t PD -o "%.18i %.9P %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | wc -l)
  total_jobs=$(($jobs_running + $jobs_pending))

  echo "####################"
  echo "Jobs for user $USER"
  echo "####################"
  echo "Total jobs: $total_jobs"

  # if [ "$core_flag" == "-v" ]; then
  # cores_list=$(qos | grep "$USER" | awk '{print $3}')
  cores_list=$(sqp | grep ag5805 | grep " R " | awk '{print $10}')

  # Sum all instances of total_cores
  total_cores_sum=0
  for core in $cores_list; do
      total_cores_sum=$((total_cores_sum + core))
  done

  echo "Running jobs: $jobs_running, procs $total_cores_sum";
  # else
  #   echo "Running jobs: $jobs_running";
  # fi

  echo "Pending jobs: $jobs_pending"

  # # if cluster is TIGER3
  # if [[ $CLUSTER == "TIGER3" ]]; then
    
  #   echo "####################"
  #   echo "More details ..."
  #   ########################################

  #   local user="${1:-$USER}"  # Default to current user if no username is provided
  #     declare -A qos_counts=( ["test"]=0 ["vshort"]=0 ["short"]=0 ["medium"]=0 ["long"]=0 )

  #     # Run squeue and parse the output
  #     squeue -u "$user" -o "%.10l" | tail -n +2 | while read -r time_limit; do
  #         # Convert TIME_LIMIT to total hours
  #         if [[ "$time_limit" == "UNLIMITED" ]]; then
  #             continue
  #         fi
          
  #         # Extract hours, minutes, and seconds from time strings (e.g., "2-12:00:00")
  #         if [[ "$time_limit" =~ ([0-9]+)-([0-9]+):([0-9]+):([0-9]+) ]]; then
  #             # Format: D-HH:MM:SS
  #             days="${BASH_REMATCH[1]}"
  #             hours="${BASH_REMATCH[2]}"
  #             minutes="${BASH_REMATCH[3]}"
  #             total_hours=$((days * 24 + hours + minutes / 60))
  #         elif [[ "$time_limit" =~ ([0-9]+):([0-9]+):([0-9]+) ]]; then
  #             # Format: HH:MM:SS
  #             hours="${BASH_REMATCH[1]}"
  #             minutes="${BASH_REMATCH[2]}"
  #             total_hours=$((hours + minutes / 60))
  #         else
  #             total_hours=0  # Default to 0 for malformed strings
  #         fi

  #         # Categorize jobs based on time limit
  #         if (( total_hours <= 1 )); then
  #             qos_counts["test"]=$((qos_counts["test"] + 1))      
  #         elif (( total_hours > 1 && total_hours <= 5 )); then
  #             qos_counts["vshort"]=$((qos_counts["vshort"] + 1))
  #         elif (( total_hours > 5 && total_hours <= 24 )); then
  #             qos_counts["short"]=$((qos_counts["short"] + 1))
  #         elif (( total_hours > 24 && total_hours <= 72 )); then
  #             qos_counts["medium"]=$((qos_counts["medium"] + 1))
  #         elif (( total_hours > 72 && total_hours < 144 )); then
  #             qos_counts["long"]=$((qos_counts["long"] + 1))
  #         fi
  #     done

  #     # Print the results
  #     for qos in "test" "vshort" "short" "medium" "long"; do
  #         echo "$qos, Jobs: ${qos_counts[$qos]}"
  #     done
  # fi
  #   echo "####################"
}


# myjobs_master(){

# # Define the user variable
# user=$USER

# # Print header
# echo "QoS,JobID,JobName,User,Partition,State,Elapsed,Timelimit"

# # Get all jobs for the user with their QoS information
# jobs=$(sacct --user=$user --format=JobID,JobName,User,Partition,State,Elapsed,Timelimit,QOS -n)

# # Loop through each job and print details
# while IFS= read -r job; do
#     # Extract QoS from job information
#     qos=$(echo $job | awk '{print $8}')
#     # Print job information with QoS
#     echo "$qos,$job"
# done <<< "$jobs"

# }

# setting up environment in cluster
alias modl='module load'
alias sb='sbatch'
alias js='jobstats'
alias jspending='scontrol show job'

# conda related alias
alias conda_a='conda activate'
alias conda_d='conda deactivate'
alias l_base='module load anaconda3/2024.6; conda activate base'
# alias l_dpdev='module load anaconda3/2021.5; conda activate dpdev'
alias l_planet_evo='module load anaconda3/2021.5; conda activate planet_evo'
alias l_chhota_apple='module load anaconda3/2024.6; conda activate chhota_apple'
# alias l_deepmd_cpu='module load anaconda3/2021.5; conda activate deepmd_cpu'

if [[ $CLUSTER == "DELLA" ]]; then
  alias l_deepmd_gpu='module load anaconda3/2021.5; conda activate deepmd_gpu' #DELLA; deepmd-kit 2.1.1
  alias l_deepmd_cpu='module load anaconda3/2024.6; conda activate deepmd_cpu' #DELLA; deepmd-kit 2.2.10
elif [[ $CLUSTER == "TIGER" ]]; then
  alias l_dp='module load anaconda3/2024.2; conda activate deepmd' #TIGER
elif [[ $CLUSTER == "TIGER3" ]]; then
  alias l_deepmd='module load anaconda3/2024.6; conda activate deepmd' #TIGER3
elif [[ $CLUSTER == "STELLAR" ]]; then
  alias l_deepmd='module load anaconda3/2024.6; conda activate deepmd' #STELLAR
fi

# alias l_dp2='module load anaconda3/2021.5; conda activate dp2.2.7; export PLUMED_KERNEL=$CONDA_PREFIX/lib/libplumedKernel.so; LAMMPS_PLUGIN_PATH=$CONDA_PREFIX/lib/deepmd_lmp; patchelf --add-rpath $CONDA_PREFIX/lib dpplugin.so'
alias l_mda='module load anaconda3/2024.6; conda activate mda_env' #TIGER3 | STELLAR | DELLA
alias l_asap='module load anaconda3/2024.6; conda activate asap'

if [[ $CLUSTER == "STELLAR" ]]; then
  alias l_qmda='module load anaconda3/2024.6; conda activate qmda' #STELLAR
elif [[ $CLUSTER == "DELLA" ]]; then
  alias l_qmda='module load anaconda3/2024.2; conda activate qmda' #DELLA
fi 
# alias l_dp_plmd='module load anaconda3/2024.2; conda activate dp_plmd' #DELLA; deepmd-kit 2.2.12-dev
# alias l_dpdev='module load anaconda3/2024.6; conda activate dpdev' #DELLA; deepmd-kit 2.2.12-dev

# git
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

# Defining useful folder paths
export AG_BURROWS="/projects/BURROWS/akashgpt"
export LOCAL_AG_BURROWS="$SCRATCH/local_copy__projects/BURROWS/akashgpt"
export AG_JIEDENG="/projects/JIEDENG/akashgpt"
export AG_TIGERDATA="/tigerdata/burrows/planet_evo/akashgpt"
export BACKUP_DIR="$SCRATCH/akashgpt_ucla_desktop_backup_20231231"
export VASP_ANALYSIS="$AG_BACKUP/Academics/Research/VASP/analysis_codes"
export VASP_DATA="$SCRATCH/qmd_data"
export mldp="$AG_BURROWS/misc_libraries/scripts_Jie/mldp"
export local_mldp="$LOCAL_AG_BURROWS/misc_libraries/scripts_Jie/mldp"
export JIE_SCRIPTS_DIR="$AG_BURROWS/misc_libraries/scripts_Jie"
export LARS_SCRIPTS_DIR="$AG_BURROWS/misc_libraries/Box_Lars"
export MY_MLMD_SCRIPTS="$AG_BURROWS/run_scripts/MLMD_scripts"
export APPTAINER_REPO="$SCRATCH/softwares/APPTAINER_REPO"
export DPAL__dev="$AG_BURROWS/run_scripts/DPAL"
export DPAL__main="$AG_BURROWS/run_scripts/DPAL__in_use"
# export MDNN="/scratch/gpfs/ag5805/qmd_data/NH3_MgSiO3/sim_data_ML"

##########
# Adding to different default paths
##########
export PATH=$PATH:$AG_BURROWS/misc_libraries/
export PATH=$PATH:$LARS_SCRIPTS_DIR
export PATH=$PATH:$JIE_SCRIPTS_DIR
export PATH=$PATH:$mldp

export PYTHONPATH=$PYTHONPATH:$AG_BURROWS/misc_libraries/
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



# Setting up local copy of BURROWS and JIEDENG directories from projects/
DIR1="misc_libraries/scripts_Jie"
DIR2="misc_libraries/Box_Lars"
DIR3="misc_libraries/vatic-master"
DIR4="misc_libraries/XDATCAR_toolkit"
DIR5="run_scripts/MLMD_scripts/mol_systems/MgSiOHN"
DIR6="run_scripts/MLMD_scripts/TRAIN_MLMD_scripts"
DIR7="run_scripts/DPAL__in_use"

FILE1="myshortcuts.sh"

mkdir -p $LOCAL_AG_BURROWS/$DIR1
mkdir -p $LOCAL_AG_BURROWS/$DIR2
mkdir -p $LOCAL_AG_BURROWS/$DIR3
mkdir -p $LOCAL_AG_BURROWS/$DIR4
mkdir -p $LOCAL_AG_BURROWS/$DIR5
mkdir -p $LOCAL_AG_BURROWS/$DIR6
mkdir -p $LOCAL_AG_BURROWS/$DIR7

# # only update new or recently updated files in the local copy of the BURROWS and JIEDENG directory
# rsync -av --update --progress $AG_BURROWS/* $SCRATCH/local_copy__projects/BURROWS/akashgpt/ --exclude='/projects/BURROWS/akashgpt/run_scripts/MLMD_scripts/iteration_CROSS_CLUSTER' --exclude='/projects/BURROWS/akashgpt/VASP_POTPAW' --exclude='run_scripts/MLMD_scripts/mol_systems/MgSiOHN/deepmd_collection_TRAIN'  --exclude='run_scripts/MLMD_scripts/mol_systems/MgSiOHN/deepmd_collection_TEST'
rsync -av --update --progress --delete $AG_BURROWS/$DIR1/*  $LOCAL_AG_BURROWS/$DIR1 > /dev/null 2>&1
rsync -av --update --progress --delete $AG_BURROWS/$DIR2/*  $LOCAL_AG_BURROWS/$DIR2 > /dev/null 2>&1
rsync -av --update --progress --delete $AG_BURROWS/$DIR3/*  $LOCAL_AG_BURROWS/$DIR3 > /dev/null 2>&1
rsync -av --update --progress --delete $AG_BURROWS/$DIR4/*  $LOCAL_AG_BURROWS/$DIR4 > /dev/null 2>&1
rsync -av --update --progress --delete  --exclude='$AG_BURROWS/$DIR5/deepmd_collection_TRAIN' --exclude='$AG_BURROWS/$DIR5/deepmd_collection_TEST' --exclude='deepmd_collection_TRAIN' --exclude='deepmd_collection_TEST' $AG_BURROWS/$DIR5/*  $LOCAL_AG_BURROWS/$DIR5 > /dev/null 2>&1
rsync -av --update --progress --delete $AG_BURROWS/$DIR6/*  $LOCAL_AG_BURROWS/$DIR6 > /dev/null 2>&1
rsync -av --update --progress --delete --exclude='$AG_BURROWS/$DIR7/iteration_CROSS_CLUSTER' $AG_BURROWS/$DIR7/*  $LOCAL_AG_BURROWS/$DIR7 > /dev/null 2>&1

rsync -av --update --progress --delete $AG_BURROWS/$FILE1  $LOCAL_AG_BURROWS/$FILE1 > /dev/null 2>&1
# rsync -av --update --progress $AG_BURROWS/VASP_POTPAW/* $SCRATCH/local_copy__projects/BURROWS/VASP_POTPAW
