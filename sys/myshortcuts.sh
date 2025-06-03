#!/bin/bash

####################################################################################################
# Summary:
#   This script is meant to be sourced in the .bashrc file to set up useful aliases and environment variables.
#   It is meant to be used in the Princeton clusters (DELLA, TIGER, TIGER3, STELLAR).
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
end_block1=$(date +%s)
elapsed_block1=$(( end_block1 - start_block1 ))
if [ $elapsed_block1 -gt 10 ]; then
  echo "WARNING: Cluster name detection took $elapsed_block1 seconds!"
fi


##########################################
# BLOCK 2: squeue aliases
##########################################
start_block2=$(date +%s)
# if verbose; then
if [ $verbose -eq 1 ]; then
  echo "Setting up aliases at $(date) ..."
fi
# squeue
alias sqp='squeue -o "%.18i %Q %.9q %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | more' #priority rating
alias sqpmy='squeue -o "%.18i %Q %.9q %.8j %.8u %.10a %.2t %.10M %.10L %.6C %R" | grep $USER' #priority rating
alias sq='squeue -u $USER -o "%.18i %.9P %.12j %.8u %.2t %.10M %.6D %.8C %.10l"
'
end_block2=$(date +%s)
elapsed_block2=$(( end_block2 - start_block2 ))
if [ $elapsed_block2 -gt 10 ]; then
  echo "WARNING: Defining squeue aliases took $elapsed_block2 seconds!"
fi


##########################################
# BLOCK 3: Color support & other ls/grep aliases
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
alias cq='checkquota'

mk() { mkdir -p "$1" && cd "$1"; }
cl() { cd "$1" && ll; } # uses alias defined above
alias ..='cd .. && ll'
alias ...='cd ../.. && ll'
alias pwd='pwd -P'
end_block3=$(date +%s)
elapsed_block3=$(( end_block3 - start_block3 ))
if [ $elapsed_block3 -gt 10 ]; then
  echo "WARNING: Color/ls aliases block took $elapsed_block3 seconds!"
fi


##########################################
# BLOCK 4: myfind & hog/hog_gpu functions
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
end_block4=$(date +%s)
elapsed_block4=$(( end_block4 - start_block4 ))
if [ $elapsed_block4 -gt 10 ]; then
  echo "WARNING: hog/hog_gpu definition block took $elapsed_block4 seconds!"
fi


##########################################
# BLOCK 5: myjobs function
##########################################
start_block5=$(date +%s)
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
}
end_block5=$(date +%s)
elapsed_block5=$(( end_block5 - start_block5 ))
if [ $elapsed_block5 -gt 10 ]; then
  echo "WARNING: myjobs function block took $elapsed_block5 seconds!"
fi


##########################################
# BLOCK 6: environment & conda aliases
##########################################
start_block6=$(date +%s)
if [ $verbose -eq 1 ]; then
  echo "Setting up environment and conda aliases at $(date) ..."
fi
# setting up environment in cluster
alias modl='module load'
alias sb='sbatch'
alias js='jobstats'
alias jspending='scontrol show job'

# conda related alias
alias conda_a='conda activate'
alias conda_d='conda deactivate'
alias l_base='module load anaconda3/2024.6; conda activate base'
alias l_hpc='module load anaconda3/2024.6; conda activate hpc-tools'
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
alias l_dp_plmd='module load anaconda3/2024.6; conda activate dp_plmd_09' #STELLAR | DELLA
alias l_mda='module load anaconda3/2024.6; conda activate mda_env' #TIGER3 | STELLAR | DELLA
alias l_asap='module load anaconda3/2024.6; conda activate asap'

if [[ $CLUSTER == "STELLAR" ]]; then
  alias l_qmda='module load anaconda3/2024.6; conda activate qmda' #STELLAR
elif [[ $CLUSTER == "DELLA" ]]; then
  alias l_qmda='module load anaconda3/2024.2; conda activate qmda' #DELLA
fi 
# alias l_dp_plmd='module load anaconda3/2024.2; conda activate dp_plmd' #DELLA; deepmd-kit 2.2.12-dev
# alias l_dpdev='module load anaconda3/2024.6; conda activate dpdev' #DELLA; deepmd-kit 2.2.12-dev

alias l_ase='module load anaconda3/2024.6; conda activate ase_env'

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

end_block6=$(date +%s)
elapsed_block6=$(( end_block6 - start_block6 ))
if [ $elapsed_block6 -gt 10 ]; then
  echo "WARNING: environment/conda aliases block took $elapsed_block6 seconds!"
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
export AG_TIGERDATA="/tigerdata/burrows/planet_evo/akashgpt"
export AG_TIGERDATA_2="/tigerdata/jiedeng/exoplanet/akashgpt"
export BACKUP_DIR="$SCRATCH/akashgpt_ucla_desktop_backup_20231231"
export VASP_ANALYSIS="$AG_BACKUP/Academics/Research/VASP/analysis_codes"
export VASP_DATA="$SCRATCH/qmd_data"
export mldp="$AG_BURROWS/misc_libraries/scripts_Jie/mldp"
export LOCAL_mldp="$LOCAL_AG_BURROWS/misc_libraries/scripts_Jie/mldp"
export JIE_SCRIPTS_DIR="$AG_BURROWS/misc_libraries/scripts_Jie"
export LOCAL_JIE_SCRIPTS_DIR="$LOCAL_AG_BURROWS/misc_libraries/scripts_Jie"
export LARS_SCRIPTS_DIR="$AG_BURROWS/misc_libraries/Box_Lars"
export LOCAL_LARS_SCRIPTS_DIR="$LOCAL_AG_BURROWS/misc_libraries/Box_Lars"
export MY_MLMD_SCRIPTS="$AG_BURROWS/run_scripts/MLMD_scripts"
export APPTAINER_REPO="$SCRATCH/softwares/APPTAINER_REPO"
export DPAL__dev="$AG_BURROWS/run_scripts/DPAL__dev"
export DPAL__main="$AG_BURROWS/run_scripts/DPAL__in_use"
export CONDA_SECONDARY_DIR="$SCRATCH/softwares/conda_envs_dir_secondary"

export HELP_SCRIPTS="$AG_BURROWS/run_scripts/helpful_scripts"
export LOCAL_HELP_SCRIPTS="$LOCAL_AG_BURROWS/run_scripts/helpful_scripts"
export HELP_SCRIPTS_qmd="$AG_BURROWS/run_scripts/helpful_scripts/qmd"
export HELP_SCRIPTS_vasp="$AG_BURROWS/run_scripts/helpful_scripts/qmd/vasp"
export HELP_SCRIPTS_plmd="$AG_BURROWS/run_scripts/helpful_scripts/qmd/plmd"
export HELP_SCRIPTS_TI="$AG_BURROWS/run_scripts/helpful_scripts/qmd/TI"
export HELP_SCRIPTS_DPAL="$AG_BURROWS/run_scripts/helpful_scripts/qmd/DPAL"



# export MDNN="/scratch/gpfs/ag5805/qmd_data/NH3_MgSiO3/sim_data_ML"

##########
# Adding to different default paths
##########
export PATH=$PATH:$LOCAL_AG_BURROWS/misc_libraries/
export PATH=$PATH:$LOCAL_LARS_SCRIPTS_DIR
export PATH=$PATH:$LOCAL_JIE_SCRIPTS_DIR
export PATH=$PATH:$LOCAL_mldp
export PATH=$HOME/local/bin:$PATH # for patchelf


export PYTHONPATH=$PYTHONPATH:$LOCAL_AG_BURROWS/misc_libraries/
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
  echo "WARNING: Exporting folder paths block took $elapsed_block7 seconds!"
fi








##########################################
# BLOCK 8: Creating local copy directories
##########################################
start_block8=$(date +%s)
if [ $verbose -eq 1 ]; then
  echo "Creating local copy of useful directories at $(date) ..."
fi

# Setting up local copy of folders and files from {/projects/BURROWS}++ directories
DIR1="misc_libraries/scripts_Jie"
DIR2="misc_libraries/Box_Lars"
DIR3="misc_libraries/vatic-master"
DIR4="misc_libraries/XDATCAR_toolkit"
# DIR5="run_scripts/MLMD_scripts/mol_systems/MgSiOHN"
DIR6="run_scripts/helpful_scripts"
DIR7="run_scripts/DPAL__in_use"
DIR8=

FILE1="myshortcuts.sh"
FILE2=".bashrc"

mkdir -p $LOCAL_AG_BURROWS/$DIR1
mkdir -p $LOCAL_AG_BURROWS/$DIR2
mkdir -p $LOCAL_AG_BURROWS/$DIR3
mkdir -p $LOCAL_AG_BURROWS/$DIR4
# mkdir -p $LOCAL_AG_BURROWS/$DIR5
mkdir -p $LOCAL_AG_BURROWS/$DIR6
mkdir -p $LOCAL_AG_BURROWS/$DIR7
end_block8=$(date +%s)
elapsed_block8=$(( end_block8 - start_block8 ))
if [ $elapsed_block8 -gt 10 ]; then
  echo "WARNING: Creating local copy directories took $elapsed_block8 seconds!"
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
# # only update new or recently updated files in the local copy of the BURROWS and JIEDENG directory
# rsync -av --update --progress $AG_BURROWS/* $SCRATCH/local_copy__projects/BURROWS/akashgpt/ --exclude='/projects/BURROWS/akashgpt/run_scripts/MLMD_scripts/iteration_CROSS_CLUSTER' --exclude='/projects/BURROWS/akashgpt/VASP_POTPAW' --exclude='run_scripts/MLMD_scripts/mol_systems/MgSiOHN/deepmd_collection_TRAIN'  --exclude='run_scripts/MLMD_scripts/mol_systems/MgSiOHN/deepmd_collection_TEST'
rsync -av --update --progress --delete $AG_BURROWS/$DIR1/*  $LOCAL_AG_BURROWS/$DIR1 > /dev/null 2>&1
rsync -av --update --progress --delete $AG_BURROWS/$DIR2/*  $LOCAL_AG_BURROWS/$DIR2 > /dev/null 2>&1
rsync -av --update --progress --delete $AG_BURROWS/$DIR3/*  $LOCAL_AG_BURROWS/$DIR3 > /dev/null 2>&1
rsync -av --update --progress --delete $AG_BURROWS/$DIR4/*  $LOCAL_AG_BURROWS/$DIR4 > /dev/null 2>&1
# rsync -av --update --progress --delete  --exclude='$AG_BURROWS/$DIR5/deepmd_collection_TRAIN' --exclude='$AG_BURROWS/$DIR5/deepmd_collection_TEST' --exclude='deepmd_collection_TRAIN' --exclude='deepmd_collection_TEST' $AG_BURROWS/$DIR5/*  $LOCAL_AG_BURROWS/$DIR5 > /dev/null 2>&1
rsync -av --update --progress --delete $AG_BURROWS/$DIR6/*  $LOCAL_AG_BURROWS/$DIR6 > /dev/null 2>&1
rsync -av --update --progress --delete --exclude='iteration_CROSS_CLUSTER' "$AG_BURROWS/$DIR7/" "$LOCAL_AG_BURROWS/$DIR7" > /dev/null 2>&1

rsync -av --update --progress --delete $AG_BURROWS/$FILE1  $LOCAL_AG_BURROWS/$FILE1 > /dev/null 2>&1
rsync -av --update --progress --delete $AG_BURROWS/$FILE1  $HELP_SCRIPTS/sys/$FILE1 > /dev/null 2>&1
rsync -av --update --progress --delete $HOME/$FILE2  $HELP_SCRIPTS/sys/${CLUSTER}${FILE2} > /dev/null 2>&1
# rsync -av --update --progress $AG_BURROWS/VASP_POTPAW/* $SCRATCH/local_copy__projects/BURROWS/VASP_POTPAW
end_block9=$(date +%s)
elapsed_block9=$(( end_block9 - start_block9 ))
if [ $elapsed_block9 -gt 10 ]; then
  echo "WARNING: rsync operations block took $elapsed_block9 seconds!"
fi


# Tag time when this script ends
echo "Done with myshortcuts.sh at $(date)"
echo ""

# Create a tag file with .bashrc to indicate that the script has been sourced
rm -f $HOME/.tag.myshortcuts.sh
touch $HOME/.tag.myshortcuts.sh