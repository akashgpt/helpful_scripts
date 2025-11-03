# .bashrc

# author: akashgpt
# date: 20250211

verbose=${1:-0}

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# my aliases and functions
unset PYTHONPATH

# check if /projects/BURROWS/akashgpt/myshortcuts.sh exists, if yes -- source it
# if not, check if /projects/BURROWS/akashgpt/run_scripts/helpful_scripts/sys/myshortcuts.sh exists -- then source it

if [ -f /projects/BURROWS/akashgpt/myshortcuts.sh ] && [ ! -z "$PS1" ]; then
  source /projects/BURROWS/akashgpt/myshortcuts.sh ${verbose}
fi

if [ -f $SCRATCH/local_copy__projects/BURROWS/akashgpt/myshortcuts.sh ]; then
  source $SCRATCH/local_copy__projects/BURROWS/akashgpt/myshortcuts.sh ${verbose}
fi

if [ -f /projects/BURROWS/akashgpt/run_scripts/helpful_scripts/sys/myshortcuts.sh ] && [ ! -z "$PS1" ]; then
  source /projects/BURROWS/akashgpt/run_scripts/helpful_scripts/sys/myshortcuts.sh ${verbose}
fi

if [ -f $SCRATCH/local_copy__projects/BURROWS/akashgpt/run_scripts/helpful_scripts/sys/myshortcuts.sh ]; then
  source $SCRATCH/local_copy__projects/BURROWS/akashgpt/run_scripts/helpful_scripts/sys/myshortcuts.sh ${verbose}
fi

#adding VASP to path
# export VASP_EXEC="$HOME/softwares/vasp.6.3.2/bin/"
# export PATH=$PATH:$VASP_EXEC

# to edit terminal header look
# PS1='[\u@\h:\W]\$ '


# plumed installation -- for conda environment dp_plmd
# export PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dp_plmd/bin:$PATH"
# export C_INCLUDE_PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dp_plmd/include:$C_INCLUDE_PATH"
# export CPLUS_INCLUDE_PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dp_plmd/include:$CPLUS_INCLUDE_PATH"
# export LD_LIBRARY_PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dp_plmd/lib:$LD_LIBRARY_PATH"
# export PKG_CONFIG_PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dp_plmd/lib/pkgconfig:$PKG_CONFIG_PATH"
# export PLUMED_KERNEL=/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dp_plmd/lib/libplumedKernel.so


# plumed installation -- for conda environment dpdev
# export PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dpdev/bin:$PATH"
# export C_INCLUDE_PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dpdev/include:$C_INCLUDE_PATH"
# export CPLUS_INCLUDE_PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dpdev/include:$CPLUS_INCLUDE_PATH"
# export LD_LIBRARY_PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dpdev/lib:$LD_LIBRARY_PATH"
# export PKG_CONFIG_PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dpdev/lib/pkgconfig:$PKG_CONFIG_PATH"
# export PLUMED_KERNEL=/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dpdev/lib/libplumedKernel.so
# To create a tcl module that sets all the variables above, use this one as a starting point:
# /scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dpdev/lib/plumed/modulefile
# A vim plugin can be found here: /scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dpdev/lib/plumed/vim/
# Copy it to /home/ag5805/.vim/ directory
# Alternatively:
# - Set this environment variable         : PLUMED_VIMPATH=/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/dpdev/lib/plumed/vim
# - Add the command 'let &runtimepath.=','.$PLUMED_VIMPATH' to your .vimrc file
# From vim, you can use :set syntax=plumed to enable it


# adding new conda environments to PATH and PYTHONPATH
# export PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/deepmd_cpu/bin:$PATH"
# # export PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/deepmd_gpu/bin:$PATH"
# export PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/qmda/bin:$PATH"
# export PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/mda_env/bin:$PATH"
# export PATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/planet_evo/bin:$PATH"

# export PYTHONPATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/deepmd_cpu/lib/python3.10/site-packages:$PYTHONPATH"
# # export PYTHONPATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/deepmd_gpu/lib/python*/site-packages:$PYTHONPATH"
# export PYTHONPATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/qmda/lib/python*/site-packages:$PYTHONPATH"
# export PYTHONPATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/mda_env/lib/python3.12/site-packages:$PYTHONPATH"
# export PYTHONPATH="/scratch/gpfs/ag5805/softwares/conda_envs_dir_secondary/envs/planet_evo/lib/python*/site-packages:$PYTHONPATH"

# for cmake -- add to path /home/ag5805/.local/bin
export PATH="$HOME/.local/bin:$PATH"