# .bashrc

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

# # check if the $HOME/.tag.myshortcuts.sh file exists and when it was created. Run the following commands if it is older than the /projects/BURROWS/akashgpt/myshortcuts.sh file
# OG_myshortcuts_file="/projects/BURROWS/akashgpt/myshortcuts.sh"
# if [ -f $HOME/.tag.myshortcuts.sh ]; then
#   if [ $HOME/.tag.myshortcuts.sh -ot $OG_myshortcuts_file ]; then
#     echo "The $HOME/.tag.myshortcuts.sh file is older than the $OG_myshortcuts_file file. Running myshortcuts.sh."
#     if [ -f $OG_myshortcuts_file ] && [ ! -z "$PS1" ]; then
#       source $OG_myshortcuts_file
#     fi
#   # fi
#   else
#     echo "The $HOME/.tag.myshortcuts.sh file is newer than the $OG_myshortcuts_file file. Not running myshortcuts.sh."
#   fi
# else
#   echo "The $HOME/.tag.myshortcuts.sh file does not exist. Creating the $HOME/.tag.myshortcuts.sh file."
#   source $OG_myshortcuts_file
# fi


# COPY_myshortcuts_file="$SCRATCH/local_copy__projects/BURROWS/akashgpt/myshortcuts.sh"
# if [ -f $HOME/.tag.myshortcuts.sh ]; then
#   if [ $HOME/.tag.myshortcuts.sh -ot $COPY_myshortcuts_file ]; then
#     echo "The $HOME/.tag.myshortcuts.sh file is older than the $COPY_myshortcuts_file file. Running myshortcuts.sh."
#     if [ -f $COPY_myshortcuts_file ]; then
#       source $COPY_myshortcuts_file
#     fi
#   # fi
#   else
#     echo "The $HOME/.tag.myshortcuts.sh file is newer than the $COPY_myshortcuts_file file. Not running myshortcuts.sh."
#   fi
# else
#   echo "The $HOME/.tag.myshortcuts.sh file does not exist. Creating the $HOME/.tag.myshortcuts.sh file."
#   source $OG_myshortcuts_file
# fi

if [ -f /projects/BURROWS/akashgpt/myshortcuts.sh ] && [ ! -z "$PS1" ]; then
  source /projects/BURROWS/akashgpt/myshortcuts.sh
fi

if [ -f $SCRATCH/local_copy__projects/BURROWS/akashgpt/myshortcuts.sh ]; then
  source $SCRATCH/local_copy__projects/BURROWS/akashgpt/myshortcuts.sh
fi

#adding VASP to path
# export VASP_EXEC="$HOME/softwares/vasp.6.3.2/bin"
# export PATH=$PATH:$VASP_EXEC
# /scratch/gpfs/BURROWS/akashgpt/softwares/vasp.6.4.3/bin
# export VASP_EXEC="/scratch/gpfs/BURROWS/akashgpt/softwares/vasp.6.4.3/bin"
# export PATH=$PATH:$VASP_EXEC
# define the chunk to remove
old="$HOME/softwares/vasp.6.3.2/bin"
# 1) remove any “old:” at the front
PATH=${PATH#"$old:"}
# 2) remove any “:old” at the end
PATH=${PATH%":$old"}
# 3) remove any “:old:” in the middle
PATH=${PATH//":$old:"/:}
export PATH

# to edit terminal header look
# PS1='[\u@\h:\W]\$ '

# export PLUMED_KERNEL=/home/ag5805/.conda/envs/dp2.2.7/lib/libplumedKernel.so
# export LAMMPS_PLUGIN_PATH=/home/ag5805/.conda/envs/dp2.2.7/lib/deepmd_lmp
# patchelf --add-rpath $CONDA_PREFIX/lib dpplugin.so

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/licensed/anaconda3/2024.6/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/licensed/anaconda3/2024.6/etc/profile.d/conda.sh" ]; then
        . "/usr/licensed/anaconda3/2024.6/etc/profile.d/conda.sh"
    else
        export PATH="/usr/licensed/anaconda3/2024.6/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<



conda deactivate