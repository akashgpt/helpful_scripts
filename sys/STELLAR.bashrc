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
export VASP_EXEC="$HOME/softwares/vasp.6.3.2/bin"
export PATH=$PATH:$VASP_EXEC

# to edit terminal header look
# PS1='[\u@\h:\W]\$ '

# export PLUMED_KERNEL=/home/ag5805/.conda/envs/dp2.2.7/lib/libplumedKernel.so
# export LAMMPS_PLUGIN_PATH=/home/ag5805/.conda/envs/dp2.2.7/lib/deepmd_lmp
# patchelf --add-rpath $CONDA_PREFIX/lib dpplugin.so
