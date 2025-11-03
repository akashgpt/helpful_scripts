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
if [ -d ~/.bashrc.d ]; then
	for rc in ~/.bashrc.d/*; do
		if [ -f "$rc" ]; then
			. "$rc"
		fi
	done
fi

unset rc


# User specific aliases and functions

# my aliases and functions
# unset PYTHONPATH

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

# cp -r 

#adding VASP to path
# export VASP_EXEC="/scratch/gpfs/BURROWS/akashgpt/softwares/vasp.6.3.2/bin"
export VASP_EXEC="/scratch/gpfs/BURROWS/akashgpt/softwares/vasp.6.4.3/bin"
export PATH=$PATH:$VASP_EXEC


#
# Based on the following instructions when installing deepmd via conda:
#
# To enable it, first install UCX (conda install -c conda-forge ucx).                                       
# Afterwards, set the environment variables                                                                 
# OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx                                                                         
# before launching your MPI processes.                                                                      
# Equivalently, you can set the MCA parameters in the command line:                                         
# mpiexec --mca pml ucx --mca osc ucx ...                                                                   
#                                                                                                
# On Linux, Open MPI is built with UCC support but it is disabled by default.                               
# To enable it, first install UCC (conda install -c conda-forge ucc).                                       
# Afterwards, set the environment variables                                                                 
# OMPI_MCA_coll_ucc_enable=1                                                                                
# before launching your MPI processes.                                                                      
# Equivalently, you can set the MCA parameters in the command line:
# mpiexec --mca coll_ucc_enable 1 ...
#
# On Linux, Open MPI is built with CUDA awareness but it is disabled by default.
# To enable it, please set the environment variable
# OMPI_MCA_opal_cuda_support=true
# before launching your MPI processes.
# Equivalently, you can set the MCA parameter in the command line:
# mpiexec --mca opal_cuda_support 1 ...
# Note that you might also need to set UCX_MEMTYPE_CACHE=n for CUDA awareness via
# UCX. Please consult UCX documentation for further details
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_opal_cuda_support=true
export UCX_MEMTYPE_CACHE=n

# conda install cuda-cudart cuda-version=12

# # APPTAINER
# export APPTAINER_CACHEDIR=$SCRATCH/APPTAINER_CACHE
# export APPTAINER_TMPDIR=/tmp
