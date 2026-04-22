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

if [ -f /projects/bguf/akashgpt/run_scripts/helpful_scripts/sys/myshortcuts.sh ] && [ ! -z "$PS1" ]; then
	source /projects/bguf/akashgpt/run_scripts/helpful_scripts/sys/myshortcuts.sh ${verbose}
elif [ -f $SCRATCH/local_copy__projects/bguf/akashgpt/run_scripts/helpful_scripts/sys/myshortcuts.sh ]; then
	source $SCRATCH/local_copy__projects/bguf/akashgpt/run_scripts/helpful_scripts/sys/myshortcuts.sh ${verbose}
fi
