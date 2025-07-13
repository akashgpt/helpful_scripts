#!/bin/bash

############################################################################
# RUN_VASP_MASTER.sh     vasp submit script                                #
#  																	   	   #
# Script created to automate submission of multiple jobs one after another #
#				                                                           #
# Author: Akash Gupta                                                      #
############################################################################

# echo process ID of this script
echo ""
echo ""
echo ""
echo "+++++++++++++++++++++++++++++++"
echo "Process ID: $$"
echo "+++++++++++++++++++++++++++++++"
echo ""

master_id=$(basename "$PWD") 
base_dir=$(pwd)

num_jobs=${1:-5}  # Default to 5 jobs if not specified
echo "Job series for ${master_id} started for ${num_jobs} simulations at" `date `


RUN_VASP_TIME=${2:-24}	# time of simulations, default of 24; options: 0.1, 0.5, 1, 4, 8, 12, 24, 48, 72, 96
						# if 0 -- retains OG RUN_VASP.sh file

CLUSTER_NAME=$(scontrol show config | grep ClusterName | awk '{print $3}')

if [ "$CLUSTER_NAME" == "tiger2" ]; then
	RUN_VASP_DIR="$HELP_SCRIPTS_vasp/RUN_VASP/TIGER"
	RUN_VASP_NODES=${3:-4} #number of nodes used, default of 4; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "tiger3" ]; then
	RUN_VASP_DIR="$HELP_SCRIPTS_vasp/RUN_VASP/TIGER3"
	RUN_VASP_NODES=${3:-2} #number of nodes used, default of 4; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "della" ]; then
	RUN_VASP_DIR="$HELP_SCRIPTS_vasp/RUN_VASP/DELLA"
	RUN_VASP_NODES=${3:-1} #number of nodes used, default of 1; options: 1, 2, 4, 8
elif [ "$CLUSTER_NAME" == "stellar" ]; then
	RUN_VASP_DIR="$HELP_SCRIPTS_vasp/RUN_VASP/STELLAR"
	RUN_VASP_NODES=${3:-2} #number of nodes used, default of 2; options: 1, 2, 4, 8
fi


PERCENTAGE_RESTART_SHIFT=${4:-20} # 15% of the run time steps

MINIMUM_TIME_STEP_THRESHOLD=${5:-200} # minimum time step threshold to consider a job as completed; default is 100

RESTART_MODE=${6:-0} # default of 0: initial condition based on RUN_DIRNAME{last_run} (just as usual); 5: initial condition based on shifting SCALEE_5, ...

# reduce PERCENTAGE_RESTART_SHIFT by 10; i.e. starting from 10%
# PERCENTAGE_RESTART_SHIFT=$(( PERCENTAGE_RESTART_SHIFT - 10 )) ## TEMPORARY

#################################################
#################################################
OUTCAR_size_l_limit_MB=1 # 1 MB
extended_job_flag=-1 # if value=-1 (default), jobs ONLY upto 'z' at max or 26 in number at max; if = 1, jobs ALREADY beyond z, i.e., into the 'aX' zone; if = 0, jobs WILL go beyond z this time.
#################################################
#################################################


echo ""
echo "========================="
echo "Time: $(date)"
echo "Cluster name: $CLUSTER_NAME"
echo "Number of nodes: $RUN_VASP_NODES"
echo "Number of jobs: $num_jobs"
echo "Run VASP time: $RUN_VASP_TIME"
echo "Run VASP directory: $RUN_VASP_DIR"
echo "Percentage restart shift: $PERCENTAGE_RESTART_SHIFT"
echo "OUTCAR size limit: $OUTCAR_size_l_limit_MB MB"
echo "Extended job flag: $extended_job_flag"
echo "Restart mode: $RESTART_MODE"
echo "Minimum time step threshold: $MINIMUM_TIME_STEP_THRESHOLD"
echo "========================="
echo ""


RUN_SCRIPT_FILE="${RUN_VASP_DIR}/RUN_VASP_T${RUN_VASP_TIME}h_N${RUN_VASP_NODES}.sh"

#OUTCAR_size_l_limit__Bytes in terms of OUTCAR_size_l_limit_MB
OUTCAR_size_l_limit__Bytes=$((OUTCAR_size_l_limit_MB * 1024 * 1024)) # convert MB to Bytes

# for converting decimal to ASCII
chr() {
	printf \\$(printf '%03o' $1)
}

# for converting ASCII to decimal
ord() {
	printf '%d' "'$1"
}


# check which is the last job completed
for letter in {a..z}; do
	if [ -d "../${master_id}${letter}" ]; then
		last_job_id_letter=$letter
	fi
done
for letter in {a..z}; do
	if [ -d "../${master_id}a${letter}" ]; then
		last_job_id_letter=$letter
		extended_job_flag=1
	fi
done

if (( $extended_job_flag==1 )); then
	cd ../"${master_id}a${last_job_id_letter}"/
	echo "Last job record found: ${master_id}a${last_job_id_letter}"
elif (( $extended_job_flag==-1 )); then
	cd ../"${master_id}${last_job_id_letter}"/
	echo "Last job record found: ${master_id}${last_job_id_letter}"
fi




start_job_ascii=$(ord $last_job_id_letter); (( start_job_ascii+=1 ))
start_job_letter=$(chr $start_job_ascii)
final_job_ascii=$(ord $last_job_id_letter); (( final_job_ascii+=$num_jobs ))
final_job_letter=$(chr $final_job_ascii)

if (($start_job_ascii > 122)); then
	extended_job_flag=0
	start_job_ascii_temp=$start_job_ascii
	(( start_job_ascii_temp-=26))
	start_job_letter=$(chr $start_job_ascii_temp)
fi
if (($final_job_ascii > 122)); then
	extended_job_flag=0
	final_job_ascii_temp=$final_job_ascii
	(( final_job_ascii_temp-=26))
	final_job_letter=$(chr $final_job_ascii_temp)
fi
letter_old=$last_job_id_letter
counter=0

first_letter=$(chr 97)
last_letter=$(chr 122)

# echo extended_job_flag
echo ""
echo "extended_job_flag: $extended_job_flag"
echo ""

# start new jobs

#if jobs crossing over from a-z to aa-az
if (( $extended_job_flag == 0 )); then
	if (( $start_job_ascii<123)); then
		for letter in $(eval echo {$start_job_letter..$last_letter}); do

			# check if the last job completed properly by just seeing if OUTCAR is sufficiently big
			OUTCAR_size=$(stat -c%s OUTCAR)
			if (( OUTCAR_size < OUTCAR_size_l_limit__Bytes )); then
				echo "WARNING: OUTCAR size < $OUTCAR_size_l_limit_MB MB"
				echo "Seems the job before ${master_id}${letter_old} did not finish correctly. Terminating this script at" `date `
				break
			fi

			# check the number of steps taken by the last job
			source data_4_analysis.sh
			run_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
			if (( run_time_steps < $MINIMUM_TIME_STEP_THRESHOLD )); then
				echo "WARNING: run_time_steps < $MINIMUM_TIME_STEP_THRESHOLD"
				echo "Seems the job before ${master_id}${letter_old} did not finish correctly. Terminating this script at" `date `
				break
			fi

			(( counter+=1 ))

			# make new job directory and submit job
			# if counter is 1 and restart mode > 0
			if (( counter == 1 && RESTART_MODE > 0 )); then
				echo "RESTART_MODE is set to ${RESTART_MODE} and counter is $counter. Copying SCALEE_${RESTART_MODE} as the initial condition for the first job."
				cp -r ../SCALEE_${RESTART_MODE} ../"${master_id}${letter}"
				cp ../"${master_id}${letter_old}"/INCAR ../"${master_id}${letter}"/ || {
					echo "INCAR file not found in ${master_id}${letter_old}. Exiting."
					exit 1
				}
			else
				echo "RESTART_MODE is set to ${RESTART_MODE} and counter is $counter. Copying the last job as the initial condition for the next job."
				cp -r ../"${master_id}${letter_old}" ../"${master_id}${letter}"
				# if $master_id includes "SCALEE", then cp ../"${master_id}$"/INCAR ../"${master_id}${letter}"/
				if [[ $master_id == SCALEE* ]]; then
					cp ../"${master_id}${letter_old}"/INCAR ../"${master_id}${letter}"/ || {
						echo "INCAR file not found in ${master_id}${letter_old}. Exiting."
						exit 1
					}
				fi
			fi
			# cp -r ../"${master_id}${letter_old}" ../"${master_id}${letter}"
			cd ../"${master_id}${letter}" || exit 1  # Exit if the directory change fails
			# cp CONTCAR POSCAR
			run_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
			restart_shift=$(( run_time_steps * PERCENTAGE_RESTART_SHIFT / 100 )) # 20% of the run time steps
			python $HELP_SCRIPTS_vasp/continue_run_ase.py -r $restart_shift
			rm WAVECAR

			# replace the line with "Total time steps" in analysis/peavg.out with "Total time steps = -1" to mark that this is a new job
			sed -i "s/Total time steps.*/Total time steps = -1/" analysis/peavg.out

			# if RUN_VASP_TIME > 0, then copy the RUN_VASP.sh file from the master directory
			# else keep the old RUN_VASP.sh file
			if (( $RUN_VASP_TIME > 0 )); then
				cp $RUN_SCRIPT_FILE RUN_VASP.sh
				echo "Copying the new RUN_VASP.sh file"
			else
				echo "Keeping the old RUN_VASP.sh file"
			fi
			rm -f slurm* log* log*

			# cp $RUN_SCRIPT_FILE RUN_VASP.sh
			sbatch RUN_VASP.sh
			job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}')
			echo ""
			echo "------------------------"
			echo "Job ${master_id}${letter} (${job_id}; ${counter}) submitted at" `date `

			#after submission, now wait as the job gets finished
			while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id
			do 
				# echo "Job ${master_id}${letter} (${job_id}; ${counter}) under work"
				sleep 600
			done
			sleep 60
			echo "Job ${master_id}${letter} (${job_id}; ${counter}) done at" `date `

			letter_old=$letter
		done
	fi


	for letter in $(eval echo {$first_letter..$final_job_letter}); do

		# check if the last job completed properly by just seeing if OUTCAR is sufficiently big
		OUTCAR_size=$(stat -c%s OUTCAR)
		if (( OUTCAR_size < OUTCAR_size_l_limit__Bytes )); then
			if (( $letter == a)); then
				echo "WARNING: OUTCAR size < $OUTCAR_size_l_limit_MB MB"
				echo "Seems the job before ${master_id}${letter_old} did not finish correctly. Terminating this script at" `date `
			else
				echo "Seems the job before ${master_id}a${letter_old} did not finish correctly. Terminating this script at" `date `
			fi
			break
		fi

		# check the number of steps taken by the last job
		source data_4_analysis.sh
		run_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
		if (( run_time_steps < $MINIMUM_TIME_STEP_THRESHOLD )); then
			echo "WARNING: run_time_steps < $MINIMUM_TIME_STEP_THRESHOLD"
			echo "Seems the job before ${master_id}${letter_old} did not finish correctly. Terminating this script at" `date `
			break
		fi

		(( counter+=1 ))

		# make new job directory and submit job
		if (( counter == 1 && RESTART_MODE > 0 )); then
			echo "RESTART_MODE is set to ${RESTART_MODE} and counter is $counter. Copying SCALEE_${RESTART_MODE} as the initial condition for the first job."
			if (( $letter == a)); then
				cp -r ../SCALEE_${RESTART_MODE} ../"${master_id}a${letter}"
				cp ../"${master_id}${letter_old}"/INCAR ../"${master_id}a${letter}"/ || {
					echo "INCAR file not found in ${master_id}${letter_old}. Exiting."
					exit 1
				}
			else
				cp -r ../SCALEE_${RESTART_MODE} ../"${master_id}a${letter}"
				cp ../"${master_id}a${letter_old}"/INCAR ../"${master_id}a${letter}" || {
					echo "INCAR file not found in ${master_id}a${letter_old}. Exiting."
					exit 1
				}
			fi
		else
			echo "RESTART_MODE is set to ${RESTART_MODE} and counter is $counter. Copying the last job as the initial condition for the next job."
			if (( $letter == a)); then
				cp -r ../"${master_id}${letter_old}" ../"${master_id}a${letter}"
				# if $master_id includes "SCALEE", then cp ../"${master_id}$"/INCAR ../"${master_id}${letter}"/
				if [[ $master_id == SCALEE* ]]; then
					cp ../"${master_id}${letter_old}"/INCAR ../"${master_id}a${letter}"/
				fi
			else
				cp -r ../"${master_id}a${letter_old}" ../"${master_id}a${letter}"
				# if $master_id includes "SCALEE", then cp ../"${master_id}$"/INCAR ../"${master_id}${letter}"/
				if [[ $master_id == SCALEE* ]]; then
					cp ../"${master_id}a${letter_old}"/INCAR ../"${master_id}a${letter}"/
				fi
			fi
		fi

		cd ../"${master_id}a${letter}" || exit 1  # Exit if the directory change fails
		# cp CONTCAR POSCAR
		run_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
		#restart_shift=int(run_time_steps*0.2)
		restart_shift=$(( run_time_steps * PERCENTAGE_RESTART_SHIFT / 100 )) # 20% of the run time steps
		python $HELP_SCRIPTS_vasp/continue_run_ase.py -r $restart_shift 
		rm WAVECAR
		
		# replace the line with "Total time steps" in analysis/peavg.out with "Total time steps = -1" to mark that this is a new job
		sed -i "s/Total time steps.*/Total time steps = -1/" analysis/peavg.out
		
		# if RUN_VASP_TIME > 0, then copy the RUN_VASP.sh file from the master directory
		# else keep the old RUN_VASP.sh file
		if (( $RUN_VASP_TIME > 0 )); then
			cp $RUN_SCRIPT_FILE RUN_VASP.sh
			echo "Copying the new RUN_VASP.sh file"
		else
			echo "Keeping the old RUN_VASP.sh file"
		fi
		rm -f slurm* log* log*

		sbatch RUN_VASP.sh
		job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}')
		echo ""
		echo "------------------------"
		echo "Job ${master_id}a${letter} (${job_id}; ${counter}) submitted at" `date `

		#after submission, now wait as the job gets finished
		while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id
		do 
			# echo "Job ${master_id}${letter} (${job_id}; ${counter}) under work"
			sleep 600
		done
		sleep 60
		echo "Job ${master_id}a${letter} (${job_id}; ${counter}) done at" `date `

		letter_old=$letter

	done

#if jobs in the range aa-az
elif (( $extended_job_flag == 1 )); then

	for letter in $(eval echo {$start_job_letter..$final_job_letter}); do

		# check if the last job completed properly by just seeing if OUTCAR if sufficiently big
		OUTCAR_size=$(stat -c%s OUTCAR)
		if (( OUTCAR_size < OUTCAR_size_l_limit__Bytes )); then
			echo "WARNING: OUTCAR size < $OUTCAR_size_l_limit_MB MB"
			echo "Seems the job before ${master_id}a${letter_old} did not finish correctly. Terminating this script at" `date `
			break
		fi

		# check the number of steps taken by the last job
		source data_4_analysis.sh
		run_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
		if (( run_time_steps < $MINIMUM_TIME_STEP_THRESHOLD )); then
			echo "WARNING: run_time_steps < $MINIMUM_TIME_STEP_THRESHOLD"
			echo "Seems the job before ${master_id}${letter_old} did not finish correctly. Terminating this script at" `date `
			break
		fi

		(( counter+=1 ))

		# make new job directory and submit job
		# cp -r ../"${master_id}a${letter_old}" ../"${master_id}a${letter}"
		# if counter is 1 and restart mode > 0
		if (( counter == 1 && RESTART_MODE > 0 )); then
			echo "RESTART_MODE is set to ${RESTART_MODE} and counter is $counter. Copying SCALEE_${RESTART_MODE} as the initial condition for the first job."
			cp -r ../SCALEE_${RESTART_MODE} ../"${master_id}a${letter}"
			cp ../"${master_id}a${letter_old}"/INCAR ../"${master_id}a${letter}"/ || {
				echo "INCAR file not found in ${master_id}a${letter_old}. Exiting."
				exit 1
			}
		else
			echo "RESTART_MODE is set to ${RESTART_MODE} and counter is $counter. Copying the last job as the initial condition for the next job."
			cp -r ../"${master_id}a${letter_old}" ../"${master_id}a${letter}"
			# if $master_id includes "SCALEE", then cp ../"${master_id}$"/INCAR ../"${master_id}${letter}"/
			if [[ $master_id == SCALEE* ]]; then
				cp ../"${master_id}a${letter_old}"/INCAR ../"${master_id}a${letter}"/ || {
					echo "INCAR file not found in ${master_id}a${letter_old}. Exiting."
					exit 1
				}
			fi
		fi

		cd ../"${master_id}a${letter}" || exit 1  # Exit if the directory change fails
		# cp CONTCAR POSCAR
		run_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
		restart_shift=$(( run_time_steps * PERCENTAGE_RESTART_SHIFT / 100 )) # 20% of the run time steps
		python $HELP_SCRIPTS_vasp/continue_run_ase.py -r $restart_shift 
		rm WAVECAR
		
		# replace the line with "Total time steps" in analysis/peavg.out with "Total time steps = -1" to mark that this is a new job
		sed -i "s/Total time steps.*/Total time steps = -1/" analysis/peavg.out
		
		
		# if RUN_VASP_TIME > 0, then copy the RUN_VASP.sh file from the master directory
		# else keep the old RUN_VASP.sh file
		if (( $RUN_VASP_TIME > 0 )); then
			cp $RUN_SCRIPT_FILE RUN_VASP.sh
			echo "Copying the new RUN_VASP.sh file"
		else
			echo "Keeping the old RUN_VASP.sh file"
		fi
		rm -f slurm* log* log*

		sbatch RUN_VASP.sh
		job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}')
		echo ""
		echo "------------------------"
		echo "Job ${master_id}a${letter} (${job_id}; ${counter}) submitted at" `date `

		#after submission, now wait as the job gets finished
		while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id
		do 
			# echo "Job ${master_id}${letter} (${job_id}; ${counter}) under work"
			sleep 600
		done
		sleep 60
		echo "Job ${master_id}a${letter} (${job_id}; ${counter}) done at" `date `

		letter_old=$letter
	done


#if jobs in the range a-z
elif (( $extended_job_flag == -1 )); then

	for letter in $(eval echo {$start_job_letter..$final_job_letter}); do

		# check if the last job completed properly by just seeing if OUTCAR if sufficiently big
		OUTCAR_size=$(stat -c%s OUTCAR)
		if (( OUTCAR_size < OUTCAR_size_l_limit__Bytes )); then
			echo "WARNING: OUTCAR size < $OUTCAR_size_l_limit_MB MB"
			echo "Seems the job before ${master_id}${letter_old} did not finish correctly. Terminating this script at" `date `
			break
		fi

		# check the number of steps taken by the last job
		source data_4_analysis.sh
		run_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
		if (( run_time_steps < $MINIMUM_TIME_STEP_THRESHOLD )); then
			echo "WARNING: run_time_steps < $MINIMUM_TIME_STEP_THRESHOLD"
			echo "Seems the job before ${master_id}${letter_old} did not finish correctly. Terminating this script at" `date `
			break
		fi

		(( counter+=1 ))

		# make new job directory and submit job
		# cp -r ../"${master_id}${letter_old}" ../"${master_id}${letter}"
		# if counter is 1 and restart mode > 0
		if (( counter == 1 && RESTART_MODE > 0 )); then
			echo "RESTART_MODE is set to ${RESTART_MODE} and counter is $counter. Copying SCALEE_${RESTART_MODE} as the initial condition for the first job."
			cp -r ../SCALEE_${RESTART_MODE} ../"${master_id}${letter}"
			cp ../"${master_id}${letter_old}"/INCAR ../"${master_id}${letter}" || {
				echo "INCAR file not found in ${master_id}${letter_old}. Exiting."
				exit 1
			}
		else
			echo "RESTART_MODE is set to ${RESTART_MODE} and counter is $counter. Copying the last job as the initial condition for the next job."
			cp -r ../"${master_id}${letter_old}" ../"${master_id}${letter}"
			# if $master_id includes "SCALEE", then cp ../"${master_id}$"/INCAR ../"${master_id}${letter}"/
			if [[ $master_id == SCALEE* ]]; then
				cp ../"${master_id}${letter_old}"/INCAR ../"${master_id}${letter}"/ || {
					echo "INCAR file not found in ${master_id}${letter_old}. Exiting."
					exit 1
				}
			fi
		fi

		cd ../"${master_id}${letter}" || exit 1  # Exit if the directory change fails
		# cp CONTCAR POSCAR
		run_time_steps=$(grep time analysis/peavg.out | awk '{print $5}')
		restart_shift=$(( run_time_steps * PERCENTAGE_RESTART_SHIFT / 100 )) # 20% of the run time steps
		python $HELP_SCRIPTS_vasp/continue_run_ase.py -r $restart_shift
		rm WAVECAR

		# replace the line with "Total time steps" in analysis/peavg.out with "Total time steps = -1" to mark that this is a new job
		sed -i "s/Total time steps.*/Total time steps = -1/" analysis/peavg.out
		
		# if RUN_VASP_TIME > 0, then copy the RUN_VASP.sh file from the master directory
		# else keep the old RUN_VASP.sh file
		if (( $RUN_VASP_TIME > 0 )); then
			cp $RUN_SCRIPT_FILE RUN_VASP.sh
			echo "Copying the new RUN_VASP.sh file"
		else
			echo "Keeping the old RUN_VASP.sh file"
		fi
		rm -f slurm* log* log*

		sbatch RUN_VASP.sh
		job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}')
		echo ""
		echo "------------------------"
		echo "Job ${master_id}${letter} (${job_id}; ${counter}) submitted at" `date `

		#after submission, now wait as the job gets finished
		while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id
		do 
			# echo "Job ${master_id}${letter} (${job_id}; ${counter}) under work"
			sleep 600
		done
		sleep 60
		echo "Job ${master_id}${letter} (${job_id}; ${counter}) done at" `date `

		letter_old=$letter
	done
fi


cd ..


echo ""
echo ""
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Job series for ${master_id} finsihed after ${counter} simulations at" `date `
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo ""
echo ""

cd $base_dir
touch done_RUN_VASP_MASTER_extended__SCALEE
##########################################################
