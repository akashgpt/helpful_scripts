############################################################################
# RUN_VASP_MASTER.sh     vasp submit script                                #
# 
# Script created to automate submission of multiple jobs one after another #
############################################################################
#!/bin/bash

master_id=$(basename "$PWD")

if [[ $# -eq 0 ]]; then
	num_jobs=5
else
	num_jobs=$1
fi
echo "Job series for ${master_id} started for ${num_jobs} simulations at" `date `

OUTCAR_size_l_limit=100000000 #1E8 Bytes = 100 MB
# OUTCAR_size_l_limit=100000 #1E5 Bytes = 100 KB


# for converting decimal to ASCII
chr() {
  printf \\$(printf '%03o' $1)
}

# for converting ASCII to decimal
ord() {
  printf '%d' "'$1"
}


# check which is the last job completed
for letter in {a..z}
do
if [ -d "../${master_id}${letter}" ]; then
	last_job_id_letter=$letter
fi
done

cd ../"${master_id}${last_job_id_letter}"/
echo "Last job record found: ${master_id}${last_job_id_letter}"

start_job_ascii=$(ord $last_job_id_letter); (( start_job_ascii+=1 ))
start_job_letter=$(chr $start_job_ascii)
final_job_ascii=$(ord $last_job_id_letter); (( final_job_ascii+=$num_jobs ))
final_job_letter=$(chr $final_job_ascii)

letter_old=$last_job_id_letter
counter=0

# start new jobs
for letter in $(eval echo {$start_job_letter..$final_job_letter}); do

	# check if the last job completed properly by just seeing if OUTCAR if sufficiently big
	OUTCAR_size=$(stat -c%s OUTCAR)
	if (( OUTCAR_size < OUTCAR_size_l_limit )); then
		echo "Seems the job before ${master_id}${letter_old} did not finish correctly. Terminating this script at" `date `
		break
	fi

	(( counter+=1 ))


	# make new job directory and submit job
	cp -r ../"${master_id}${letter_old}" ../"${master_id}${letter}"
	cd ../"${master_id}${letter}"/
	cp CONTCAR POSCAR
	cp ../run_scripts/RUN_VASP_23h.sh RUN_VASP.sh
	qsub RUN_VASP.sh
	job_id=$(myjobs | tail -n 1 | awk '{print $1}')
	echo ""
	echo "------------------------"
	echo "Job ${master_id}${letter} (${job_id}; ${counter}) submitted at" `date `

	#after submission, now wait as the job gets finished
	while  myjobs | grep -q $job_id
	do 
		# echo "Job ${master_id}${letter} (${job_id}; ${counter}) under work"
		sleep 600
	done
	sleep 60
	echo "Job ${master_id}${letter} (${job_id}; ${counter}) done at" `date `

	letter_old=$letter
done


cd ..


echo ""
echo ""
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Job series for ${master_id} finished after ${counter} simulations at" `date `
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo ""
echo ""
##########################################################
