##########################################################################
# RUN_VASP.sh     vasp submit script                                     #
#                                                                        #
# FILE CREATED ON: 2018NOV30                                             #
# LAST EDITED: 2018NOV30                                                 #
# CONTACT: RD (hpc@ucla.edu)                                             #
#                                                                        #
# This script assumes that you have built vasp with intel/18.0.3         #
#                                                                        #
# VASP_DIR should reflect the location of your VASP binary               #
#                                                                        #
# change the name of the vasp binary as needed                           #
#                                                                        #
# change h_rt (run-time), h_data (memory/core) as needed                 #
#                                                                        #
# remove exclusive and/or highp as needed                                #
#                                                                        #
# N.B.: only use highp if sponsor has contributed nodes otherwise the    #
#       job will NEVER start                                             #
#                                                                        #
# change the number after dc* to change the number of parallel workers   #
#                                                                        #
# make sure that this script is executable before submitting:            #
# run: chmod u+x RUN_VASP.sh                                             #
#                                                                        #
# submit your vasp job with:                                             #
# qsub RUN_VASP.sh                                                       #
#                                                                        #
##########################################################################
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
#$ -l arch=intel-gold*,h_rt=23:00:00,h_data=1G,exclusive
#$ -pe dc* 36 
# Email address to notify
##$ -M $USER@mail
# Notify when
##$ -m bea

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo "Job $JOB_ID will run on:"
cat $PE_HOSTFILE
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
#module load intel/18.0.3
#module load intel
module load intel/2020.4
module li
pwd
VASP_DIR=/u/project/ESS/lstixrud/lstixrud/vasp/vasp.5.4.4/bin/

# debug:
/usr/bin/time -v `which mpirun` -env I_MPI_DEBUG=10 -n $NSLOTS hostname > ./output.$JOB_ID 2>&1
# run VAPS:
/usr/bin/time -v `which mpirun` -env I_MPI_DEBUG=10 -n $NSLOTS $VASP_DIR/vasp_std >> ./output.$JOB_ID 2>&1

echo "Job $JOB_ID done at:   " `date `
##########################################################

