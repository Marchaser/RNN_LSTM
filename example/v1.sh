#!/bin/bash
#
# Execute the job from the current directory.
#$ -cwd
#
# Merge the standard error with the standard output.
#$ -j y
#
# Maximum runtime/wallclock. Please change it if your job requires more 
# than the system default, 24 hours.
#$ -l h_rt=72:00:00
#
# Maximum virtual memory per CPU core. Please change it if your job 
# requires more than the system default, 3G.
#$ -l h_vmem=3G
#
#$ -N lstm_train
#
 
# Set up proper modules.
. /etc/profile
module load matlab/R2014b
module load intel/2015u3
module list
 
# Run your batch/serial job.
matlab -nojvm -nodisplay -r "run ./Asimov" > train.txt 
 
exit 0

