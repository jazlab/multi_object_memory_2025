#!/bin/bash

NUM_JOBS=80

# Get user confirmation to launch the array
echo "Ready to submit ${NUM_JOBS} jobs? \
Press 'y' and Enter to continue."
read confirmation
if [[ ! $confirmation =~ ^[Yy]$ ]]
then
  echo "Launch canceled."
  exit
fi

# Submit job array to SLURM
echo "Submitting job array..."
array_launch=$(\
  sbatch --array=0-$((NUM_JOBS - 1)) task_triangle_gain_dmfc.sh \
  $CONFIG_NAME $LOG_DIR_NAME\
)
echo "Job array submitted."
job_id=${array_launch##*' '}
echo "Launched job ${job_id}"

# Create directory for logs
mkdir -p slurm_logs/${job_id}

# Write config  name to a file
echo ${CONFIG_NAME} > slurm_logs/${job_id}/config_name.log

# Write log_dir to a file
echo ${LOG_DIR_NAME} > slurm_logs/${job_id}/log_dir_name.log
