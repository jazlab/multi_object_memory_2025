#!/bin/bash

NUM_JOBS=660

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
  sbatch --array=0-$((NUM_JOBS - 1)) task_triangle_gain.sh
)
echo "Job array submitted."
job_id=${array_launch##*' '}
echo "Launched job ${job_id}"

# Create directory for logs
mkdir -p slurm_logs/${job_id}
