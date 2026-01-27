#!/bin/bash

MODEL_NAME="triangle_gain"
LOG_DIR_NAME="synthetic_datasets/triangle_gain"

# Submit job array to SLURM
launch=$(sbatch task_triangle.sh $MODEL_NAME $LOG_DIR_NAME)
echo "Job submitted."
job_id=${launch##*' '}
echo "Launched job ${job_id}"

# Create directory for logs
mkdir -p slurm_logs/${job_id}

# Write log_dir to a file
echo ${LOG_DIR_NAME} > slurm_logs/${job_id}/log_dir_name.log
