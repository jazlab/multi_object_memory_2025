#!/bin/bash

#SBATCH -o ./slurm_logs/%A/logs.out
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --partition=use-everything

MODEL_NAME=$1
LOG_DIR_NAME=$2

# Print hostname
HOSTNAME=$(hostname)
echo "node ${HOSTNAME}"

# Make a list of (subject, session, seed) tuples to run
SUBJECTS_SESSIONS_SEEDS=(
  "Elgar 2022-08-19 2"
  "Elgar 2022-08-20 2"
  "Elgar 2022-08-22 2"
  "Elgar 2022-08-24 2"
  "Elgar 2022-09-04 2"
  "Perle 2022-05-30 2"
  "Perle 2022-05-31 2"
  "Perle 2022-06-01 2"
  "Perle 2022-06-03 2"
  "Perle 2022-06-04 2"
)

# Get the task ID
slurm_array_task_id=${SLURM_ARRAY_TASK_ID}

# Activate conda environment
source ~/.bashrc
conda activate wm_paper
cd ../../training

# Iterate through parameters
for ((i = 0; i < ${#SUBJECTS_SESSIONS_SEEDS[@]}; i++)); do
  subject_session_seed=${SUBJECTS_SESSIONS_SEEDS[$i]}

  # Extract the subject and session and seed by splitting on spaces
  subject=$(echo $subject_session_seed | cut -d ' ' -f 1)
  session=$(echo $subject_session_seed | cut -d ' ' -f 2)
  seed=$(echo $subject_session_seed | cut -d ' ' -f 3)

  # Make config overrides
  model_log_dir="../../cache/modeling/main/${MODEL_NAME}/${subject}/${session}/${seed}"
  config_overrides="[\
  {\"node\":[\"kwargs\",\"log_dir\"],\"value\":\"${model_log_dir}\"}\
  ]"
  echo $config_overrides

  # Make log directory
  log_dir="../../cache/modeling/${LOG_DIR_NAME}/${subject}/${session}/${seed}"
  echo "log_dir: ${log_dir}"

  # Remove log directory if it exists
  if [ -d "${log_dir}" ]; then
    echo "Log directory exists. Removing."
    rm -r ${log_dir}
  fi

  # Run run_evaluation.py
  python3 run_training.py \
    --config="configs.generate_synthetic.config" \
    --config_overrides="${config_overrides}" \
    --log_directory="${log_dir}" \
    --metadata="Task ID: ${SLURM_ARRAY_TASK_ID}"
  echo "Finished running run_training.py"
done
