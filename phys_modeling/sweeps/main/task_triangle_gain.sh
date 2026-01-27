#!/bin/bash

#SBATCH -o ./slurm_logs/%A/%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --partition=use-everything

CONFIG_NAME="triangle_gain"
LOG_DIR_NAME="main/triangle_gain"

# Print hostname
HOSTNAME=$(hostname)
echo "node ${HOSTNAME}"

# Make a list of (subject, session) pairs to run
SUBJECTS_SESSIONS_DATA=(
  "Elgar 2022-08-19 0.16 9800"
  "Elgar 2022-08-20 0.18 9200"
  "Elgar 2022-08-21 0.14 4200"
  "Elgar 2022-08-22 0.1 5900"
  "Elgar 2022-08-23 0.16 4400"
  "Elgar 2022-08-24 0.16 9000"
  "Elgar 2022-08-25 0.12 1100"
  "Elgar 2022-08-26 0.14 3100"
  "Elgar 2022-08-31 0.08 8600"
  "Elgar 2022-09-01 0.16 9900"
  "Elgar 2022-09-02 0.12 1100"
  "Elgar 2022-09-03 0.14 2500"
  "Elgar 2022-09-04 0.12 1100"
  "Elgar 2022-09-05 0.02 1100"
  "Perle 2022-05-26 0.08 2400"
  "Perle 2022-05-27 0.06 6100"
  "Perle 2022-05-28 0.08 6000"
  "Perle 2022-05-29 0.08 1100"
  "Perle 2022-05-30 0.08 9200"
  "Perle 2022-05-31 0.06 5300"
  "Perle 2022-06-01 0.06 9200"
  "Perle 2022-06-03 0.08 9400"
  "Perle 2022-06-04 0.08 4200"
  "Perle 2022-06-05 0.1 2500"
)

# Make list of random seeds
RANDOM_SEEDS=(2 3 4 5 6 7 8 9 10 11)

# Get the task ID
slurm_array_task_id=${SLURM_ARRAY_TASK_ID}

# Iterate through parameters
count=-1
for ((i = 0; i < ${#SUBJECTS_SESSIONS_DATA[@]}; i++)); do
  subject_session_data=${SUBJECTS_SESSIONS_DATA[$i]}

  # Extract the specs for this run
  subject=$(echo $subject_session_data | cut -d ' ' -f 1)
  session=$(echo $subject_session_data | cut -d ' ' -f 2)
  time_noise_sigma=$(echo $subject_session_data | cut -d ' ' -f 3)
  step=$(echo $subject_session_data | cut -d ' ' -f 4)

  for random_seed in ${RANDOM_SEEDS[@]}; do
    count=$((count + 1))
    if [[ $count -eq $slurm_array_task_id ]]; then
      # Activate conda environment
      source ~/.bashrc
      conda activate wm_paper
      cd ../../training

      # Make config overrides
      config_overrides="[\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"dataset\",\"kwargs\",\"subject\"],\
      \"value\":\"${subject}\"},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"dataset\",\"kwargs\",\"session\"],\
      \"value\":\"${session}\"},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"dataset\",\"kwargs\",\"random_seed\"],\
      \"value\":${random_seed}},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"sigmoid_attention\"],\
      \"value\":\"true\"},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"position_noise_sigma\"],\
      \"value\":0.05},\
      {\"node\":[\"kwargs\",\"start_optimizing_per_trial\"],\
      \"value\":1000},\
      {\"node\":[\"kwargs\",\"lr_embedding\"],\
      \"value\":0.003},\
      {\"node\":[\"kwargs\",\"lr_per_trial\"],\
      \"value\":0.003},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"time_noise_sigma\"],\
      \"value\":${time_noise_sigma}},\
      {\"node\":[\"kwargs\",\"stop_step\"],\
      \"value\":${step}}\
      ]"
      echo $config_overrides

      # Make log directory
      log_dir="../../cache/modeling/${LOG_DIR_NAME}/${subject}/${session}/${random_seed}"
      echo "log_dir: ${log_dir}"

      # If already ran, skip this run
      if [ -f "${log_dir}/test_metrics.json" ]; then
        echo "Already ran. Skipping run."
        continue
      fi

      # Remove log directory if it exists
      if [ -d "${log_dir}" ]; then
        echo "Log directory exists. Removing."
        rm -r ${log_dir}
      fi

      # Run run_evaluation.py
      python3 run_training.py \
        --config="configs.neural.${CONFIG_NAME}" \
        --config_overrides="${config_overrides}" \
        --log_directory="${log_dir}" \
        --metadata="Task ID: ${SLURM_ARRAY_TASK_ID}"
      echo "Finished running run_training.py"
    fi
  done
done
