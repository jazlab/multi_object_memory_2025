#!/bin/bash

#SBATCH -o ./slurm_logs/%A/%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --partition=use-everything

CONFIG_NAME="triangle_switching"
LOG_DIR_NAME="controls/triangle_switching_dmfc"

# Print hostname
HOSTNAME=$(hostname)
echo "node ${HOSTNAME}"

# Make a list of (subject, session) pairs to run
SUBJECTS_SESSIONS_DATA=(
  "Elgar 2022-08-20 0.16 2100"
  "Elgar 2022-08-24 0.18 2000"
  "Elgar 2022-09-03 0.14 2200"
  "Elgar 2022-09-04 0.1 1400"
  "Perle 2022-05-31 0.06 7800"
  "Perle 2022-06-01 0.06 9800"
  "Perle 2022-06-03 0.06 2300"
  "Perle 2022-06-04 0.06 1400"
)

BRAIN_AREAS="[\"DMFC\"]"
QUALITIES="[\"good\",\"mua\"]"

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
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"dataset\",\"kwargs\",\
      \"unit_filter\",\"kwargs\",\"brain_areas\"],\
      \"value\":${BRAIN_AREAS}},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"dataset\",\"kwargs\",\
      \"unit_filter\",\"kwargs\",\"qualities\"],\
      \"value\":${QUALITIES}},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"position_noise_sigma\"],\
      \"value\":0.05},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"smoothness_attention_ms\"],\
      \"value\":100},\
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
