#!/bin/bash

#SBATCH -o ./slurm_logs/%A/%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --partition=use-everything

CONFIG_NAME="triangle_gain"
LOG_DIR_NAME="synthetic"

# Print hostname
HOSTNAME=$(hostname)
echo "node ${HOSTNAME}"

# Make list of random seeds
RANDOM_SEEDS=(3 4 5 6 7)

# Make a list of (subject, session) pairs to run
SUBJECTS_SESSIONS_DATA=(
  "triangle_gain Elgar 2022-08-19 0.16 8800"
  "triangle_gain Elgar 2022-08-20 0.2 9600"
  "triangle_gain Elgar 2022-08-22 0.12 8900"
  "triangle_gain Elgar 2022-08-24 0.2 9500"
  "triangle_gain Elgar 2022-09-04 0.12 1700"
  "triangle_gain Perle 2022-05-30 0.08 9700"
  "triangle_gain Perle 2022-05-31 0.08 9900"
  "triangle_gain Perle 2022-06-01 0.06 9600"
  "triangle_gain Perle 2022-06-03 0.1 9200"
  "triangle_gain Perle 2022-06-04 0.1 7600"

  "triangle_switching Elgar 2022-08-19 0.2 8000"
  "triangle_switching Elgar 2022-08-20 0.2 1900"
  "triangle_switching Elgar 2022-08-22 0.14 9300"
  "triangle_switching Elgar 2022-08-24 0.2 4400"
  "triangle_switching Elgar 2022-09-04 0.18 2200"
  "triangle_switching Perle 2022-05-30 0.1 9900"
  "triangle_switching Perle 2022-05-31 0.06 9000"
  "triangle_switching Perle 2022-06-01 0.06 9400"
  "triangle_switching Perle 2022-06-03 0.1 4800"
  "triangle_switching Perle 2022-06-04 0.08 2000"

  "triangle_slot_partition Elgar 2022-08-19 0.2 8800"
  "triangle_slot_partition Elgar 2022-08-20 0.2 9400"
  "triangle_slot_partition Elgar 2022-08-22 0.12 9400"
  "triangle_slot_partition Elgar 2022-08-24 0.2 9600"
  "triangle_slot_partition Elgar 2022-09-04 0.16 6900"
  "triangle_slot_partition Perle 2022-05-30 0.08 9800"
  "triangle_slot_partition Perle 2022-05-31 0.08 8100"
  "triangle_slot_partition Perle 2022-06-01 0.06 9600"
  "triangle_slot_partition Perle 2022-06-03 0.08 9500"
  "triangle_slot_partition Perle 2022-06-04 0.1 7400"
)

# Get the task ID
slurm_array_task_id=${SLURM_ARRAY_TASK_ID}

# Iterate through parameters
count=-1
for ((i = 0; i < ${#SUBJECTS_SESSIONS_DATA[@]}; i++)); do
  subject_session_data=${SUBJECTS_SESSIONS_DATA[$i]}

  # Extract the specs for this run
  synthetic_dataset=$(echo $subject_session_data | cut -d ' ' -f 1)
  subject=$(echo $subject_session_data | cut -d ' ' -f 2)
  session=$(echo $subject_session_data | cut -d ' ' -f 3)
  time_noise_sigma=$(echo $subject_session_data | cut -d ' ' -f 4)
  step=$(echo $subject_session_data | cut -d ' ' -f 5)

  for random_seed in ${RANDOM_SEEDS[@]}; do
    count=$((count + 1))
    if [[ $count -eq $slurm_array_task_id ]]; then
      # Activate conda environment
      source ~/.bashrc
      conda activate wm_paper
      cd ../../training

      # If time_noise_sigma starts with a decimal, prepend a zero
      if [[ $time_noise_sigma == .* ]]; then
        time_noise_sigma="0$time_noise_sigma"
      fi

      # Make config overrides
      snapshot_data_dir="../../cache/modeling/synthetic_datasets/${synthetic_dataset}/${subject}/${session}/2"
      config_overrides="[\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"dataset\",\"kwargs\",\"dataset\",\"kwargs\",\"subject\"],\
      \"value\":\"${subject}\"},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"dataset\",\"kwargs\",\"dataset\",\"kwargs\",\"session\"],\
      \"value\":\"${session}\"},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"dataset\",\"kwargs\",\"dataset\",\"kwargs\",\"random_seed\"],\
      \"value\":${random_seed}},\
      {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"dataset\",\"kwargs\",\"snapshot_data_dir\"],\
      \"value\":\"${snapshot_data_dir}\"},\
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
      log_dir="../../cache/modeling/${LOG_DIR_NAME}/${synthetic_dataset}/${CONFIG_NAME}/${subject}/${session}/${random_seed}"
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
        --config="configs.synthetic.${CONFIG_NAME}" \
        --config_overrides="${config_overrides}" \
        --log_directory="${log_dir}" \
        --metadata="Task ID: ${SLURM_ARRAY_TASK_ID}"
      echo "Finished running run_training.py"
    fi
  done
done
