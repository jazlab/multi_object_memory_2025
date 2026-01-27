#!/bin/bash

#SBATCH -o ./slurm_logs/%A/%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --partition=use-everything

CONFIG_NAME="triangle_switching"
LOG_DIR_NAME="controls_hyperparams/triangle_switching_good"

# Print hostname
HOSTNAME=$(hostname)
echo "node ${HOSTNAME}"

# Constants
NUM_TIME_NOISE_SIGMA_STEPS=11
RANDOM_SEEDS=(0 1)

# Make a list of (subject, session) pairs to run
SUBJECTS_SESSIONS_TIME_NOISE_SIGMA=(
  "Elgar 2022-08-20 0.0"  #
  "Elgar 2022-08-24 0.0"  #
  "Elgar 2022-09-03 0.0"  #
  "Elgar 2022-09-04 0.0"  #
  "Perle 2022-05-31 0.0"  #
  "Perle 2022-06-01 0.0"  #
  "Perle 2022-06-03 0.0"  #
  "Perle 2022-06-04 0.0"  #
)

BRAIN_AREAS="[\"DMFC\",\"FEF\"]"
QUALITIES="[\"good\"]"

# Get the task ID
slurm_array_task_id=${SLURM_ARRAY_TASK_ID}
# Iterate through parameters
count=-1
for ((i = 0; i < ${#SUBJECTS_SESSIONS_TIME_NOISE_SIGMA[@]}; i++)); do
  subject_session_time_noise_sigma=${SUBJECTS_SESSIONS_TIME_NOISE_SIGMA[$i]}

  # Extract the specs for this run
  subject=$(echo $subject_session_time_noise_sigma | cut -d ' ' -f 1)
  session=$(echo $subject_session_time_noise_sigma | cut -d ' ' -f 2)
  min_time_noise_sigma=$(echo $subject_session_time_noise_sigma | cut -d ' ' -f 3)

  # Make time_noise_sigma array by starting at min_time_noise_sigma and going up in steps of 50
  time_noise_sigma_array=()
  for ((j = 0; j < ${NUM_TIME_NOISE_SIGMA_STEPS}; j++)); do
    time_noise_sigma_array+=($(echo "$min_time_noise_sigma + $j * 0.02" | bc -l))
  done

  for time_noise_sigma in ${time_noise_sigma_array[@]}; do
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
        {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"smoothness_attention_ms\"],\
        \"value\":100},\
        {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"position_noise_sigma\"],\
        \"value\":0.05},\
        {\"node\":[\"kwargs\",\"start_optimizing_per_trial\"],\
        \"value\":1000},\
        {\"node\":[\"kwargs\",\"lr_embedding\"],\
        \"value\":0.003},\
        {\"node\":[\"kwargs\",\"lr_per_trial\"],\
        \"value\":0.003},\
        {\"node\":[\"kwargs\",\"model\",\"kwargs\",\"time_noise_sigma\"],\
        \"value\":${time_noise_sigma}}\
        ]"
        echo $config_overrides

        # Make log directory
        log_dir="../../cache/modeling/${LOG_DIR_NAME}/${subject}/${session}/${random_seed}/${time_noise_sigma}"
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
done
