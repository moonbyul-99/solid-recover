#!/bin/bash

# List of config paths
config_paths=(
    #'/home/rsun@ZHANGroup.local/sr_project/configs/scratch_eval_configs/config_128.yaml'
    '/home/rsun@ZHANGroup.local/sr_project/configs/scratch_eval_configs/config_256.yaml'
    '/home/rsun@ZHANGroup.local/sr_project/configs/scratch_eval_configs/config_512.yaml'
    #'/home/rsun@ZHANGroup.local/sr_project/configs/scratch_eval_configs/config_2048.yaml'
)

# Loop through each config path and run rna_train.py in the background
for config_path in "${config_paths[@]}"; do
    config_name=$(basename "$config_path" .yaml)
    output_file="${config_name}.eval_out"
    python /home/rsun@ZHANGroup.local/sr_project/src/sr_scratch_eval.py --config_path "$config_path" > "$output_file" 2>&1 &
done

# Wait for all background processes to complete
wait