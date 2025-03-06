#!/bin/bash

# List of config paths
config_paths=(
    "/home/rsun@ZHANGroup.local/sr_project/configs/paired_configs/config_scratch.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/paired_configs/config_3.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/paired_configs/config_4.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/config_5.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/config_6.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/config_7.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/config_8.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/config_9.yaml"
)

# Loop through each config path and run rna_train.py in the background
for config_path in "${config_paths[@]}"; do
    config_name=$(basename "$config_path" .yaml)
    output_file="${config_name}.out"
    python /home/rsun@ZHANGroup.local/sr_project/src/pair_train_scratch.py "$config_path" > "$output_file" 2>&1 &
done

# Wait for all background processes to complete
wait