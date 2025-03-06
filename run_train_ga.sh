#!/bin/bash

# List of config paths
config_paths=(
    "/home/rsun@ZHANGroup.local/sr_project/configs/ga_configs/config_small.yaml"
    "/home/rsun@ZHANGroup.local/sr_project/configs/ga_configs/config_large.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/ga_configs/config_3.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/ga_configs/config_4.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/ga_configs/config_5.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/ga_configs/config_6.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/config_7.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/config_8.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/config_9.yaml"
)

# Loop through each config path and run rna_train.py in the background
for config_path in "${config_paths[@]}"; do
    config_name=$(basename "$config_path" .yaml)
    output_file="${config_name}.gaout"
    python /home/rsun@ZHANGroup.local/sr_project/src/ga_train.py "$config_path" > "$output_file" 2>&1 &
done

# Wait for all background processes to complete
wait