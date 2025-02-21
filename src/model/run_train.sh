#!/bin/bash

# List of config paths
config_paths=(
    "/path/to/config1.yaml"
    "/path/to/config2.yaml"
    "/path/to/config3.yaml"
)

# Loop through each config path and run rna_train.py in the background
for config_path in "${config_paths[@]}"; do
    python /home/rsun@ZHANGroup.local/sr_project/src/model/rna_train.py "$config_path" &
done

# Wait for all background processes to complete
wait