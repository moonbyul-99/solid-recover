#!/bin/bash

# List of config paths
config_paths=(
    "/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_2_large_sc_new.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_1_large_pretrain.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_2_large_pretrain.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_3_large_pretrain.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_4_large_pretrain.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_1_large_scratch.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_2_large_scratch.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_3_large_scratch.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_4_large_scratch.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_1_small_pretrain.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_2_small_pretrain.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_3_small_pretrain.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_4_small_pretrain.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_1_small_scratch.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_2_small_scratch.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_3_small_scratch.yaml"
    #"/home/rsun@ZHANGroup.local/sr_project/configs/pair_eval_configs/config_4_small_scratch.yaml"
)

# Loop through each config path and run rna_train.py in the background
for config_path in "${config_paths[@]}"; do
    config_name=$(basename "$config_path" .yaml)
    output_file="${config_name}.eval_out"
    python /home/rsun@ZHANGroup.local/sr_project/src/sr_eval.py --config_path "$config_path" > "$output_file" 2>&1 &
done

# Wait for all background processes to complete
wait