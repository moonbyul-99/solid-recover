#!/bin/bash


# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case1.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case2.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case3.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case4.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case5.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case6.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case8.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case2.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/case1_wc.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/case10_wc.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/case9_wc.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/case8_wc.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/case8_wc.yaml"
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case11_wc.yaml'
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case14_wc.yaml'
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case8_wc.yaml'
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case16_al.yaml'
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/three_modal.yaml'
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/pertfate.yaml'
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case_renal.yaml'
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case_brain.yaml'
CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case_mus_kidney.yaml'
TO_GPU=False
# if True, move all data to GPU, this will accelerate training because do not need to move data to GPU every iteration, but will consume more GPU memory 
# without effective training. if False, data 
# 

# === 执行 Python 脚本 ===
echo "▶️  Starting training sr_pair scratch model with config: $CONFIG_PATH"
echo "     to_gpu: $TO_GPU"

# python pair_scratch.py \
#     --config "$CONFIG_PATH" 


python pair_scratch.py \
    --config "$CONFIG_PATH" \
    --to_gpu "$TO_GPU"

echo "✅ Training completed."