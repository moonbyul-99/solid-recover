#!/bin/bash


# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case4_align.yaml'
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case15_align.yaml'
# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case15_al.yaml'
CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/case16_pretrain.yaml'

TO_GPU=False
# if True, move all data to GPU, this will accelerate training because do not need to move data to GPU every iteration, but will consume more GPU memory 
# without effective training. if False, data 
# 

# === 执行 Python 脚本 ===
echo "▶️  Starting training sr_pair scratch model with config: $CONFIG_PATH"
echo "     to_gpu: $TO_GPU"

# python pair_scratch.py \
#     --config "$CONFIG_PATH" 


python pair_pretrain.py \
    --config "$CONFIG_PATH" \
    --to_gpu "$TO_GPU"

echo "✅ Training completed."