#!/bin/bash


# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case1.yaml"
CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case2.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case3.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case4.yaml"
# === 执行 Python 脚本 ===
echo "▶️  Starting training sr_pair scratch model with config: $CONFIG_PATH"

python pair_scratch.py \
    --config "$CONFIG_PATH" 

echo "✅ Training completed."