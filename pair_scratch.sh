#!/bin/bash


# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case1.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case2.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case3.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case4.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case5.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case6.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case8.yaml"
# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case2.yaml"
CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/case4_wc.yaml"
TO_GPU=True

# === 执行 Python 脚本 ===
echo "▶️  Starting training sr_pair scratch model with config: $CONFIG_PATH"
echo "     to_gpu: $TO_GPU"

# python pair_scratch.py \
#     --config "$CONFIG_PATH" 


python pair_scratch.py \
    --config "$CONFIG_PATH" \
    --to_gpu "$TO_GPU"

echo "✅ Training completed."