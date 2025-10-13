#!/bin/bash

# run_eval.sh - Bash script to run evaluation with config

# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case1.yaml'
# CKPT_PATH='/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case1/models/ckpt_2000.pth'
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/evaluation/cross_prediction_case1"

# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case2.yaml'
# CKPT_PATH='/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case2.yaml_20250922_0829/models/ckpt_8000.pth'
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/evaluation/cross_prediction_case2"

# CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case3.yaml'
# CKPT_PATH='/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case3_20250922_1553/models/ckpt_8000.pth'
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/evaluation/cross_prediction_case3"

CONFIG_PATH='/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case4.yaml'
CKPT_PATH='/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case4_v1/models/ckpt_9000.pth'
EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/evaluation/cross_prediction_case4"


# === 执行 Python 脚本 ===
echo "▶️  Starting evaluation with config: $CONFIG_PATH"
echo "    Ckpt dir: $CKPT_PATH"
echo "    Output dir: $EVAL_RESULT_DIR"

python eval_prediction.py \
    --config "$CONFIG_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --eval_dir "$EVAL_RESULT_DIR"

echo "✅ Evaluation completed. Results saved to: $EVAL_RESULT_DIR"