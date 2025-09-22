#!/bin/bash

# run_eval.sh - Bash script to run evaluation with config


# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case2.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case2.yaml_20250922_0829/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/eval_result/pair_scratch_case2.yaml_20250922_0829"

# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case3.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case3_20250922_0829/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_case3_20250922_0829"

CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case1.yaml"
EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/sr_pair_demo/models"
EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/sr_pair_demo_case1"

# === 执行 Python 脚本 ===
echo "▶️  Starting evaluation with config: $CONFIG_PATH"
echo "    Model dir: $EVAL_DIR"
echo "    Output dir: $EVAL_RESULT_DIR"

python eval_pair_scratch.py \
    --config "$CONFIG_PATH" \
    --eval_dir "$EVAL_DIR" \
    --eval_result_dir "$EVAL_RESULT_DIR"

echo "✅ Evaluation completed. Results saved to: $EVAL_RESULT_DIR"