#!/bin/bash

# run_eval.sh - Bash script to run evaluation with config


# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case1.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case1/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_case1"


# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case2.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case2/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_case2"

# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case3.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case3_20250922_1441/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_case3_20250922_1441"


# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case4.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case4_20250922_1652/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_case4_20250922_1652"

# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case4.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case4_20250923_0357/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_case4_20250923_0357"

# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_kidney.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_kidney/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_kidney"

# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_kidney_v1.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_kidney_v1/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_kidney_v1"

# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case4_v1.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case4_v1/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_case4_v1"

# CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case4_v2.yaml"
# EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case4_v1_20250923_1522/models"
# EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_case4_v2"

CONFIG_PATH="/home/rsun@ZHANGroup.local/solid-recover/configs/pair_scratch_case4_v4.yaml"
EVAL_DIR="/home/rsun@ZHANGroup.local/solid-recover/runs/pair_scratch_case4_v4/models"
EVAL_RESULT_DIR="/home/rsun@ZHANGroup.local/solid-recover/eval_result/pair_scratch_case4_v4"


# === 执行 Python 脚本 ===
echo "▶️  Starting evaluation with config: $CONFIG_PATH"
echo "    Model dir: $EVAL_DIR"
echo "    Output dir: $EVAL_RESULT_DIR"

python eval_pair_scratch.py \
    --config "$CONFIG_PATH" \
    --eval_dir "$EVAL_DIR" \
    --eval_result_dir "$EVAL_RESULT_DIR"

echo "✅ Evaluation completed. Results saved to: $EVAL_RESULT_DIR"