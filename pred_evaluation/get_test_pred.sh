#!/bin/bash

# 遇到错误立即停止 (可选)
set -e 

echo "Starting batch process..."

echo "Running case_8..."
python get_sr_pred.py \
    --case_id "case_8" \
    --output_dir '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case8_wc_new_20251102_0359' \
    --sr_ckpt 4000 \
    --device "cuda"

echo "Running case_9..."
python get_sr_pred.py \
    --case_id "case_9" \
    --output_dir '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case9_wc_20251101_1439' \
    --sr_ckpt 1500 \
    --device "cuda"

echo "Running case_11..."
python get_sr_pred.py \
    --case_id "case_11" \
    --output_dir '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case11_wc_20251104_1421' \
    --sr_ckpt 2000 \
    --device "cuda"

echo "Running case_12..."
python get_sr_pred.py \
    --case_id "case_12" \
    --output_dir '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case12_wc' \
    --sr_ckpt 2500 \
    --device "cuda"

echo "All tasks finished."