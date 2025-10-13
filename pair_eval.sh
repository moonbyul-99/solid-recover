#!/bin/bash

# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case1'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case3'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251012_1650'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_0338'
OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case3_20251013_0339'
# === 执行 Python 脚本 ===
echo "▶️  Starting evaluation sr_pair scratch model in output dir: $OUTPUT_DIR"

python pair_eval.py \
    --output_dir "$OUTPUT_DIR" \

echo "✅ Training completed."