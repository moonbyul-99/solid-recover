#!/bin/bash

# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case1'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case3'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251012_1650'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_0338'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case3_20251013_0339'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_0747'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_0754'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_1009'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_1011'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_1124'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251014_0134'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2_8dim'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2_11dim'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2_16dim'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2_32dim'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251014_0205'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2_64dim'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2_128dim'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case5'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case5_20251018_1208'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case5_20251018_1231'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case5_20251018_1246'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case5_20251018_1302'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case5_20251018_1311'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case5_20251018_1321'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case5_20251018_1329'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case6'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case8'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2_noclip'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2_nocross'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2_wc'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_wc'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_wc_20251029_0133'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_wc_20251029_0719'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_wc_20251029_1027'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case8_wc'
# OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case8_wc_noweight'
OUTPUT_DIR='/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case8_noclip_v1'
# === 执行 Python 脚本 ===
echo "▶️  Starting evaluation sr_pair scratch model in output dir: $OUTPUT_DIR"

python pair_eval.py \
    --output_dir "$OUTPUT_DIR" \

echo "✅ Training completed."