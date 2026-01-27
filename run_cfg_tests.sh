#!/bin/bash

# 设置通用参数
GPU=2
WIDTH=384
HEIGHT=384
SEED=42
MODEL="black-forest-labs/FLUX.1-dev"

# true_cfg_scale 值
CFG_SCALES=(1.0 3.5 5.0)

# 定义所有prompts
declare -A PROMPTS

# Test 1 系列
PROMPTS["test1_prompt1"]="Generating an uniform pure color image with hex color code: #9966CC."
PROMPTS["test1_prompt2"]="Generating an uniform pure color image with hex color code: #9966CC, Strictly no shadows, no gradients, and no metallic textures."
PROMPTS["test1_prompt3"]="Generating an uniform pure color image with hex color code: #9966CC, Strictly no cloud patterns or water ripples."

# 创建输出目录
mkdir -p cfg_scale_tests_2

# 遍历所有prompts
for test_name in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$test_name]}"
    echo "=========================================="
    echo "Running: $test_name"
    echo "Prompt: $prompt"
    echo "=========================================="
    
    # 每个prompt测试不同的cfg_scale
    for cfg in "${CFG_SCALES[@]}"; do
        output_file="cfg_scale_tests_2/${test_name}_cfg${cfg}.png"
        
        echo "  CFG Scale $cfg -> $output_file"
        
        CUDA_VISIBLE_DEVICES=$GPU python test_inference.py \
            --prompt "$prompt" \
            --output_image "$output_file" \
            --width $WIDTH \
            --height $HEIGHT \
            --seed $SEED \
            --model $MODEL \
            --true_cfg_scale $cfg \
            --negative_prompt ""
        
        echo "  Done."
    done
    echo ""
done

echo "All tests completed!"