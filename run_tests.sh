#!/bin/bash

# 设置通用参数
GPU=2
WIDTH=384
HEIGHT=384
SEEDS=(42 123 456)

# 定义所有prompts
declare -A PROMPTS

# Test 3 系列
PROMPTS["test3_prompt1"]="A split image with two solid color blocks: the left 50% is a pure solid color with hex code #AB1213, the right 50% is a pure solid color with hex code #000000"
PROMPTS["test3_prompt2"]="A split image with two solid color blocks: the left 31.5% is a pure solid color with hex code #AB1213, the right 68.5% is a pure solid color with hex code #000000"
PROMPTS["test3_prompt3"]="A common photographic composition with two solid color blocks: the left 33.3% is a pure solid color with hex code #AB1213, the right 66.7% is a pure solid color with hex code #000000"

# 创建输出目录
mkdir -p prompt_tests_2

# 遍历所有prompts
for test_name in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$test_name]}"
    echo "=========================================="
    echo "Running: $test_name"
    echo "Prompt: $prompt"
    echo "=========================================="
    
    # 每个prompt跑3次，使用不同的seed
    for i in {0..2}; do
        seed=${SEEDS[$i]}
        run_num=$((i + 1))
        output_file="prompt_tests_2/${test_name}_run${run_num}_seed${seed}.png"
        
        echo "  Run $run_num with seed $seed -> $output_file"
        
        CUDA_VISIBLE_DEVICES=$GPU python test_inference.py \
            --prompt "$prompt" \
            --output_image "$output_file" \
            --width $WIDTH \
            --height $HEIGHT \
            --seed $seed
        
        echo "  Done."
    done
    echo ""
done

echo "All tests completed!"