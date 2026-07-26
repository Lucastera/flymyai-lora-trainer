#!/bin/bash
###############################################################
# 小规模冒烟测试：睡觉前先跑一遍，确认4个模型都能正常加载+出图
# 每个模型只生成4张图，用完整8卡，前台运行，出错立刻打印+中止
###############################################################

MODELS=(
    # "Qwen/Qwen-Image"
    # "black-forest-labs/FLUX.1-dev"
    "black-forest-labs/FLUX.2-klein-4B"
    # "Tongyi-MAI/Z-Image"
)

VARIATIONS="3"
WIDTH=512
HEIGHT=512
GPUS="0,1,2,3,4,5,6,7"
NUM_PROC=8
TEST_SEED=42
MAX_SAMPLES=4   # 每个模型只测4个样本，快速验证是否能跑通

for model in "${MODELS[@]}"; do
    echo ""
    echo "=============================================="
    echo "🧪 测试模型: ${model}"
    echo "=============================================="
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --num_processes=${NUM_PROC} \
        inference_multi_volin_v2.py \
        --variations ${VARIATIONS} \
        --model_name="${model}" \
        --base_seed=${TEST_SEED} \
        --max_samples=${MAX_SAMPLES} \
        --width=${WIDTH} \
        --height=${HEIGHT}

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ 模型 ${model} 测试失败！请检查报错信息，先排查问题再跑正式任务。"
        exit 1
    else
        echo "✅ 模型 ${model} 测试通过"
    fi
done

echo ""
echo "🎉 4个模型全部测试通过！可以放心执行 run_all_models_notify.sh 开始正式跑批。"