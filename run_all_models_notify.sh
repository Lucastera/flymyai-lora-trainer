#!/bin/bash
###############################################################
# 顺序跑 4 个模型 × 3 个 seed = 12 个任务
# 每个任务用满 8 张 GPU；某个任务失败会自动重试(最多 MAX_RETRY 次)
# 重试/成功/最终失败跳过 均会推送微信通知
###############################################################

KEY="SCT311782TSxxalR1CG55SoCD8QXqet4x6"
HOST=$(hostname)

MODELS=(
    "Qwen/Qwen-Image"
    "black-forest-labs/FLUX.1-dev"
    "black-forest-labs/FLUX.2-klein-4B"
    "Tongyi-MAI/Z-Image"
)

SEEDS=(42 0 123)

VARIATIONS="3"
WIDTH=512
HEIGHT=512
GPUS="0,1,2,3,4,5,6,7"
NUM_PROC=8
MAX_RETRY=3   # 含首次尝试；即最多跑3次，3次都失败则跳过

send_notify() {
    local title="$1"
    local desp="$2"
    curl -s "https://sctapi.ftqq.com/${KEY}.send" \
        -d "title=${title}" \
        -d "desp=${desp}" > /dev/null
}

run_one_task() {
    local model_name="$1"
    local seed="$2"
    local tag=$(echo "${model_name}_seed${seed}" | tr '/' '_')
    local attempt=1
    local exit_code=1

    while [ $attempt -le $MAX_RETRY ]; do
        local log_file="run_${tag}_attempt${attempt}_$(date '+%Y%m%d_%H%M%S').log"
        local start_time=$(date '+%Y-%m-%d %H:%M:%S')

        if [ $attempt -gt 1 ]; then
            send_notify "🔁任务重试(第${attempt}次)" \
                "主机: ${HOST}%0A模型: ${model_name}%0Aseed: ${seed}%0A时间: ${start_time}%0A说明: 上一次运行失败，正在重新唤醒进程"
        fi

        CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
            --num_processes=${NUM_PROC} \
            inference_multi_volin_v2.py \
            --variations ${VARIATIONS} \
            --model_name="${model_name}" \
            --base_seed=${seed} \
            --width=${WIDTH} \
            --height=${HEIGHT} 2>&1 | tee "$log_file"

        exit_code=${PIPESTATUS[0]}
        local end_time=$(date '+%Y-%m-%d %H:%M:%S')
        local tail_log=$(tail -100 "$log_file" | sed 's/$/&%0A/g' | tr -d '\n')

        if [ $exit_code -eq 0 ]; then
            send_notify "✅任务完成" \
                "主机: ${HOST}%0A模型: ${model_name}%0Aseed: ${seed}%0A开始: ${start_time}%0A结束: ${end_time}%0A%0A最后日志:%0A\`\`\`%0A${tail_log}\`\`\`"
            return 0
        else
            if [ $attempt -eq $MAX_RETRY ]; then
                send_notify "❌任务失败(已重试${MAX_RETRY}次,已跳过)" \
                    "主机: ${HOST}%0A模型: ${model_name}%0Aseed: ${seed}%0A开始: ${start_time}%0A结束: ${end_time}%0A退出码: ${exit_code}%0A%0A最后日志:%0A\`\`\`%0A${tail_log}\`\`\`"
            fi
        fi

        attempt=$((attempt + 1))
    done

    return $exit_code
}

TOTAL=$((${#MODELS[@]} * ${#SEEDS[@]}))
COUNT=0

send_notify "🚀开始跑批" "主机: ${HOST}%0A共 ${TOTAL} 个任务(4模型 x 3seed)即将开始，逐个串行执行"

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "===== [${COUNT}/${TOTAL}] 模型: ${model}  seed: ${seed} ====="
        run_one_task "$model" "$seed"
    done
done

send_notify "🎉全部任务已跑完" "主机: ${HOST}%0A共 ${TOTAL} 个任务(模型x seed)已全部处理完毕（含重试/跳过的情况）"