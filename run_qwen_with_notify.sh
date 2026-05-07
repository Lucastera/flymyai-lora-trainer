#!/bin/bash

KEY="SCT311782TSxxalR1CG55SoCD8QXqet4x6"
HOST=$(hostname)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="run_$(date '+%Y%m%d_%H%M%S').log"

# 执行命令并保存日志
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 inference_multi.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# 获取最后100行日志
TAIL_LOG=$(tail -100 "$LOG_FILE" | sed 's/$/&%0A/g' | tr -d '\n')

# 发送通知
if [ $EXIT_CODE -eq 0 ]; then
    curl -s "https://sctapi.ftqq.com/${KEY}.send" \
        -d "title=✅任务完成" \
        -d "desp=主机: ${HOST}%0A开始: ${START_TIME}%0A结束: ${END_TIME}%0A%0A最后日志:%0A\`\`\`%0A${TAIL_LOG}\`\`\`"
else
    curl -s "https://sctapi.ftqq.com/${KEY}.send" \
        -d "title=❌任务失败" \
        -d "desp=主机: ${HOST}%0A开始: ${START_TIME}%0A结束: ${END_TIME}%0A退出码: ${EXIT_CODE}%0A%0A最后日志:%0A\`\`\`%0A${TAIL_LOG}\`\`\`"
fi