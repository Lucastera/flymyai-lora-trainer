# 定义所有需要评估的目录
GEN_DIRS=(
    "outputs/FLUX.2-klein-4B_V1-2-3_256x256"
    "outputs/FLUX.2-klein-4B_V1-2-3_512x512"
    "outputs/Z-Image_V1-2-3_256x256"
    "outputs/Z-Image_V1-2-3_512x512"
)

# 循环执行评估
for gen_dir in "${GEN_DIRS[@]}"; do
    echo "========================================="
    echo "Evaluating: $gen_dir"
    echo "========================================="
    python evaluate.py --gen_dir "$gen_dir"
    echo ""
done

echo "✅ All evaluations completed!"