import os
import json

def verify_generation(output_dir):
    """验证生成结果的数量"""
    
    # 1. 读取 sample_list.json
    list_file = os.path.join(output_dir, "sample_list.json")
    if not os.path.exists(list_file):
        print(f"错误: 找不到 {list_file}")
        return
    
    with open(list_file, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    expected_count = len(saved_data['samples'])
    print(f"sample_list.json 中记录的样本数: {expected_count}")
    
    # 2. 统计实际生成的 PNG 文件数量
    png_files = [f for f in os.listdir(output_dir) if f.endswith('_gen.png')]
    actual_count = len(png_files)
    print(f"实际生成的 PNG 文件数: {actual_count}")
    
    # 3. 检查原始数据目录
    base_dir = saved_data.get('base_dir', '')
    prompt_levels = saved_data.get('prompt_levels', [])
    color_levels = saved_data.get('color_levels', [])
    split = saved_data.get('split', 'test')
    max_samples = saved_data.get('max_samples_per_dir')
    
    print(f"\n配置信息:")
    print(f"  base_dir: {base_dir}")
    print(f"  prompt_levels: {prompt_levels}")
    print(f"  color_levels: {color_levels}")
    print(f"  split: {split}")
    print(f"  max_samples_per_dir: {max_samples}")
    
    # 4. 统计原始数据每个目录的文件数
    print(f"\n各目录原始文件统计:")
    total_original = 0
    total_sampled = 0
    
    for p_level in prompt_levels:
        for c_level in color_levels:
            dir_path = os.path.join(
                base_dir,
                f"Prompt_Level_{p_level}",
                f"Color_Level_{c_level}",
                split
            )
            
            if not os.path.exists(dir_path):
                print(f"  [不存在] {dir_path}")
                continue
            
            # 统计 txt 文件
            txt_count = 0
            for root, dirs, files in os.walk(dir_path):
                txt_count += len([f for f in files if f.endswith('.txt')])
            
            # 计算采样后数量
            if max_samples and txt_count > max_samples:
                sampled = max_samples
            else:
                sampled = txt_count
            
            total_original += txt_count
            total_sampled += sampled
            
            print(f"  P{p_level}_C{c_level}: 原始 {txt_count}, 采样后 {sampled}")
    
    print(f"\n汇总:")
    print(f"  原始总文件数: {total_original}")
    print(f"  理论采样数: {total_sampled}")
    print(f"  sample_list 记录: {expected_count}")
    print(f"  实际生成图片: {actual_count}")
    
    # 5. 验证结果
    print(f"\n验证结果:")
    if total_sampled == expected_count:
        print(f"  ✅ 采样数量正确 ({total_sampled} == {expected_count})")
    else:
        print(f"  ❌ 采样数量不匹配 ({total_sampled} != {expected_count})")
    
    if actual_count == expected_count:
        print(f"  ✅ 生成完成 ({actual_count}/{expected_count})")
    else:
        print(f"  ⏳ 生成中或未完成 ({actual_count}/{expected_count}, 剩余 {expected_count - actual_count})")


if __name__ == "__main__":
    output_dir = "outputs/Qwen-Image_lora_P1-2-3-4-5-6_C1-2-3_test_1000"
    verify_generation(output_dir)