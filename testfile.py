import os
from pathlib import Path
from collections import defaultdict

def count_image_files_in_subdirs(base_path):
    """统计指定目录下所有 train 和 test 子目录的图片文件数量"""
    
    # 定义图片文件扩展名
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    
    base_path = Path(base_path)
    results = defaultdict(lambda: {'train': 0, 'test': 0})
    
    # 遍历所有子目录
    for prompt_level in base_path.iterdir():
        if not prompt_level.is_dir():
            continue
            
        for color_level in prompt_level.iterdir():
            if not color_level.is_dir():
                continue
            
            # 构建路径标识
            path_key = f"{prompt_level.name}/{color_level.name}"
            
            # 统计 train 目录中的图片
            train_path = color_level / "train"
            if train_path.exists() and train_path.is_dir():
                images = [f for f in train_path.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions]
                results[path_key]['train'] = len(images)
            
            # 统计 test 目录中的图片
            test_path = color_level / "test"
            if test_path.exists() and test_path.is_dir():
                images = [f for f in test_path.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions]
                results[path_key]['test'] = len(images)
    
    return results

def main():
    base_path = "ColorBench-v1/Finetune_Level1_Sets"
    
    print("统计 ColorBench-v1/Finetune_Level1_Sets 目录下的图片文件数量\n")
    print("=" * 80)
    
    results = count_image_files_in_subdirs(base_path)
    
    # 按路径排序并打印结果
    for path_key in sorted(results.keys()):
        counts = results[path_key]
        print(f"\n{path_key}:")
        print(f"  Train: {counts['train']} 张图片")
        print(f"  Test:  {counts['test']} 张图片")
    
    # 统计总数
    total_train = sum(counts['train'] for counts in results.values())
    total_test = sum(counts['test'] for counts in results.values())
    
    print("\n" + "=" * 80)
    print(f"\n总计:")
    print(f"  Train 总图片数: {total_train}")
    print(f"  Test 总图片数:  {total_test}")
    print(f"  总图片数:      {total_train + total_test}")

if __name__ == "__main__":
    main()