import os

def find_prompt_file(base_dir, target_prompt):
    """
    在数据集中查找包含指定 prompt 的 txt 文件
    """
    target_prompt = target_prompt.strip()
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_path = os.path.join(root, file)
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if content == target_prompt:
                        print(f"Found: {txt_path}")
                        return txt_path
                        
                except Exception as e:
                    continue
    
    print("Not found.")
    return None


if __name__ == "__main__":
    base_dir = "ColorBench-v1/Test_Sets"
    
    target_prompt = "Represented in Hex code, produce an image divided into four equal parts: TR colored #5FABAB, BR colored #7931D3, TL colored #2B2630, BL colored #10361A. The colors should meet directly, no borders."
    
    find_prompt_file(base_dir, target_prompt)