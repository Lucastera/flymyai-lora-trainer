import requests
import json
import os


def download_images(json_str, filepath):
    """Download image from API response"""
    data_dict = json.loads(json_str)
    image_list = data_dict.get("data", [])
    
    if not image_list:
        print("未发现图片数据")
        return

    for index, item in enumerate(image_list):
        url = item.get("url")
        if url:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"✓ Successfully saved to {filepath}")
            else:
                print(f"✗ 下载失败，状态码: {response.status_code}")


def call_LLM(prompt, img_path1, img_path2, save_path, api_key, model="flux-kontext-max", image_size="1024"):
    """Call image editing API"""
    url = "https://api.bltcy.ai/v1/images/edits"

    # Convert image_size to WxH format
    size_value = f"{image_size}x{image_size}"
    
    payload = {
        "prompt": prompt,
        "model": model,
        "size": size_value,
        "aspect_ratio": "1:1",
    }

    # 支持多个图片上传，按文档说明使用相同的字段名
    files = [
        ('image', (os.path.basename(img_path1), open(img_path1, 'rb'), 'image/jpeg')),
        ('mask', (os.path.basename(img_path2), open(img_path2, 'rb'), 'image/png'))
    ]

    headers = {
        'Authorization': f'Bearer {api_key}',
    }

    print(f"Calling API...")
    print(f"  Model: {model}")
    print(f"  Size: {size_value}")
    print(f"  Aspect ratio: 1:1")
    print(f"  Image 1: {img_path1}")
    print(f"  Image 2: {img_path2}")

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(f"Response status: {response.status_code}")
    print(f"Response: {response.text}")

    download_images(response.text, save_path)
    print(f"Saving to {save_path}")


if __name__ == '__main__':
    # Configuration
    API_KEY = "sk-o9WYOsdKKZC853Ng560e70Fd2b8249139f4b989fE771F9Dd"
    MODEL = "flux-kontext-max"
    IMAGE_SIZE = "256"
    
    BASE_DIR = "/home/kuan/code/flymyai-lora-trainer/VIOLIN_v2/data"
    
    # Updated paths based on your example
    img1_path = os.path.join(BASE_DIR, "Variation_4_raw_image/images/000000000.jpg")
    img2_path = os.path.join(BASE_DIR, "Variation_4_raw_image/inpainting_mask/000000000.png")
    
    # Save to test_outputs directory (保持原逻辑)
    save_path = os.path.join("test_outputs", f"test_{MODEL}_{IMAGE_SIZE}.png")
    
    # Updated prompt
    prompt = "Apply the binary mask in Image 2 to the image in Image 1. For every pixel, if the mask value is white (value 255), keep the original color from Image 1; if the mask value is black (value 0), change it to pure black."
    
    print("="*60)
    print("TEST IMAGE EDITING API")
    print("="*60)
    
    # Check if input files exist
    if not os.path.exists(img1_path):
        print(f"✗ Error: Image 1 not found: {img1_path}")
        exit(1)
    if not os.path.exists(img2_path):
        print(f"✗ Error: Image 2 not found: {img2_path}")
        exit(1)
    
    print(f"✓ Input files found")
    print()
    
    # Call API
    call_LLM(
        prompt=prompt,
        img_path1=img1_path,
        img_path2=img2_path,
        save_path=save_path,
        api_key=API_KEY,
        model=MODEL,
        image_size=IMAGE_SIZE
    )
    
    print()
    print("="*60)
    print("TEST COMPLETED")
    print("="*60)