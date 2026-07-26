import requests
import json


def download_images(json_str, filepath):
    data_dict = json.loads(json_str)
    image_list = data_dict.get("data", [])
    
    if not image_list:
        print("未发现图片数据")
        return

    for index, item in enumerate(image_list):
        url = item.get("url")
        if url:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                print(f"下载失败，状态码: {response.status_code}")

def call_LLM(prompt, img_path1, img_path2, save_path):
    url = "https://api.bltcy.ai/v1/images/edits"
    YOUR_API_KEY = "sk-cRzRku7mmb0n7ODKsd9iwd8pTXYhRjO3CI0jH1lDyNfB921h"

    img_path1 = '/home/hong/hongyu/violin/benchmark/data/'+img_path1
    img_path2 = '/home/hong/hongyu/violin/benchmark/data/'+img_path2

    payload = {
        "prompt": prompt,
        "model": "nano-banana-2",
        "aspect_ratio":"1:1",
        "image_size":"1K",
    }

    files=[
        ('image',(img_path1,open(img_path1,'rb'),'image/jpeg')),
        ('mask',(img_path2,open(img_path2,'rb'),'image/png'))
    ]

    headers = {
        'Authorization': f'Bearer {YOUR_API_KEY}',
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    # print(response.text)
    download_images(response.text, save_path)
    print(f"Saving to {save_path}")


if __name__ == '__main__':
    call_LLM("Apply the binary mask in Image 2 to the image in Image 1. For every pixel, if the mask value is white (value 255), keep the original color from Image 1; if the mask value is black (value 0), change it to pure black.", 
    "Variation_4_raw_image/images/000000000.jpg", 
    "Variation_4_raw_image/inpainting_mask/000000000.png", 
    "/home/hong/hongyu/violin/test_results/test.png")
