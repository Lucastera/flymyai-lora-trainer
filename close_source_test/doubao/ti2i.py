import requests
import json
import os
from urllib.parse import urlparse

def download_image(json_str, filepath):
    data = json.loads(json_str)
    image_url = data['data'][0]['url']

    path = urlparse(image_url).path
    filename = os.path.basename(path) 
    
    if not filename:
        filename = f"image_{data.get('created', 'unknown')}.jpg"

    response = requests.get(image_url, stream=True, timeout=30)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        print(f"下载失败，状态码: {response.status_code}")
            


def call_LLM(prompt, img_path1, img_path2, save_path):
    url = "https://api.bltcy.ai/v1/images/edits"
    YOUR_API_KEY = "sk-IOpu6NE1OHVAIYo04yo7QVYs1HVNmTP6jaBrdo9dC2IBURJf"

    img_path1 = '/home/hong/hongyu/violin/benchmark/data/'+img_path1
    img_path2 = '/home/hong/hongyu/violin/benchmark/data/'+img_path2

    payload = {
        "model": "doubao-seedream-5-0-260128",
        "prompt": prompt,
        "n": 1,
        "response_format": "url",
        "size": "2K",
        "aspect_ratio": "1:1",
        "watermark": False
    }
    headers = {
    'Authorization': f'Bearer {YOUR_API_KEY}',
    }

    files=[
        ('image',(img_path1,open(img_path1,'rb'),'image/jpeg')),
        ('mask',(img_path2,open(img_path2,'rb'),'image/png'))
    ]

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    print(response.text)

    try:
        download_image(response.text, save_path)
        print(f"Saving to {save_path}")
    except:
        print('Downloading error')
    

if __name__ == '__main__':
    call_LLM("Apply the binary mask in Image 2 to the image in Image 1. For every pixel, if the mask value is white (value 255), keep the original color from Image 1; if the mask value is black (value 0), change it to pure black.", 
    "Variation_4_raw_image/images/000000000.jpg", 
    "Variation_4_raw_image/inpainting_mask/000000000.png", 
    "/home/hong/hongyu/violin/test_results/test.png")