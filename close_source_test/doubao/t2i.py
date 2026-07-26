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
            


def call_LLM(prompt, save_path):
    url = "https://api.bltcy.ai/v1/images/generations"
    YOUR_API_KEY = "sk-IOpu6NE1OHVAIYo04yo7QVYs1HVNmTP6jaBrdo9dC2IBURJf"

    payload = json.dumps({
        "model": "doubao-seedream-5-0-260128",
        "prompt": prompt,
        "n": 1,
        "response_format": "url",
        "size": "2K",
        "aspect_ratio": "1:1",
        "watermark": False
    })
    headers = {
    'Authorization': f'Bearer {YOUR_API_KEY}',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)

    download_image(response.text, save_path)
    print(f"Saving to {save_path}")

if __name__ == '__main__':
    call_LLM("a pure color red", "/home/hong/hongyu/violin/test_results/test.png")