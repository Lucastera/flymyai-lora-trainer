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

def call_LLM(prompt, save_path):
    url = "https://api.bltcy.ai/v1/images/generations"
    YOUR_API_KEY = "sk-cRzRku7mmb0n7ODKsd9iwd8pTXYhRjO3CI0jH1lDyNfB921h"

    payload = json.dumps({
        "prompt": prompt,
        "model": "nano-banana-2",
        "aspect_ratio":"1:1",
        "image_size":"1K",
    })
    headers = {
        'Authorization': f'Bearer {YOUR_API_KEY}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)
    download_images(response.text, save_path)
    print(f"Saving to {save_path}")


if __name__ == '__main__':
    call_LLM('a cat', "/home/hong/hongyu/violin/test_results/test.png")
