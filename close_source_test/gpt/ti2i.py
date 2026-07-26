import requests
import json
import os
import base64

def encode_image(image_path):
    """将图片文件转换为 base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
                print(f"保存成功: {filepath}")
            else:
                print(f"下载失败，状态码: {response.status_code}")


def save_b64_images(json_str, filepath):
    """
    解析 JSON 字符串，解码 Base64 数据并保存为图片
    """
    try:
        data_dict = json.loads(json_str)
    except json.JSONDecodeError:
        print("解析 JSON 失败")
        return

    # 获取数据列表
    image_list = data_dict.get("data", [])
    
    if not image_list:
        # 如果报错，打印出错误信息方便调试
        if "error" in data_dict:
            print(f"API 报错: {data_dict['error']}")
        else:
            print("未发现图片数据")
        return

    # 注意：如果生成多张图，这里需要处理文件名，否则会被覆盖
    for index, item in enumerate(image_list):
        # 此时获取的是 b64_json 字段
        b64_data = item.get("b64_json")
        
        if b64_data:
            # 确定保存路径（处理多图情况）
            current_path = filepath if len(image_list) == 1 else f"{os.path.splitext(filepath)[0]}_{index}.png"
            
            # 将 base64 字符串转换为二进制数据
            img_data = base64.b64decode(b64_data)
            
            # 直接写入文件
            with open(current_path, 'wb') as f:
                f.write(img_data)
            print(f"保存成功: {current_path}")
        else:
            print("数据中未包含 b64_json 字段")


def call_LLM(prompt, img_path1, img_path2, save_path):
    YOUR_API_KEY = "sk-WqpREiB8RQ0W8wVDzIqMvfYcORkJMxXbvvNcaaFTUrwGtxyC"
    url = "https://api.bltcy.ai/v1/images/generations"

    img_path1 = '/home/hong/hongyu/violin/benchmark/data/'+img_path1
    img_path2 = '/home/hong/hongyu/violin/benchmark/data/'+img_path2

    base64_image1 = encode_image(img_path1)
    base64_image2 = encode_image(img_path2)

    # payload = json.dumps({
    #     "model": "gpt-image-2",
    #     "prompt": prompt,
    #     "aspect_ratio": "1:1",
    #     "response_format":"b64_json",
    #     "image":[
    #         img_path1, img_path2
    #     ]
    # })

    payload = json.dumps({
        "model": "gpt-image-2",
        "prompt": prompt,
        "aspect_ratio": "1:1",
        "response_format": "b64_json",
        "image": [
            f"data:image/jpeg;base64,{base64_image1}", # 格式通常需要加上 Data URI 前缀
            f"data:image/png;base64,{base64_image2}"
        ]
    })

    headers = {
        'Authorization': f'Bearer {YOUR_API_KEY}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)
    # download_images(response.text, save_path)
    save_b64_images(response.text, save_path)
    print(f"Saving to {save_path}")


if __name__ == '__main__':
    call_LLM("Apply the binary mask in Image 2 to the image in Image 1. For every pixel, if the mask value is white (value 255), keep the original color from Image 1; if the mask value is black (value 0), change it to pure black.", 
    "Variation_4_raw_image/images/000000000.jpg", 
    "Variation_4_raw_image/inpainting_mask/000000000.png", 
    "/home/hong/hongyu/violin/test_results/test.png")
