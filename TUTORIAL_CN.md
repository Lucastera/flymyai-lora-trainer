# FlyMy.AI LoRA Trainer 代码仓库详解

## 🎯 写给初学者的话

你好！如果你是第一次看这个项目，觉得很复杂，完全不用担心。这份教程会**一步一步**带你理解这个代码仓库的**每一个部分**。

我会用最简单的语言，从零开始讲解。相信我，读完这份教程后，你会对整个项目有清晰的认识！

---

## 📚 目录

1. [这个项目是做什么的？](#1-这个项目是做什么的)
2. [核心概念讲解](#2-核心概念讲解)
3. [目录结构详解](#3-目录结构详解)
4. [核心代码文件讲解](#4-核心代码文件讲解)
5. [数据准备详解](#5-数据准备详解)
6. [训练流程详解](#6-训练流程详解)
7. [推理使用详解](#7-推理使用详解)
8. [实战示例](#8-实战示例)
9. [常见问题](#9-常见问题)

---

## 1. 这个项目是做什么的？

### 1.1 简单回答

**这个项目是一个 AI 图像生成模型的训练工具**。

用大白话说：
- 你有一些图片和它们的描述文字
- 你想训练一个 AI 模型，让它能够根据你的风格生成新图片
- 这个项目就是帮你完成这个训练过程的工具

### 1.2 支持的模型

这个项目支持训练三种主流的 AI 图像生成模型：

1. **Qwen-Image**：阿里巴巴开发的中文友好图像生成模型
2. **Qwen-Image-Edit**：可以编辑图片的模型（比如改变图片中的内容）
3. **FLUX.1-dev**：一个非常强大的人像和角色生成模型

### 1.3 什么是 LoRA？

**LoRA**（Low-Rank Adaptation）是一种**高效的模型微调技术**。

**为什么需要 LoRA？**

想象一下：
- 一个完整的 AI 图像模型可能有**几十亿个参数**（就像大脑中的神经元）
- 如果你想训练整个模型，需要：
  - 超级强大的 GPU（显卡）
  - 大量的时间
  - 很多训练数据

**LoRA 的聪明之处：**
- 不训练整个模型的所有参数
- 只训练一小部分**额外添加的参数**（可能只有几百万个）
- 就像给模型加了一个"小插件"，而不是重新训练整个模型
- 训练速度快，需要的显存小，效果还很好！

**举个例子：**
```
完整模型训练：需要 80GB 显存，训练 10 天
LoRA 训练：需要 24GB 显存，训练 2-4 小时
```

---

## 2. 核心概念讲解

在深入代码之前，我们需要理解几个核心概念：

### 2.1 什么是扩散模型（Diffusion Model）？

**扩散模型**是目前最先进的图像生成技术。

**工作原理（用简单的比喻）：**

想象你在画画：
1. **正向过程**：把一张清晰的图片逐渐加入噪点，最后变成完全的噪声
2. **反向过程（生成）**：从噪声开始，AI 模型逐步去除噪声，最后生成清晰的图片

```
清晰图片 → 加噪 → 加噪 → 加噪 → 纯噪声
纯噪声 → 去噪 → 去噪 → 去噪 → 清晰图片 ✨
```

### 2.2 什么是 Transformer？

**Transformer** 是 AI 模型的一种架构，特别擅长理解和生成序列数据。

在这个项目中：
- Transformer 负责理解**文本提示词**（prompt）
- 它把文字转换成 AI 能理解的数字表示
- 然后指导图像生成过程

### 2.3 什么是 VAE？

**VAE**（Variational Autoencoder，变分自编码器）是一个**图像压缩和解压工具**。

**为什么需要 VAE？**

- 一张 1024×1024 的 RGB 图片有 **300 多万个像素点**
- 直接处理这么大的数据会非常慢
- VAE 可以把图片压缩成小得多的"潜在表示"（latent）

**工作流程：**
```
原始图片 (1024×1024×3) 
  ↓ VAE 编码
潜在表示 (小得多，比如 128×128×4)
  ↓ AI 模型在这个小空间里工作
潜在表示 (处理后)
  ↓ VAE 解码
生成的图片 (1024×1024×3)
```

### 2.4 训练过程的核心概念

#### 2.4.1 损失函数（Loss Function）

**损失函数**衡量 AI 模型的预测有多"错"。

- 损失值越小，模型预测越准确
- 训练的目标就是不断**降低损失值**

#### 2.4.2 学习率（Learning Rate）

**学习率**控制模型每次更新参数的步长。

```
学习率太大：模型可能学不好，跳来跳去
学习率太小：模型学得太慢
合适的学习率：稳步进步 ✨
```

#### 2.4.3 批次大小（Batch Size）

**批次大小**是每次训练使用多少张图片。

```
批次大小 = 1：每次看 1 张图
批次大小 = 4：每次看 4 张图，然后一起更新模型
```

- 批次越大，训练越稳定，但需要更多显存
- 批次越小，显存占用少，但训练可能不太稳定

---

## 3. 目录结构详解

让我们看看项目的文件和文件夹都是干什么的：

```
flymyai-lora-trainer/
├── README.md                    # 项目说明文档（英文）
├── TUTORIAL_CN.md              # 本教程文件
├── requirements.txt             # Python 依赖包列表
├── LICENSE                      # 开源协议
│
├── assets/                      # 资源文件夹
│   ├── flymy_transparent.png   # 项目 Logo
│   ├── lora.png                # 示例输出图片
│   └── ...                     # 其他示例图片
│
├── image_datasets/             # 数据集处理模块
│   ├── dataset.py              # 核心：数据加载器
│   └── control_dataset.py      # 图像编辑专用数据加载器
│
├── train_configs/              # 训练配置文件夹
│   ├── train_lora.yaml         # Qwen-Image LoRA 训练配置
│   ├── train_lora_4090.yaml    # 24GB 显存优化配置
│   ├── train_flux_config.yaml  # FLUX 模型训练配置
│   ├── train_lora_qwen_edit.yaml  # 图像编辑训练配置
│   └── ...                     # 其他配置文件
│
├── train.py                    # Qwen-Image LoRA 训练主程序
├── train_4090.py               # 低显存版训练程序
├── train_flux_lora.py          # FLUX LoRA 训练主程序
├── train_qwen_edit_lora.py     # 图像编辑训练程序
├── train_full_qwen_image.py    # Qwen 完整模型训练
├── train_kandinsky_lora.py     # Kandinsky 模型训练
├── train_z_image_lora.py       # Z-Image 模型训练
│
├── inference.py                # 推理（生成图片）程序
├── qwen_full_inference_example.py  # 完整模型推理示例
├── qwen_image_lora_example.json    # ComfyUI 工作流文件
│
└── utils/                      # 工具函数
    └── validate_dataset.py     # 数据集验证工具
```

### 3.1 重点文件说明

| 文件名 | 作用 | 重要程度 |
|--------|------|----------|
| `train.py` | Qwen-Image 训练的核心代码 | ⭐⭐⭐⭐⭐ |
| `train_flux_lora.py` | FLUX 训练的核心代码 | ⭐⭐⭐⭐⭐ |
| `image_datasets/dataset.py` | 数据加载和预处理 | ⭐⭐⭐⭐⭐ |
| `train_configs/*.yaml` | 训练参数配置 | ⭐⭐⭐⭐ |
| `inference.py` | 使用训练好的模型生成图片 | ⭐⭐⭐⭐ |
| `requirements.txt` | 安装所需的 Python 包 | ⭐⭐⭐ |

---

## 4. 核心代码文件讲解

现在我们深入到代码层面，一步步理解每个文件在做什么。

### 4.1 train.py - Qwen-Image 训练主程序

这是 Qwen-Image LoRA 训练的核心文件。让我们分段理解：

#### 4.1.1 导入必要的库

```python
import argparse  # 命令行参数解析
import torch     # PyTorch 深度学习框架
from accelerate import Accelerator  # 加速训练（支持多 GPU 等）
from diffusers import QwenImagePipeline  # Qwen 图像生成管道
from peft import LoraConfig  # LoRA 配置
```

**解释：**
- 每个 `import` 都是引入一个工具库
- 就像在做菜前准备食材和工具

#### 4.1.2 加载配置

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return args.config

args = OmegaConf.load(parse_args())
```

**这段代码做什么？**
1. 从命令行读取配置文件路径
2. 加载 YAML 配置文件
3. 把配置存储在 `args` 变量中

**举例：**
```bash
python train.py --config ./train_configs/train_lora.yaml
```

#### 4.1.3 初始化 Accelerator

```python
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
)
```

**Accelerator 是什么？**
- 这是 Hugging Face 提供的训练加速工具
- 自动处理多 GPU、混合精度训练等复杂操作
- 让你的代码可以轻松从单 GPU 扩展到多 GPU

**混合精度（Mixed Precision）：**
- 正常情况下，数字用 32 位存储（float32）
- 混合精度用 16 位存储（float16 或 bfloat16）
- **优点**：速度更快，显存占用减半
- **风险**：精度稍微降低（但通常影响很小）

#### 4.1.4 加载模型组件

```python
# 1. 加载文本编码管道
text_encoding_pipeline = QwenImagePipeline.from_pretrained(
    args.pretrained_model_name_or_path, 
    transformer=None,  # 不加载 transformer
    vae=None,          # 不加载 VAE
)

# 2. 加载 VAE（图像编码器/解码器）
vae = AutoencoderKLQwenImage.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="vae",
)

# 3. 加载 Transformer（核心生成模型）
flux_transformer = QwenImageTransformer2DModel.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="transformer",
)
```

**为什么分开加载？**
- 每个组件有不同的作用
- 我们只想训练 Transformer 的 LoRA 部分
- VAE 和文本编码器保持不变（冻结参数）

#### 4.1.5 配置 LoRA

```python
lora_config = LoraConfig(
    r=args.rank,                              # LoRA 秩（通常 4-128）
    lora_alpha=args.rank,                     # LoRA 缩放因子
    init_lora_weights="gaussian",             # 初始化方法
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # 要训练的模块
)

flux_transformer.add_adapter(lora_config)
```

**参数解释：**

- **rank (r)**：LoRA 的秩，控制 LoRA 层的大小
  - 越大：表达能力越强，但参数越多
  - 越小：参数少，训练快，但可能学不好
  - 推荐值：8-32

- **target_modules**：在哪些层添加 LoRA
  - `to_q`、`to_k`、`to_v`：注意力机制的查询、键、值层
  - `to_out.0`：输出层

#### 4.1.6 冻结不训练的参数

```python
vae.requires_grad_(False)          # VAE 不训练
flux_transformer.requires_grad_(False)  # 先全部冻结

# 只解冻 LoRA 参数
for n, param in flux_transformer.named_parameters():
    if 'lora' not in n:
        param.requires_grad = False  # 不是 LoRA，冻结
    else:
        param.requires_grad = True   # 是 LoRA，训练
```

**requires_grad 是什么？**
- `True`：这个参数会在训练中更新
- `False`：这个参数固定不变

#### 4.1.7 准备优化器

```python
optimizer = torch.optim.AdamW(
    lora_layers,                    # 只优化 LoRA 参数
    lr=args.learning_rate,          # 学习率
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
)
```

**优化器是什么？**
- 优化器决定如何更新模型参数
- **AdamW** 是目前最流行的优化器之一
- 它会根据损失函数的梯度，智能调整每个参数

#### 4.1.8 训练循环

```python
for epoch in range(1):
    for step, batch in enumerate(train_dataloader):
        img, prompts = batch  # 获取一批图片和文本
        
        # 1. 将图片编码为潜在表示
        pixel_latents = vae.encode(pixel_values).latent_dist.sample()
        
        # 2. 添加噪声
        noise = torch.randn_like(pixel_latents)
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
        
        # 3. 编码文本提示词
        prompt_embeds = text_encoding_pipeline.encode_prompt(prompts)
        
        # 4. 模型预测
        model_pred = flux_transformer(
            hidden_states=packed_noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
        )
        
        # 5. 计算损失
        target = noise - pixel_latents
        loss = torch.mean((model_pred - target) ** 2)
        
        # 6. 反向传播，更新参数
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

**训练步骤详解：**

1. **编码图片**：把图片转换成潜在表示（压缩）
2. **添加噪声**：模拟扩散过程
3. **编码文本**：把提示词转换成嵌入向量
4. **模型预测**：让模型预测如何去噪
5. **计算损失**：比较预测和真实目标
6. **更新参数**：根据损失调整 LoRA 参数

### 4.2 image_datasets/dataset.py - 数据加载器

这个文件负责加载和预处理训练数据。

#### 4.2.1 自定义数据集类

```python
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='txt'):
        # 找到所有图片
        self.images = [
            os.path.join(img_dir, i) 
            for i in os.listdir(img_dir) 
            if '.jpg' in i or '.png' in i
        ]
        self.img_size = img_size
        self.caption_type = caption_type
```

**这个类做什么？**
- 扫描图片文件夹，找到所有图片
- 记录图片路径和大小
- 准备好数据加载的基础设施

#### 4.2.2 加载单个样本

```python
def __getitem__(self, idx):
    # 1. 随机选择一张图片
    idx = random.randint(0, len(self.images) - 1)
    
    # 2. 打开图片
    img = Image.open(self.images[idx]).convert('RGB')
    
    # 3. 调整大小
    img = image_resize(img, self.img_size)
    
    # 4. 转换为张量
    img = torch.from_numpy((np.array(img) / 127.5) - 1)
    img = img.permute(2, 0, 1)  # 调整维度顺序
    
    # 5. 加载对应的文本描述
    txt_path = self.images[idx].rsplit('.', 1)[0] + '.txt'
    prompt = open(txt_path, encoding='utf-8').read()
    
    return img, prompt
```

**数据预处理步骤：**

1. **打开图片**：使用 PIL 库读取
2. **调整大小**：统一图片尺寸
3. **归一化**：把像素值从 [0, 255] 转换到 [-1, 1]
4. **调整维度**：从 (H, W, C) 转换到 (C, H, W)
5. **加载文本**：读取对应的 .txt 文件

#### 4.2.3 图片尺寸处理

```python
def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))
```

**为什么要调整大小？**
- 不同图片大小不一样
- 需要统一大小才能批量处理
- 同时保持长宽比，避免图片变形

### 4.3 train_configs/train_lora.yaml - 配置文件

配置文件定义了所有训练参数：

```yaml
# 模型路径
pretrained_model_name_or_path: Qwen/Qwen-Image

# 数据配置
data_config:
  img_dir: ./your_lora_dataset    # 图片文件夹路径
  img_size: 1024                  # 图片大小
  train_batch_size: 1             # 批次大小
  num_workers: 4                  # 数据加载线程数
  caption_dropout_rate: 0.1       # 文本丢弃率（正则化技巧）

# 训练参数
max_train_steps: 3000             # 总训练步数
learning_rate: 1e-4               # 学习率
train_batch_size: 1               # 批次大小
gradient_accumulation_steps: 1    # 梯度累积步数

# LoRA 参数
rank: 16                          # LoRA 秩

# 输出
output_dir: ./output              # 模型保存路径
checkpointing_steps: 250          # 每多少步保存一次

# 优化器参数
adam_beta1: 0.9
adam_beta2: 0.999
adam_weight_decay: 0.01

# 混合精度
mixed_precision: "bf16"           # 使用 bfloat16 混合精度
```

**重要参数说明：**

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `learning_rate` | 1e-4 到 5e-4 | 学习率，太大不稳定，太小学得慢 |
| `rank` | 8-32 | LoRA 秩，越大表达能力越强 |
| `max_train_steps` | 1000-5000 | 根据数据集大小调整 |
| `img_size` | 512-1024 | 取决于显存大小 |
| `gradient_accumulation_steps` | 1-8 | 用于模拟更大的批次 |

### 4.4 inference.py - 推理脚本

训练完成后，用这个脚本生成图片：

```python
# 1. 加载模型
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", 
    torch_dtype=torch.bfloat16
)

# 2. 加载 LoRA 权重
pipe.load_lora_weights('./output/checkpoint-1000')

# 3. 生成图片
image = pipe(
    prompt="你的提示词",
    width=1024,
    height=1024,
    num_inference_steps=50,
)

# 4. 保存图片
image.images[0].save("output.png")
```

---

## 5. 数据准备详解

数据准备是训练成功的关键！

### 5.1 数据集结构

**Qwen-Image 和 FLUX 的数据结构：**

```
my_dataset/
├── image1.jpg
├── image1.txt      # image1.jpg 的描述
├── image2.png
├── image2.txt      # image2.png 的描述
├── image3.jpg
├── image3.txt
└── ...
```

**Qwen-Image-Edit 的数据结构：**

```
my_edit_dataset/
├── images/
│   ├── img001.jpg
│   ├── img001.txt  # 目标图片描述
│   ├── img002.jpg
│   └── img002.txt
└── control/
    ├── img001.jpg  # 控制图（原始图）
    ├── img002.jpg
    └── ...
```

### 5.2 准备图片

**图片要求：**

1. **格式**：JPG、PNG、WEBP 都可以
2. **分辨率**：建议 1024×1024 或更高
3. **质量**：越高越好，避免模糊、有瑕疵的图片
4. **数量**：
   - 最少：10 张
   - 推荐：20-50 张
   - 理想：100+ 张

**图片类型建议：**

对于 FLUX 人像训练：
- 同一个人的不同照片
- 不同角度、表情、光线
- 高质量的清晰照片

对于 Qwen-Image 风格训练：
- 风格统一的图片集合
- 可以是特定艺术风格、特定主题

### 5.3 编写文本描述

**文本描述（Caption）非常重要！**

#### 5.3.1 Qwen-Image 描述示例

```
# image1.txt
一位年轻女性的专业肖像照，演播室灯光，优雅的姿势，看着镜头，柔和的阴影，高质量，详细的面部特征，电影般的灯光
```

#### 5.3.2 FLUX 描述示例（需要触发词）

```
# portrait1.txt
ohwx woman, professional headshot, studio lighting, elegant pose, looking at camera

# portrait2.txt
ohwx woman, casual outdoor photo, natural lighting, smiling, park background

# portrait3.txt
ohwx woman, close-up portrait, dramatic lighting, serious expression
```

**重要提示：**
- FLUX 训练需要使用触发词，如 `ohwx woman` 或 `ohwx man`
- Qwen-Image 不需要特殊触发词

#### 5.3.3 自动生成描述

如果你不想手写描述，可以使用 AI 工具自动生成：

1. **Florence-2**：https://huggingface.co/spaces/gokaygokay/Florence-2
   - 上传图片，自动生成英文描述
   
2. **BLIP-2**：另一个图像描述生成工具

3. **GPT-4V**：如果你有 ChatGPT Plus，可以上传图片让它写描述

### 5.4 验证数据集

使用项目提供的验证工具：

```bash
python utils/validate_dataset.py --path ./your_dataset
```

**验证内容：**
- ✅ 每张图片都有对应的 .txt 文件
- ✅ 文件命名正确
- ✅ 文本文件不为空
- ⚠️ 如有问题会提示具体是哪个文件

---

## 6. 训练流程详解

### 6.1 环境准备

#### 6.1.1 安装 Python

确保你有 Python 3.10：

```bash
python --version  # 应该显示 Python 3.10.x
```

#### 6.1.2 克隆仓库

```bash
git clone https://github.com/FlyMyAI/flymyai-lora-trainer
cd flymyai-lora-trainer
```

#### 6.1.3 安装依赖

```bash
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers
```

**requirements.txt 里有什么？**

```
accelerate==1.9.0       # 训练加速
diffusers               # Hugging Face 扩散模型库
transformers            # Transformer 模型库
peft==0.17.0           # LoRA 实现
torch                   # PyTorch 深度学习框架
...
```

### 6.2 准备训练

#### 6.2.1 选择训练脚本

根据你的需求选择：

| 需求 | 使用脚本 | 显存要求 |
|------|----------|----------|
| Qwen-Image LoRA | `train.py` | 40GB+ |
| Qwen-Image LoRA (低显存) | `train_4090.py` | 24GB |
| FLUX LoRA | `train_flux_lora.py` | 40GB+ |
| Qwen-Image-Edit LoRA | `train_qwen_edit_lora.py` | 40GB+ |

#### 6.2.2 修改配置文件

打开对应的 YAML 配置文件，修改：

```yaml
data_config:
  img_dir: ./your_dataset    # 改成你的数据集路径
  
output_dir: ./my_lora_output  # 改成你想要的输出路径

max_train_steps: 2000         # 根据数据集大小调整
```

#### 6.2.3 启动训练

```bash
# Qwen-Image LoRA
accelerate launch train.py --config ./train_configs/train_lora.yaml

# FLUX LoRA
accelerate launch train_flux_lora.py --config ./train_configs/train_flux_config.yaml
```

### 6.3 训练过程中会发生什么？

#### 6.3.1 初始化阶段

```
[INFO] Loading model from Qwen/Qwen-Image...
[INFO] Loading VAE...
[INFO] Loading transformer...
[INFO] Adding LoRA adapters...
[INFO] Total trainable parameters: 16.7M
```

**这个阶段做什么？**
- 下载预训练模型（如果本地没有）
- 加载模型组件
- 初始化 LoRA 层
- 准备训练

#### 6.3.2 训练阶段

```
Steps: 0%|          | 0/2000 [00:00<?, ?it/s]
Steps: 1%|▏         | 10/2000 [00:45<2:30:15, loss=0.1234]
Steps: 2%|▎         | 50/2000 [03:45<2:28:00, loss=0.0856]
```

**进度条显示：**
- 当前步数 / 总步数
- 预计剩余时间
- 当前损失值

**损失值（loss）的变化：**
- 开始时较大（比如 0.5）
- 逐渐下降（比如 0.1、0.05）
- 最终稳定在一个较小的值

**注意：**
- 损失值下降是好事，说明模型在学习
- 如果损失不再下降，可能需要调整学习率
- 如果损失突然升高，可能学习率太大

#### 6.3.3 保存检查点

每隔 `checkpointing_steps` 步，模型会保存一次：

```
[INFO] Saved checkpoint to ./output/checkpoint-250
[INFO] Saved checkpoint to ./output/checkpoint-500
[INFO] Saved checkpoint to ./output/checkpoint-750
```

**检查点包含：**
```
checkpoint-250/
└── pytorch_lora_weights.safetensors  # LoRA 权重文件
```

### 6.4 训练时间估算

**Qwen-Image LoRA (3000 步)：**
- RTX 4090 (24GB)：约 4-6 小时
- A100 (40GB)：约 2-3 小时

**FLUX LoRA (2000 步)：**
- RTX 4090 (24GB)：约 6-8 小时（使用优化版）
- A100 (80GB)：约 3-4 小时

### 6.5 训练完成

```
[INFO] Training completed!
[INFO] Final checkpoint saved to ./output/checkpoint-3000
```

**训练完成后，你会得到：**
- LoRA 权重文件 (`pytorch_lora_weights.safetensors`)
- 训练日志
- 可能还有一些示例生成图片（如果配置了采样）

---

## 7. 推理使用详解

训练完成后，让我们使用 LoRA 生成图片！

### 7.1 使用 Python 脚本

#### 7.1.1 Qwen-Image 推理

```python
from diffusers import DiffusionPipeline
import torch

# 加载基础模型
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# 加载你训练的 LoRA
pipe.load_lora_weights('./output/checkpoint-3000')

# 生成图片
prompt = "一位年轻女性的专业肖像照，演播室灯光，优雅的姿势"
image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    num_inference_steps=50,
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

# 保存
image.save("output.png")
```

#### 7.1.2 FLUX 推理

```python
from diffusers import DiffusionPipeline
import torch

# 加载 FLUX 模型
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# 加载 LoRA
pipe.load_lora_weights('./flux_output/checkpoint-2000')

# 生成图片（注意使用触发词）
prompt = "ohwx woman, professional headshot, studio lighting"
image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    num_inference_steps=30,
    guidance_scale=3.5,
).images[0]

image.save("flux_output.png")
```

### 7.2 使用项目的推理脚本

```bash
python inference.py \
  --model_name Qwen/Qwen-Image \
  --lora_weights ./output/checkpoint-3000 \
  --prompt "你的提示词" \
  --output_image output.png \
  --width 1024 \
  --height 1024 \
  --num_inference_steps 50
```

### 7.3 重要参数说明

#### 7.3.1 num_inference_steps（推理步数）

**这是什么？**
- 模型去噪的步数
- 步数越多，图片质量越好，但生成越慢

**建议值：**
- 快速测试：20-30 步
- 正常使用：30-50 步
- 高质量：50-100 步

#### 7.3.2 CFG Scale / Guidance Scale（引导强度）

**这是什么？**
- 控制模型遵循提示词的程度
- 值越大，越严格按照提示词生成
- 值越小，模型越自由发挥

**建议值：**
- Qwen-Image：3-7（`true_cfg_scale`）
- FLUX：2-5（`guidance_scale`）

#### 7.3.3 Width 和 Height（图片尺寸）

**建议值：**
- 常用：1024×1024（正方形）
- 横版：1280×720 或 1920×1080
- 竖版：720×1280

**注意：**
- 尺寸必须是 8 的倍数（Qwen）或 64 的倍数（FLUX）
- 尺寸越大，需要越多显存

#### 7.3.4 Seed（随机种子）

```python
generator = torch.Generator(device="cuda").manual_seed(42)
```

**作用：**
- 固定随机种子可以得到**可复现的结果**
- 同样的种子 + 同样的提示词 = 同样的图片
- 用于调试和比较

### 7.4 使用 ComfyUI（图形界面）

如果你不喜欢写代码，可以使用 ComfyUI：

1. 安装 ComfyUI
2. 下载 Qwen-Image 模型文件
3. 将你的 LoRA 文件放到 `ComfyUI/models/loras/`
4. 导入项目提供的工作流文件 `qwen_image_lora_example.json`
5. 在界面上选择你的 LoRA
6. 输入提示词，生成图片

---

## 8. 实战示例

让我们通过一个完整的例子，从头到尾走一遍流程。

### 8.1 场景：训练一个特定人物的 FLUX LoRA

**目标：** 训练一个能生成特定人物（比如你自己或你的朋友）照片的 LoRA

#### 步骤 1：准备数据

收集 20-30 张这个人的照片：
- 不同角度
- 不同表情
- 不同光线
- 不同背景

```
my_person_dataset/
├── photo01.jpg
├── photo02.jpg
├── photo03.jpg
...
├── photo20.jpg
```

#### 步骤 2：生成文本描述

为每张图片写描述（或使用 Florence-2 自动生成）：

```
# photo01.txt
ohwx woman, professional headshot, studio lighting, neutral expression

# photo02.txt
ohwx woman, outdoor photo, natural sunlight, smiling, park background

# photo03.txt
ohwx woman, close-up portrait, soft lighting, looking at camera
```

**关键点：**
- 每个描述都以 `ohwx woman` 开头（触发词）
- 描述要准确反映图片内容

#### 步骤 3：验证数据集

```bash
python utils/validate_dataset.py --path ./my_person_dataset
```

输出：
```
✅ Found 20 images
✅ All images have corresponding text files
✅ No issues found
```

#### 步骤 4：修改配置

编辑 `train_configs/train_flux_config.yaml`：

```yaml
data_config:
  img_dir: ./my_person_dataset  # 你的数据集路径
  img_size: 1024
  train_batch_size: 1
  
output_dir: ./my_person_lora
max_train_steps: 1500            # 20 张图，1500 步足够
learning_rate: 4e-4
rank: 16
```

#### 步骤 5：开始训练

```bash
accelerate launch train_flux_lora.py --config ./train_configs/train_flux_config.yaml
```

等待 6-8 小时（RTX 4090）...

#### 步骤 6：测试 LoRA

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
pipe.load_lora_weights('./my_person_lora/checkpoint-1500')

# 测试不同的场景
prompts = [
    "ohwx woman, professional business photo, wearing suit, office background",
    "ohwx woman, casual beach photo, summer vibes, sunset lighting",
    "ohwx woman, artistic portrait, dramatic lighting, black and white",
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, num_inference_steps=30).images[0]
    image.save(f"test_{i}.png")
```

#### 步骤 7：评估结果

查看生成的图片：
- ✅ 人物特征是否准确？
- ✅ 图片质量是否高？
- ✅ 是否能在不同场景下保持一致？

**如果效果不好：**
- 尝试增加训练步数
- 检查数据集质量
- 调整学习率
- 增加更多训练图片

---

## 9. 常见问题

### 9.1 训练相关

#### Q1: 显存不够怎么办？

**解决方案：**

1. 使用低显存版本训练脚本：
   ```bash
   accelerate launch train_4090.py --config ./train_configs/train_lora_4090.yaml
   ```

2. 减小批次大小：
   ```yaml
   train_batch_size: 1  # 改为 1
   ```

3. 减小图片大小：
   ```yaml
   img_size: 512  # 从 1024 改为 512
   ```

4. 使用梯度累积：
   ```yaml
   gradient_accumulation_steps: 4  # 模拟更大的批次
   ```

#### Q2: 训练速度太慢怎么办？

**解决方案：**

1. 使用混合精度：
   ```yaml
   mixed_precision: "bf16"  # 或 "fp16"
   ```

2. 增加数据加载线程：
   ```yaml
   num_workers: 8  # 增加到 8
   ```

3. 使用更快的存储（SSD 而不是 HDD）

4. 减少训练步数（但可能影响效果）

#### Q3: 损失不下降怎么办？

**可能的原因和解决方案：**

1. **学习率太小**：
   ```yaml
   learning_rate: 5e-4  # 从 1e-4 增加到 5e-4
   ```

2. **数据问题**：
   - 检查图片和文本是否对应
   - 确保文本描述准确

3. **训练步数太少**：
   - 增加 `max_train_steps`

#### Q4: 过拟合怎么办？

**症状：**
- 训练损失很低，但生成效果不好
- 只能生成和训练集很像的图片

**解决方案：**

1. 增加数据集：
   - 至少 30-50 张图片

2. 使用 caption dropout：
   ```yaml
   caption_dropout_rate: 0.1  # 10% 的概率丢弃文本
   ```

3. 减少训练步数

4. 降低 LoRA rank：
   ```yaml
   rank: 8  # 从 16 降到 8
   ```

### 9.2 推理相关

#### Q1: 生成的图片不像训练的内容？

**解决方案：**

1. **检查触发词**（FLUX 必须）：
   ```python
   prompt = "ohwx woman, ..."  # 一定要加触发词
   ```

2. **增加推理步数**：
   ```python
   num_inference_steps=50  # 从 20 增加到 50
   ```

3. **调整 CFG/Guidance Scale**：
   ```python
   true_cfg_scale=6  # Qwen，尝试 5-7
   guidance_scale=4  # FLUX，尝试 3-5
   ```

#### Q2: 生成的图片质量不好？

**解决方案：**

1. 增加推理步数（50-100 步）
2. 使用更好的提示词
3. 尝试不同的随机种子
4. 检查 LoRA 权重是否正确加载

#### Q3: 生成速度太慢？

**解决方案：**

1. 使用量化推理：
   ```python
   # 使用 inference.py，它包含量化优化
   ```

2. 减少推理步数（但会影响质量）

3. 使用更小的图片尺寸

4. 使用 CPU offload：
   ```python
   pipe.enable_model_cpu_offload()
   ```

### 9.3 环境相关

#### Q1: 安装依赖失败？

**解决方案：**

1. 确保 Python 版本正确（3.10）
2. 更新 pip：
   ```bash
   pip install --upgrade pip
   ```
3. 单独安装失败的包
4. 使用 conda 环境

#### Q2: CUDA out of memory 错误？

这是显存不足，参见 Q9.1.1 的解决方案。

---

## 🎓 总结

恭喜你读到这里！现在你应该对这个项目有了全面的理解。

### 你学到了什么？

✅ LoRA 是一种高效的模型微调技术
✅ 扩散模型的工作原理
✅ 项目的目录结构和文件作用
✅ 训练数据的准备方法
✅ 训练和推理的完整流程
✅ 常见问题的解决方案

### 下一步做什么？

1. **动手实践**：准备一个小数据集，尝试训练第一个 LoRA
2. **实验参数**：尝试不同的学习率、rank 等参数
3. **优化提示词**：学习如何写出更好的 prompt
4. **分享成果**：将你训练的 LoRA 分享到 Hugging Face

### 学习资源

- **Hugging Face Diffusers 文档**：https://huggingface.co/docs/diffusers
- **PEFT (LoRA) 文档**：https://huggingface.co/docs/peft
- **FlyMy.AI 文档**：https://docs.flymy.ai
- **Discord 社区**：https://discord.com/invite/t6hPBpSebw

---

## 📞 需要帮助？

如果你在使用过程中遇到问题：

1. **查看本教程**：仔细阅读相关章节
2. **查看常见问题**：第 9 章可能已经回答了你的问题
3. **查看 GitHub Issues**：看看别人是否遇到过类似问题
4. **加入 Discord**：在社区寻求帮助
5. **提交 Issue**：如果发现 bug，在 GitHub 提交 issue

---

**祝你训练愉快！🚀**
