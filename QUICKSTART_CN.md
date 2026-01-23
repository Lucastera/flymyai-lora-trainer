# 快速入门指南（零基础版）

## 🎯 5 分钟快速了解

### 这个项目能做什么？

简单来说：**让你能用自己的图片训练 AI，生成你想要的风格的图片。**

比如：
- 训练一个能生成特定人物照片的 AI
- 训练一个能生成特定艺术风格图片的 AI
- 训练一个能编辑图片的 AI

---

## 🚀 最简单的使用流程

### 步骤 1：准备环境（10 分钟）

```bash
# 1. 克隆项目
git clone https://github.com/FlyMyAI/flymyai-lora-trainer
cd flymyai-lora-trainer

# 2. 安装依赖
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers
```

### 步骤 2：准备数据（30 分钟）

创建一个文件夹，放入你的图片：

```
my_dataset/
├── photo1.jpg
├── photo1.txt  # 写上：一位年轻女性的照片
├── photo2.jpg
├── photo2.txt  # 写上：一位年轻女性在公园的照片
└── ...
```

**最少 10 张图片，推荐 20-50 张。**

### 步骤 3：修改配置（5 分钟）

打开 `train_configs/train_lora.yaml`，只需要改一行：

```yaml
data_config:
  img_dir: ./my_dataset  # 改成你的数据集路径
```

### 步骤 4：开始训练（2-6 小时）

```bash
accelerate launch train.py --config ./train_configs/train_lora.yaml
```

然后等待训练完成...

### 步骤 5：使用模型生成图片（1 分钟）

```python
from diffusers import DiffusionPipeline
import torch

# 加载模型和你训练的 LoRA
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.load_lora_weights('./output/checkpoint-3000')

# 生成图片
image = pipe(
    prompt="你想生成的内容描述",
    width=1024,
    height=1024,
    num_inference_steps=50
).images[0]

image.save("output.png")
```

---

## 🎓 三个等级的学习路径

### 🌱 初级：我只想快速上手

**推荐阅读：**
1. README.md 的 Installation 和 Training 部分
2. 本文件（QUICKSTART_CN.md）
3. 动手训练一次

**时间投入：** 1-2 小时

### 🌿 中级：我想理解原理

**推荐阅读：**
1. QUICKSTART_CN.md（本文件）
2. TUTORIAL_CN.md 的第 1-3 章
3. 尝试调整参数，多次训练

**时间投入：** 3-5 小时

### 🌳 高级：我想深入掌握

**推荐阅读：**
1. 完整阅读 TUTORIAL_CN.md
2. 阅读核心代码文件
3. 尝试不同模型和数据集
4. 解决各种问题

**时间投入：** 1-2 天

---

## 💡 最常见的 3 个问题

### Q1: 我的显卡够用吗？

**推荐配置：**
- RTX 4090 (24GB)：✅ 可以训练所有模型
- RTX 3090 (24GB)：✅ 可以训练，使用优化版
- RTX 3080 (10GB)：❌ 显存不够

**如果显存不够：**
- 使用 `train_4090.py`（低显存优化版）
- 减小图片大小
- 减小批次大小

### Q2: 训练需要多少张图片？

**最少：** 10 张
**推荐：** 20-50 张
**理想：** 100+ 张

图片越多，效果越好，但也要看具体任务。

### Q3: 训练要多久？

**取决于：**
- 你的 GPU
- 数据集大小
- 训练步数

**参考时间：**
- RTX 4090，30 张图，3000 步：4-6 小时
- A100，30 张图，3000 步：2-3 小时

---

## 🎯 不同任务的快速配置

### 任务 1：训练特定人物（FLUX）

**数据准备：**
- 20-30 张同一个人的照片
- 每张图的描述都以 `ohwx woman` 开头

**配置：**
```yaml
# train_configs/train_flux_config.yaml
max_train_steps: 1500
learning_rate: 4e-4
rank: 16
```

**训练：**
```bash
accelerate launch train_flux_lora.py --config ./train_configs/train_flux_config.yaml
```

### 任务 2：训练特定风格（Qwen-Image）

**数据准备：**
- 30-50 张相同风格的图片
- 每张图的描述要详细

**配置：**
```yaml
# train_configs/train_lora.yaml
max_train_steps: 3000
learning_rate: 1e-4
rank: 16
```

**训练：**
```bash
accelerate launch train.py --config ./train_configs/train_lora.yaml
```

### 任务 3：图像编辑（Qwen-Image-Edit）

**数据准备：**
```
dataset/
├── images/       # 目标图片
│   ├── img1.jpg
│   └── img1.txt
└── control/      # 控制图（原图）
    └── img1.jpg
```

**训练：**
```bash
accelerate launch train_qwen_edit_lora.py --config ./train_configs/train_lora_qwen_edit.yaml
```

---

## 🔧 参数速查表

### 最重要的 5 个参数

| 参数 | 作用 | 推荐值 | 调整建议 |
|------|------|--------|----------|
| `learning_rate` | 学习速度 | 1e-4 到 5e-4 | 太慢则增大，不稳定则减小 |
| `rank` | LoRA 大小 | 8-32 | 数据多用 32，数据少用 8-16 |
| `max_train_steps` | 训练步数 | 1500-3000 | 数据少用少步数 |
| `img_size` | 图片大小 | 512-1024 | 显存小用 512 |
| `train_batch_size` | 批次大小 | 1-2 | 显存大可以用 2 |

### 推理时重要参数

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `num_inference_steps` | 生成质量 | 30-50 |
| `guidance_scale` / `true_cfg_scale` | 遵循提示词程度 | 3-7 |
| `width` x `height` | 图片尺寸 | 1024×1024 |

---

## 📚 下一步学习

**如果你想深入理解：**
→ 阅读 [TUTORIAL_CN.md](./TUTORIAL_CN.md)

**如果你遇到问题：**
→ 查看 TUTORIAL_CN.md 第 9 章「常见问题」

**如果你想看示例：**
→ 查看 README.md 的 Sample Output 部分

**如果你需要帮助：**
→ 加入 [Discord 社区](https://discord.com/invite/t6hPBpSebw)

---

## 🎉 开始你的第一次训练！

不要害怕，大胆尝试！

1. 准备 10-20 张图片
2. 写简单的描述
3. 运行训练命令
4. 等待几小时
5. 享受成果！

**记住：** 第一次训练可能效果不完美，多试几次就会越来越好！

**祝你好运！** 🚀
