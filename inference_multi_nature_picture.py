import torch
from diffusers import DiffusionPipeline
from optimum.quanto import quantize, qfloat8, freeze
from tqdm.auto import tqdm
from accelerate import PartialState
import os
import time
import json
from datetime import datetime


# ===== 自然图片生成的Prompt列表 (类似GenEval风格) =====
NATURAL_PROMPTS = [
    # 单物体
    "a red apple on a wooden table",
    "a blue car parked on the street",
    "a yellow sunflower in a garden",
    "a white cat sitting on a sofa",
    "a green frog on a lily pad",
    
    # 多物体
    "a red ball and a blue cube on a white surface",
    "a black dog next to a brown fence",
    "a pink rose in a glass vase",
    "a orange butterfly on a purple flower",
    "a silver laptop on a black desk",
    
    # 颜色组合
    "a red and blue striped umbrella",
    "a green tree with yellow leaves",
    "a white bird with black wings",
    "a golden crown on a red cushion",
    "a blue ocean under an orange sunset",
    
    # 场景
    "a cozy living room with a brown leather sofa",
    "a snowy mountain with a blue sky",
    "a colorful hot air balloon in the sky",
    "a vintage red bicycle by a brick wall",
    "a rainbow over a green meadow",
    
    # 复杂场景
    "a chef cooking in a modern kitchen",
    "a child playing with a red ball in the park",
    "a sailboat on calm blue water",
    "a stack of colorful books on a shelf",
    "a cat sleeping on a windowsill with sunlight",
    
    # 抽象/艺术风格
    "a geometric pattern with red, blue and yellow shapes",
    "a watercolor painting of a sunset",
    "a minimalist room with a single green plant",
    "a neon sign glowing in the dark",
    "a reflection of trees in a calm lake",
]


def generate_output_dir_name(model_name, lora_weights="", compare_mode=False):
    """Auto-generate output directory name"""
    model_short = model_name.split('/')[-1]
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    if compare_mode:
        # 对比模式
        lora_flag = ""
        if lora_weights:
            if "lora_saves_" in lora_weights:
                lora_part = lora_weights.split("lora_saves_")[-1]
                lora_name = lora_part.split("/")[0]
                if "checkpoint-" in lora_part:
                    ckpt_num = lora_part.split("checkpoint-")[-1].split("/")[0]
                    lora_flag = f"_vs_lora_{lora_name}_ckpt{ckpt_num}"
                else:
                    lora_flag = f"_vs_lora_{lora_name}"
            else:
                lora_flag = "_vs_lora"
        dir_name = f"{model_short}{lora_flag}_compare_{timestamp}"
    else:
        dir_name = f"{model_short}_natural_{timestamp}"
    
    return os.path.join("outputs", dir_name)


def save_config(output_dir, config, total_prompts):
    """Save configuration to file"""
    config_path = os.path.join(output_dir, "generation_config.txt")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"Generation Config - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        f.write(f"total_prompts: {total_prompts}\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def check_already_generated(output_dir, prompt_idx, suffix=""):
    """Check if image has already been generated"""
    output_path = os.path.join(output_dir, f"img_{prompt_idx:04d}{suffix}.png")
    return os.path.exists(output_path)


def main():
    # ===== Configuration Parameters =====
    model_name = "Qwen/Qwen-Image"
    lora_weights = "lora_saves_level_3/checkpoint-3000/pytorch_lora_weights.safetensors"
    
    # 对比模式: 同时生成有/无LoRA的图片
    compare_mode = True
    
    # 使用自定义prompt列表 (可以修改NATURAL_PROMPTS或传入自己的列表)
    prompts = NATURAL_PROMPTS
    
    # Generation parameters
    negative_prompt = " "
    width = 384
    height = 384
    num_inference_steps = 50
    true_cfg_scale = 5.0
    base_seed = 42
    batch_size = 64
    
    # Auto-generate output directory name
    output_dir = generate_output_dir_name(model_name, lora_weights, compare_mode)
    os.makedirs(output_dir, exist_ok=True)
    
    distributed_state = PartialState()
    
    # Save prompts list
    if distributed_state.is_main_process:
        prompts_file = os.path.join(output_dir, "prompts.json")
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump({'prompts': prompts}, f, ensure_ascii=False, indent=2)
        print(f"Output directory: {output_dir}")
        print(f"Total prompts: {len(prompts)}")
        print(f"Compare mode: {compare_mode}")
    
    # Save configuration
    config = {
        'model_name': model_name,
        'lora_weights': lora_weights if lora_weights else 'None',
        'compare_mode': compare_mode,
        'output_dir': output_dir,
        'negative_prompt': negative_prompt,
        'width': width,
        'height': height,
        'num_inference_steps': num_inference_steps,
        'true_cfg_scale': true_cfg_scale,
        'base_seed': base_seed,
        'batch_size': batch_size,
    }
    
    if distributed_state.is_main_process:
        save_config(output_dir, config, len(prompts))

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if distributed_state.is_main_process:
        print(f"Using {distributed_state.num_processes} GPUs")
        print(f"Batch size: {batch_size}")

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

    # 加载LoRA (如果有)
    has_lora = bool(lora_weights)
    if has_lora:
        pipe.load_lora_weights(lora_weights, adapter_name="lora")
        if distributed_state.is_main_process:
            print(f"Loaded LoRA weights from: {lora_weights}")

    if distributed_state.is_main_process:
        print("Applying qfloat8 quantization...")

    all_blocks = list(pipe.transformer.transformer_blocks)
    for block in tqdm(all_blocks, disable=not distributed_state.is_main_process):
        block.to("cuda", dtype=torch_dtype)
        quantize(block, weights=qfloat8)
        freeze(block)
        block.to('cpu')
    pipe.transformer.to("cuda", dtype=torch_dtype)
    quantize(pipe.transformer, weights=qfloat8)
    freeze(pipe.transformer)

    if distributed_state.is_main_process:
        print("Quantization complete.")

    pipe.enable_model_cpu_offload(gpu_id=distributed_state.process_index)

    # 构建任务列表
    # 如果是对比模式，每个prompt生成两次：一次有LoRA，一次无LoRA
    if compare_mode and has_lora:
        # (prompt_idx, use_lora)
        task_list = []
        for idx in range(len(prompts)):
            task_list.append((idx, False))  # 无LoRA
            task_list.append((idx, True))   # 有LoRA
    else:
        task_list = [(idx, has_lora) for idx in range(len(prompts))]
    
    task_indices = list(range(len(task_list)))

    # Start timing
    distributed_state.wait_for_everyone()
    start_time = time.time()

    with distributed_state.split_between_processes(task_indices) as local_indices:
        local_indices = list(local_indices)
        local_start = time.time()

        # 按use_lora分组处理，减少LoRA切换次数
        # 先处理所有无LoRA的，再处理所有有LoRA的
        local_tasks = [task_list[i] for i in local_indices]
        
        # 分成两组
        no_lora_tasks = [(i, task) for i, task in zip(local_indices, local_tasks) if not task[1]]
        with_lora_tasks = [(i, task) for i, task in zip(local_indices, local_tasks) if task[1]]
        
        for group_name, group_tasks, use_lora in [
            ("without LoRA", no_lora_tasks, False),
            ("with LoRA", with_lora_tasks, True)
        ]:
            if not group_tasks:
                continue
                
            # 设置LoRA状态
            if has_lora:
                if use_lora:
                    pipe.set_adapters(["lora"], adapter_weights=[1.0])
                else:
                    pipe.set_adapters(["lora"], adapter_weights=[0.0])
            
            if distributed_state.is_main_process:
                print(f"\nGenerating {group_name}...")
            
            pbar = tqdm(
                range(0, len(group_tasks), batch_size),
                desc=f"GPU {distributed_state.process_index} - {group_name}",
                disable=not distributed_state.is_local_main_process
            )

            for i in pbar:
                batch_start = time.time()
                batch_tasks = group_tasks[i:i+batch_size]
                batch_prompt_indices = [task[0] for _, task in batch_tasks]
                batch_prompts = [prompts[task[0]] for _, task in batch_tasks]
                batch_neg_prompts = [negative_prompt] * len(batch_tasks)

                seed = base_seed + batch_prompt_indices[0]
                generator = torch.Generator(device="cpu").manual_seed(seed)

                images = pipe(
                    prompt=batch_prompts,
                    negative_prompt=batch_neg_prompts,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=generator,
                ).images

                batch_time = time.time() - batch_start

                for j, (_, task) in enumerate(batch_tasks):
                    prompt_idx, task_use_lora = task
                    if compare_mode:
                        suffix = "_lora" if task_use_lora else "_base"
                    else:
                        suffix = ""
                    output_path = os.path.join(output_dir, f"img_{prompt_idx:04d}{suffix}.png")
                    images[j].save(output_path)

                pbar.set_postfix({'batch_time': f'{batch_time:.2f}s', 'per_img': f'{batch_time/len(batch_tasks):.2f}s'})

        local_time = time.time() - local_start
        print(f"[GPU {distributed_state.process_index}] Total local time: {local_time:.2f}s")

    distributed_state.wait_for_everyone()
    total_time = time.time() - start_time

    if distributed_state.is_main_process:
        total_images = len(task_list)
        print(f"\n{'='*50}")
        print(f"Total images generated: {total_images}")
        if compare_mode:
            print(f"  - Base model: {len(prompts)} images (*_base.png)")
            print(f"  - With LoRA: {len(prompts)} images (*_lora.png)")
        print(f"Batch size: {batch_size}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per image: {total_time/total_images:.2f}s")
        print(f"All images saved to {output_dir}/")
        print(f"Prompts saved to {output_dir}/prompts.json")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()