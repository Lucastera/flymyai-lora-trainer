import torch
from diffusers import DiffusionPipeline
from optimum.quanto import quantize, qfloat8, freeze
from tqdm.auto import tqdm
from accelerate import PartialState
import os
import time
import json
from datetime import datetime
from PIL import Image
import argparse
import random
import inspect


def parse_args():
    parser = argparse.ArgumentParser(description='Image generation inference')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='Tongyi-MAI/Z-Image',
                        help='Model name or path')
    parser.add_argument('--lora_weights', type=str, default='',
                        help='LoRA weights path (optional)')
    
    # Dataset configuration
    parser.add_argument('--base_dir', type=str, 
                        default='VIOLIN_v2/data',
                        help='Base directory containing test.jsonl')
    parser.add_argument('--variations', type=str, default='123',
                        help='Variation types to generate, e.g., "123", "12", "1" (default: "123")')
    
    # Sampling configuration
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None = all data). Samples randomly if specified.')
    parser.add_argument('--sample_seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    
    # Generation parameters
    parser.add_argument('--negative_prompt', type=str, default=' ',
                        help='Negative prompt')
    parser.add_argument('--width', type=int, default=256,
                        help='Image width (default: 256)')
    parser.add_argument('--height', type=int, default=256,
                        help='Image height (default: 256)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--cfg_scale', type=float, default=5.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generation')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory (auto-generated if not specified)')
    
    return parser.parse_args()


def normalize_path(path):
    """
    Normalize path separators for current OS
    Convert backslashes to forward slashes on Unix systems
    """
    path = path.replace('\\', '/')
    return os.path.normpath(path)


def parse_variations(variations_str):
    """
    Parse variations string into list of integers
    
    Args:
        variations_str: String like "123", "12", "1", etc.
        
    Returns:
        List of integers, e.g., [1,2,3], [1,2], [1]
    """
    variations = []
    for char in variations_str:
        if char.isdigit() and 1 <= int(char) <= 3:
            var = int(char)
            if var not in variations:
                variations.append(var)
    
    if not variations:
        raise ValueError(f"Invalid variations string: {variations_str}. Must contain digits 1-3.")
    
    return sorted(variations)


def generate_output_dir_name(model_name, variations, width, height, lora_weights="", max_samples=None, base_seed=None):
    """
    Auto-generate output directory name based on configuration
    
    Format: 
      - {model_short_name}_{lora_name}_V{variations}_{resolution}_seed{N}
      - With sampling: add _sample{N} suffix
    """
    model_short = model_name.split('/')[-1]
    
    lora_flag = ""
    if lora_weights:
        if "lora_saves_" in lora_weights:
            lora_part = lora_weights.split("lora_saves_")[-1]
            lora_name = lora_part.split("/")[0]
            if "checkpoint-" in lora_part:
                ckpt_num = lora_part.split("checkpoint-")[-1].split("/")[0]
                lora_flag = f"_lora_{lora_name}_ckpt{ckpt_num}"
            else:
                lora_flag = f"_lora_{lora_name}"
        else:
            timestamp = datetime.now().strftime("%m%d%H%M")
            lora_flag = f"_lora_{timestamp}"
    
    v_str = "V" + "-".join(map(str, variations))
    resolution_str = f"{width}x{height}"
    sample_suffix = f"_sample{max_samples}" if max_samples is not None else ""
    seed_suffix = f"_seed{base_seed}" if base_seed is not None else ""  # <<< 新增
    
    dir_name = f"{model_short}{lora_flag}_{v_str}_{resolution_str}{sample_suffix}{seed_suffix}"  # <<< 改这一行
    
    return os.path.join("/data4/kuan/outputs", dir_name)


def load_or_create_sample_list(output_dir, base_dir, variations, max_samples, sample_seed):
    """
    Load existing sample list or create a new one
    
    Args:
      output_dir: Output directory
      base_dir: Base directory containing test.jsonl
      variations: List of variation types [1,2,3]
      max_samples: Maximum number of samples (None = all)
      sample_seed: Random seed for sampling
      
    Returns:
      List of task dictionaries
    """
    list_file = os.path.join(output_dir, "sample_list.json")
    
    if os.path.exists(list_file):
        print(f"Loading existing sample list from: {list_file}")
        with open(list_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        for task in saved_data['samples']:
            if 'ground_truth' in task:
                task['ground_truth'] = normalize_path(task['ground_truth'])
        
        print(f"Loaded {len(saved_data['samples'])} samples from saved list")
        return saved_data['samples']
    
    else:
        print("Creating new sample list...")
        all_tasks = []
        
        jsonl_path = os.path.join(base_dir, "test.jsonl")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line.strip())
                var = task['variation']
                
                # Only process Variation 1-3
                if var not in variations:
                    continue
                
                if 'ground_truth' in task:
                    rel_path = normalize_path(task['ground_truth'])
                    task['ground_truth'] = os.path.join(base_dir, rel_path)
                
                all_tasks.append(task)
        
        print(f"Total tasks matching criteria: {len(all_tasks)}")
        
        if max_samples is not None and len(all_tasks) > max_samples:
            random.seed(sample_seed)
            sampled_tasks = random.sample(all_tasks, max_samples)
            print(f"Randomly sampled {max_samples} tasks from {len(all_tasks)} (seed={sample_seed})")
            tasks = sampled_tasks
        else:
            if max_samples is not None:
                print(f"Using all {len(all_tasks)} tasks (less than max_samples={max_samples})")
            else:
                print(f"Using all {len(all_tasks)} tasks (no sampling)")
            tasks = all_tasks
        
        sample_info = {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'base_dir': base_dir,
            'variations': variations,
            'max_samples': max_samples,
            'sample_seed': sample_seed,
            'total_available': len(all_tasks),
            'samples': tasks
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(list_file, 'w', encoding='utf-8') as f:
            json.dump(sample_info, f, ensure_ascii=False, indent=2)
        
        print(f"Created and saved {len(tasks)} samples to: {list_file}")
        return tasks


def task_to_filename(task, width, height):
    """
    Convert task to safe output filename with resolution
    
    Args:
        task: Task dict
        width: Image width
        height: Image height
        
    Returns:
        Filename without extension (e.g., "V1_id_1_256x256")
    """
    var = task['variation']
    task_id = task['id']
    resolution_str = f"{width}x{height}"
    
    return f"V{var}_{task_id}_{resolution_str}"


def check_already_generated(output_dir, task, width, height):
    """Check if image has already been generated"""
    filename = task_to_filename(task, width, height)
    output_path = os.path.join(output_dir, f"{filename}_gen.png")
    return os.path.exists(output_path)


def save_config(output_dir, config, total_tasks):
    """Save configuration to file"""
    config_path = os.path.join(output_dir, "generation_config.txt")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"Generation Config - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        f.write(f"total_data: {total_tasks}\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def apply_quantization(pipe, torch_dtype, distributed_state):
    """
    ✨ 智能量化：自动适配不同模型架构
    
    支持:
    - FLUX.2: single_transformer_blocks
    - Qwen-Image: transformer_blocks
    - Z-Image: 整体量化
    """
    print("=" * 60)
    print("🔍 Detecting model architecture...")
    
    # 检测 FLUX.2 架构 (single_transformer_blocks)
    if hasattr(pipe.transformer, 'single_transformer_blocks'):
        if distributed_state.is_main_process:
            print("✅ Detected FLUX.2 architecture (single_transformer_blocks)")
            print("Applying block-level quantization...")
        
        all_blocks = list(pipe.transformer.single_transformer_blocks)
        for block in tqdm(all_blocks, disable=not distributed_state.is_main_process, desc="Quantizing FLUX blocks"):
            block.to("cuda", dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')
    
    # 检测 Qwen-Image 架构 (transformer_blocks)
    elif hasattr(pipe.transformer, 'transformer_blocks'):
        if distributed_state.is_main_process:
            print("✅ Detected Qwen-Image architecture (transformer_blocks)")
            print("Applying block-level quantization...")
        
        all_blocks = list(pipe.transformer.transformer_blocks)
        for block in tqdm(all_blocks, disable=not distributed_state.is_main_process, desc="Quantizing Qwen blocks"):
            block.to("cuda", dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')
    
    # 其他模型（如 Z-Image）：整体量化
    else:
        if distributed_state.is_main_process:
            print("✅ Using whole-model quantization (no block structure detected)")
    
    # 最后对整个 transformer 量化（所有模型通用）
    pipe.transformer.to("cuda", dtype=torch_dtype)
    quantize(pipe.transformer, weights=qfloat8)
    freeze(pipe.transformer)
    
    if distributed_state.is_main_process:
        print("✅ Quantization complete!")
    print("=" * 60)


def prepare_generation_kwargs(pipe, batch_prompts, batch_neg_prompts, args, generator):
    """
    ✨ 智能参数准备：自动检测模型支持的参数
    
    支持:
    - FLUX.2: guidance_scale
    - Qwen-Image: true_cfg_scale
    - 自动检测 negative_prompt 支持
    """
    gen_kwargs = {
        "prompt": batch_prompts,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "generator": generator,
    }
    
    # 自动检测支持的参数
    sig = inspect.signature(pipe.__call__)
    
    # 检测 negative_prompt
    if 'negative_prompt' in sig.parameters:
        gen_kwargs["negative_prompt"] = batch_neg_prompts
    
    # 检测 CFG 参数（优先级：true_cfg_scale > guidance_scale）
    if 'true_cfg_scale' in sig.parameters:
        gen_kwargs["true_cfg_scale"] = args.cfg_scale
    elif 'guidance_scale' in sig.parameters:
        gen_kwargs["guidance_scale"] = args.cfg_scale
    
    return gen_kwargs


def main():
    args = parse_args()
    
    variations = parse_variations(args.variations)
    print(f"Processing variations: {variations}")
    print(f"Resolution: {args.width}x{args.height}")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = generate_output_dir_name(
            args.model_name, 
            variations, 
            args.width,
            args.height,
            args.lora_weights,
            args.max_samples,
            args.base_seed
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    distributed_state = PartialState()
    
    if distributed_state.is_main_process:
        tasks = load_or_create_sample_list(
            output_dir,
            args.base_dir,
            variations,
            args.max_samples,
            args.sample_seed
        )
    
    distributed_state.wait_for_everyone()
    
    if not distributed_state.is_main_process:
        list_file = os.path.join(output_dir, "sample_list.json")
        with open(list_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        tasks = saved_data['samples']
        for task in tasks:
            if 'ground_truth' in task:
                task['ground_truth'] = normalize_path(task['ground_truth'])
    
    config = {
        'model_name': args.model_name,
        'lora_weights': args.lora_weights if args.lora_weights else 'None',
        'output_dir': output_dir,
        'base_dir': args.base_dir,
        'variations': variations,
        'max_samples': args.max_samples if args.max_samples is not None else 'None (full data)',
        'sample_seed': args.sample_seed,
        'negative_prompt': args.negative_prompt,
        'width': args.width,
        'height': args.height,
        'num_inference_steps': args.num_inference_steps,
        'cfg_scale': args.cfg_scale,
        'base_seed': args.base_seed,
        'batch_size': args.batch_size,
    }
    
    pending_tasks = [task for task in tasks if not check_already_generated(output_dir, task, args.width, args.height)]
    
    if distributed_state.is_main_process:
        print(f"Output directory: {output_dir}")
        print(f"Total tasks in sample list: {len(tasks)}")
        print(f"Already generated: {len(tasks) - len(pending_tasks)}")
        print(f"Pending generation: {len(pending_tasks)}")
        save_config(output_dir, config, len(tasks))
    
    if len(pending_tasks) == 0:
        if distributed_state.is_main_process:
            print("All images already generated. Exiting.")
        return
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if distributed_state.is_main_process:
        print(f"Using {distributed_state.num_processes} GPUs")
        print(f"Batch size: {args.batch_size}")
        print(f"Loading model from: {args.model_name}")

    pipe = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch_dtype)

    if args.lora_weights:
        if distributed_state.is_main_process:
            print(f"Loading LoRA weights from: {args.lora_weights}")
        pipe.load_lora_weights(args.lora_weights, adapter_name="lora")

    # ✨ 使用智能量化函数
    apply_quantization(pipe, torch_dtype, distributed_state)

    pipe.enable_model_cpu_offload(gpu_id=distributed_state.process_index)

    task_indices = list(range(len(pending_tasks)))

    distributed_state.wait_for_everyone()
    start_time = time.time()

    with distributed_state.split_between_processes(task_indices) as local_indices:
        local_indices = list(local_indices)
        local_start = time.time()

        pbar = tqdm(
            range(0, len(local_indices), args.batch_size),
            desc=f"GPU {distributed_state.process_index}",
            disable=not distributed_state.is_local_main_process
        )

        for i in pbar:
            batch_start = time.time()
            batch_indices = local_indices[i:i+args.batch_size]
            batch_tasks = [pending_tasks[idx] for idx in batch_indices]
            
            batch_prompts = [task['prompt'] for task in batch_tasks]
            batch_neg_prompts = [args.negative_prompt] * len(batch_indices)
            
            seed = args.base_seed + batch_indices[0]
            generator = torch.Generator(device="cpu").manual_seed(seed)

            # ✨ 使用智能参数准备函数
            gen_kwargs = prepare_generation_kwargs(
                pipe, 
                batch_prompts, 
                batch_neg_prompts, 
                args, 
                generator
            )
            
            images = pipe(**gen_kwargs).images
            batch_time = time.time() - batch_start

            for j, idx in enumerate(batch_indices):
                filename = task_to_filename(batch_tasks[j], args.width, args.height)
                output_path = os.path.join(output_dir, f"{filename}_gen.png")
                images[j].save(output_path)

            pbar.set_postfix({'batch_time': f'{batch_time:.2f}s', 'per_img': f'{batch_time/len(batch_indices):.2f}s'})

        local_time = time.time() - local_start
        print(f"[GPU {distributed_state.process_index}] Total local time: {local_time:.2f}s")

    distributed_state.wait_for_everyone()
    total_time = time.time() - start_time

    if distributed_state.is_main_process:
        print(f"\n{'='*50}")
        print(f"Total images generated: {len(pending_tasks)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per image: {total_time/len(pending_tasks):.2f}s")
        print(f"All images saved to {output_dir}/")
        print(f"Sample list saved to {output_dir}/sample_list.json")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()