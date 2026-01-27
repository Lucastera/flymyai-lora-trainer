import torch
from diffusers import DiffusionPipeline
from optimum.quanto import quantize, qfloat8, freeze
from tqdm.auto import tqdm
from accelerate import PartialState
import os
import time
from datetime import datetime


def scan_test_prompts(base_dir, prompt_levels, color_levels, split='test'):
    """
    Scan txt files according to specified levels
    
    Args:
      base_dir: Base directory "ColorBench-v1/Color_Split_Sets1"
      prompt_levels: [1, 2, 3] or [1, 3] etc.
      color_levels: [1, 2, 3] or [2] etc.
      split: 'train' / 'val' / 'test'
      
    Returns:
      [(txt absolute path, prompt content, relative path), ...]
    """
    prompt_data = []
    
    for p_level in prompt_levels:
        for c_level in color_levels:
            # Construct directory path
            dir_path = os.path.join(
                base_dir, 
                f"Prompt_Level_{p_level}", 
                f"Color_Level_{c_level}", 
                split
            )
            
            # Skip if directory does not exist
            if not os.path.exists(dir_path):
                print(f"Warning: Directory not found, skipping: {dir_path}")
                continue
            
            # Scan all .txt files in the directory
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.txt'):
                        txt_path = os.path.join(root, file)
                        
                        # Read prompt
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                prompt = f.read().strip()
                        except Exception as e:
                            print(f"Warning: Failed to read {txt_path}, skipping. Error: {e}")
                            continue
                        
                        # Calculate relative path (relative to base_dir)
                        relative_path = os.path.relpath(txt_path, base_dir)
                        
                        prompt_data.append((txt_path, prompt, relative_path))
    
    return prompt_data


def path_to_filename(relative_path):
    """
    Convert file relative path to safe output filename
    
    Args:
      relative_path: "Prompt_Level_1/Color_Level_1/test/img_001.txt"
      
    Returns:
      "Prompt_Level_1_Color_Level_1_test_img_001"
    """
    # Remove .txt extension
    name = os.path.splitext(relative_path)[0]
    # Replace path separators with underscores
    name = name.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
    return name


def check_already_generated(output_dir, relative_path):
    """
    Check if image has already been generated
    
    Args:
      output_dir: Output directory
      relative_path: Relative path "Prompt_Level_1/Color_Level_1/test/img_001.txt"
      
    Returns:
      True/False
    """
    safe_name = path_to_filename(relative_path)
    output_path = os.path.join(output_dir, f"{safe_name}_gen.png")
    return os.path.exists(output_path)


def save_config(output_dir, config, total_prompts):
    """Save configuration to file"""
    config_path = os.path.join(output_dir, "generation_config.txt")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"Generation Config - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        f.write(f"total_data: {total_prompts}\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def main():
    # ===== Configuration Parameters =====
    model_name = "Qwen/Qwen-Image"
    lora_weights = ""
    output_dir = "outputs"
    
    # Dataset configuration
    base_dir = "ColorBench-v1/Test_Sets"
    prompt_levels = [1,2,3]
    color_levels = [1,2,3]
    split = 'test'
    
    # Generation parameters
    negative_prompt = " "
    width = 384
    height = 384
    num_inference_steps = 50
    true_cfg_scale = 5.0
    base_seed = 42
    batch_size = 64
    
    # Save configuration
    config = {
        'model_name': model_name,
        'lora_weights': lora_weights if lora_weights else 'None',
        'output_dir': output_dir,
        'base_dir': base_dir,
        'prompt_levels': prompt_levels,
        'color_levels': color_levels,
        'split': split,
        'negative_prompt': negative_prompt,
        'width': width,
        'height': height,
        'num_inference_steps': num_inference_steps,
        'true_cfg_scale': true_cfg_scale,
        'base_seed': base_seed,
        'batch_size': batch_size,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    distributed_state = PartialState()
    
    # Main process saves configuration
    if distributed_state.is_main_process:
        print("Scanning prompts...")
    
    # Scan all prompts
    prompt_data = scan_test_prompts(base_dir, prompt_levels, color_levels, split)
    
    # Filter out already generated
    pending_data = []
    for item in prompt_data:
        if not check_already_generated(output_dir, item[2]):
            pending_data.append(item)
    
    if distributed_state.is_main_process:
        print(f"Total prompts found: {len(prompt_data)}")
        print(f"Already generated: {len(prompt_data) - len(pending_data)}")
        print(f"Pending generation: {len(pending_data)}")
        # Save configuration with total data count
        save_config(output_dir, config, len(prompt_data))
    
    # Exit directly if all images are already generated
    if len(pending_data) == 0:
        if distributed_state.is_main_process:
            print("All images already generated. Exiting.")
        return
    
    # Build list of pending items
    prompts = [item[1] for item in pending_data]
    relative_paths = [item[2] for item in pending_data]

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if distributed_state.is_main_process:
        print(f"Using {distributed_state.num_processes} GPUs")
        print(f"Batch size: {batch_size}")

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

    if lora_weights:
        pipe.load_lora_weights(lora_weights, adapter_name="lora")

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

    task_indices = list(range(len(prompts)))

    # Start timing
    distributed_state.wait_for_everyone()
    start_time = time.time()

    with distributed_state.split_between_processes(task_indices) as local_indices:
        local_indices = list(local_indices)
        local_start = time.time()

        # Add progress bar
        pbar = tqdm(
            range(0, len(local_indices), batch_size),
            desc=f"GPU {distributed_state.process_index}",
            disable=not distributed_state.is_local_main_process
        )

        for i in pbar:
            batch_start = time.time()
            batch_indices = local_indices[i:i+batch_size]
            batch_prompts = [prompts[idx] for idx in batch_indices]
            batch_neg_prompts = [negative_prompt] * len(batch_indices)

            seed = base_seed + batch_indices[0]
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

            for j, idx in enumerate(batch_indices):
                safe_name = path_to_filename(relative_paths[idx])
                output_path = os.path.join(output_dir, f"{safe_name}_gen.png")
                images[j].save(output_path)

            pbar.set_postfix({'batch_time': f'{batch_time:.2f}s', 'per_img': f'{batch_time/len(batch_indices):.2f}s'})

        local_time = time.time() - local_start
        print(f"[GPU {distributed_state.process_index}] Total local time: {local_time:.2f}s")

    distributed_state.wait_for_everyone()
    total_time = time.time() - start_time

    if distributed_state.is_main_process:
        print(f"\n{'='*50}")
        print(f"Total images generated: {len(prompts)}")
        print(f"Batch size: {batch_size}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per image: {total_time/len(prompts):.2f}s")
        print(f"All images saved to {output_dir}/")
        print(f"Config saved to {output_dir}/generation_config.txt")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()