import torch
from diffusers import DiffusionPipeline
from optimum.quanto import quantize, qfloat8, freeze
from tqdm.auto import tqdm
from accelerate import PartialState
import os
import time

def main():
    model_name = "Qwen/Qwen-Image"
    lora_weights = "./test_lora_saves/checkpoint-10"
    output_dir = "outputs"
    negative_prompt = " "
    width = 384
    height = 384
    num_inference_steps = 50
    true_cfg_scale = 5.0
    base_seed = 655
    batch_size = 64  # 改大试试
    num_images = 12

    # 全部用同一个 prompt 方便测试质量
    prompts = ["man in the city"] * num_images

    os.makedirs(output_dir, exist_ok=True)

    distributed_state = PartialState()
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if distributed_state.is_main_process:
        print(f"Using {distributed_state.num_processes} GPUs")
        print(f"Total images: {len(prompts)}")
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

    # 计时开始
    distributed_state.wait_for_everyone()
    start_time = time.time()

    with distributed_state.split_between_processes(task_indices) as local_indices:
        local_indices = list(local_indices)
        local_start = time.time()

        for i in range(0, len(local_indices), batch_size):
            batch_start = time.time()
            batch_indices = local_indices[i:i+batch_size]
            batch_prompts = [prompts[idx] for idx in batch_indices]
            batch_neg_prompts = [negative_prompt] * len(batch_indices)

            seed = base_seed + batch_indices[0]
            generator = torch.Generator(device="cpu").manual_seed(seed)

            print(f"[GPU {distributed_state.process_index}] Generating batch: {batch_indices}")

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
                output_path = os.path.join(output_dir, f"image_{idx:03d}.png")
                images[j].save(output_path)

            print(f"[GPU {distributed_state.process_index}] Batch {batch_indices} done in {batch_time:.2f}s ({batch_time/len(batch_indices):.2f}s per image)")

        local_time = time.time() - local_start
        print(f"[GPU {distributed_state.process_index}] Total local time: {local_time:.2f}s")

    distributed_state.wait_for_everyone()
    total_time = time.time() - start_time

    if distributed_state.is_main_process:
        print(f"\n{'='*50}")
        print(f"Total images: {len(prompts)}")
        print(f"Batch size: {batch_size}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per image: {total_time/len(prompts):.2f}s")
        print(f"All images saved to {output_dir}/")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()