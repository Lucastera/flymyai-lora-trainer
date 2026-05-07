import asyncio
import aiohttp
import json
import os
from pathlib import Path
from tqdm.asyncio import tqdm
import argparse
from datetime import datetime
import random
from PIL import Image
import io
import sys
import concurrent.futures


class RateLimiter:
    """Rate limiter: max N calls per second"""
    def __init__(self, max_calls=2, time_window=1):
        self.max_calls = max_calls
        self.time_window = time_window
        self.semaphore = asyncio.Semaphore(max_calls)
        self.call_times = []
    
    async def acquire(self):
        await self.semaphore.acquire()
        
        current_time = asyncio.get_event_loop().time()
        
        # Remove expired records
        self.call_times = [t for t in self.call_times if current_time - t < self.time_window]
        
        # Wait if rate limit reached
        if len(self.call_times) >= self.max_calls:
            wait_time = self.time_window - (current_time - self.call_times[0]) + 0.01
            await asyncio.sleep(wait_time)
            self.call_times = []
        
        self.call_times.append(current_time)
    
    def release(self):
        self.semaphore.release()


def parse_args():
    parser = argparse.ArgumentParser(description='API-based Variation 4 image editing')
    
    # API configuration with defaults
    parser.add_argument('--api_url', type=str, 
                        default='https://api.bltcy.ai/v1/images/edits',
                        help='API endpoint URL (default: bltcy.ai)')
    parser.add_argument('--api_key', type=str, 
                        default='sk-o9WYOsdKKZC853Ng560e70Fd2b8249139f4b989fE771F9Dd',
                        help='API key')
    parser.add_argument('--model_name', type=str, 
                        default='flux-kontext-max',
                        help='Model name (default: flux-kontext-max)')
    
    # Dataset configuration
    parser.add_argument('--base_dir', type=str, 
                        default='VIOLIN_v2/data',
                        help='Base directory containing test.jsonl')
    parser.add_argument('--mask_types', type=str, default='inpainting',
                        help='"all", "inpainting", "outpainting", "random", or comma-separated (default: "all")')
    
    # Sampling configuration
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per mask type (None = all)')
    parser.add_argument('--sample_seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    
    # Concurrency configuration
    parser.add_argument('--max_concurrent', type=int, default=200,
                        help='Maximum concurrent requests (default: 200)')
    parser.add_argument('--rate_limit', type=int, default=10,
                        help='Maximum API calls per second (default: 10)')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Request timeout in seconds (default: 120)')
    
    # Image generation parameters
    parser.add_argument('--aspect_ratio', type=str, default='1:1',
                        help='Aspect ratio (default: 1:1)')
    parser.add_argument('--image_size', type=str, default='256',
                        help='Image size in pixels (default: 256)')
    parser.add_argument('--prompt', type=str, 
                        default='Apply the binary mask in Image 2 to the image in Image 1. For every pixel, if the mask value is white (value 255), keep the original color from Image 1; if the mask value is black (value 0), change it to pure black.',
                        help='Default prompt for image editing')
    
    # Output configuration
    parser.add_argument('--save_root', type=str, default='test_results',
                        help='Root directory for saving results (default: test_results)')
    
    # Debug options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def parse_mask_types(mask_types_str):
    """Parse mask_types string"""
    if mask_types_str.lower() == 'all':
        return ['inpainting', 'outpainting', 'random']
    
    valid_types = ['inpainting', 'outpainting', 'random']
    requested = [t.strip() for t in mask_types_str.split(',')]
    
    result = [t for t in requested if t in valid_types]
    if not result:
        raise ValueError(f"Invalid mask_types: {mask_types_str}")
    
    return result


def extract_id_from_path(filepath):
    """
    Extract ID from filepath
    Examples:
    - 'images/00000001.jpg' -> '00000001'
    - 'masks/00000123.png' -> '00000123'
    """
    filename = os.path.basename(filepath)
    # Remove extension
    id_str = os.path.splitext(filename)[0]
    return id_str


def generate_output_dir_name(model_name, mask_types, save_root, max_samples=None, image_size='1024'):
    """Generate output directory name based on model and resolution"""
    mask_str = "-".join(mask_types)
    sample_suffix = f"_sample{max_samples}" if max_samples is not None else ""
    
    dir_name = f"{model_name}_variation_4_{mask_str}_{image_size}{sample_suffix}"
    return os.path.join(save_root, dir_name)


def load_or_create_sample_list(output_dir, base_dir, mask_types, max_samples, sample_seed, verbose=False):
    """Load or create sample list"""
    list_file = os.path.join(output_dir, "sample_list.json")
    
    if os.path.exists(list_file):
        print(f"[INFO] Loading existing sample list from: {list_file}")
        with open(list_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        print(f"[INFO] ✓ Loaded {len(saved_data['samples'])} samples")
        return saved_data['samples']
    
    else:
        print("[INFO] Creating new sample list...")
        
        # Group tasks by mask_type
        tasks_by_type = {mt: [] for mt in mask_types}
        
        jsonl_path = os.path.join(base_dir, "test.jsonl")
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Cannot find test.jsonl at: {jsonl_path}")
        
        print(f"[INFO] Reading {jsonl_path}...")
        line_count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if verbose and line_count % 1000 == 0:
                    print(f"  Processed {line_count} lines...", end='\r')
                
                task = json.loads(line.strip())
                
                # Only process Variation 4
                if task['variation'] != 4:
                    continue
                
                mask_type = task.get('mask_type')
                if mask_type not in mask_types:
                    continue
                
                # Normalize paths
                task['image1_path'] = os.path.join(base_dir, task['image1_path'].replace('\\', '/'))
                task['image2_path'] = os.path.join(base_dir, task['image2_path'].replace('\\', '/'))
                task['ground_truth'] = os.path.join(base_dir, task['ground_truth'].replace('\\', '/'))
                
                tasks_by_type[mask_type].append(task)
        
        if verbose:
            print()  # New line after progress
        print(f"[INFO] ✓ Read {line_count} total lines from test.jsonl")
        
        # Sample if needed
        final_tasks = []
        for mt in mask_types:
            available = tasks_by_type[mt]
            print(f"[INFO]   {mt}: {len(available)} tasks available")
            
            if max_samples is not None and len(available) > max_samples:
                random.seed(sample_seed)
                sampled = random.sample(available, max_samples)
                print(f"[INFO]     -> Sampled {max_samples} tasks")
                final_tasks.extend(sampled)
            else:
                final_tasks.extend(available)
        
        print(f"[INFO] ✓ Total tasks to process: {len(final_tasks)}")
        
        # Save sample list
        sample_info = {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'base_dir': base_dir,
            'mask_types': mask_types,
            'max_samples_per_type': max_samples,
            'sample_seed': sample_seed,
            'samples': final_tasks
        }
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Saving sample list to: {list_file}")
        with open(list_file, 'w', encoding='utf-8') as f:
            json.dump(sample_info, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] ✓ Saved sample list")
        return final_tasks


def task_to_filename(task, image_size='1024'):
    """
    Convert task to filename with detailed information
    Format: {mask_type}_{task_id}_{img1_id}_{img2_id}_{gt_id}_{image_size}
    
    Example: inpainting_task001_00000123_00000456_00000789_256
    """
    mask_type = task['mask_type']
    task_id = task['id']
    
    # Extract IDs from paths
    img1_id = extract_id_from_path(task['image1_path'])
    img2_id = extract_id_from_path(task['image2_path'])
    gt_id = extract_id_from_path(task['ground_truth'])
    
    # Format: {mask_type}_{task_id}_{img1_id}_{img2_id}_{gt_id}_{size}
    filename = f"{mask_type}_{task_id}_img1-{img1_id}_img2-{img2_id}_gt-{gt_id}_{image_size}"
    
    return filename


def check_already_generated(output_dir, task, image_size='1024'):
    """Check if image already generated"""
    filename = task_to_filename(task, image_size)
    output_path = os.path.join(output_dir, f"{filename}_gen.png")
    return os.path.exists(output_path)


# ✅ 优化1: 使用 aiohttp 异步下载图片
async def download_image_from_url_async(session, url, filepath):
    """Async download image from URL to filepath"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                content = await response.read()
                # 使用 executor 写文件，避免阻塞
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: open(filepath, 'wb').write(content))
                return True
            else:
                raise Exception(f"Download failed with status code: {response.status}")
    except Exception as e:
        raise Exception(f"Download error: {str(e)}")


# ✅ 优化2: 使用 aiohttp 异步调用 API
async def call_api(session, task, args, rate_limiter, verbose=False):
    """Call API for image editing using aiohttp (asynchronous)"""
    await rate_limiter.acquire()
    
    try:
        size_value = f"{args.image_size}x{args.image_size}"
        
        # 使用 aiohttp FormData
        data = aiohttp.FormData()
        data.add_field('prompt', task.get('prompt', args.prompt))
        data.add_field('model', args.model_name)
        data.add_field('size', size_value)
        data.add_field('aspect_ratio', args.aspect_ratio)
        
        # 读取文件并添加到 FormData
        loop = asyncio.get_event_loop()
        image1_data = await loop.run_in_executor(None, lambda: open(task['image1_path'], 'rb').read())
        image2_data = await loop.run_in_executor(None, lambda: open(task['image2_path'], 'rb').read())
        
        data.add_field('image', image1_data,
                      filename=os.path.basename(task['image1_path']),
                      content_type='image/jpeg')
        data.add_field('mask', image2_data,
                      filename=os.path.basename(task['image2_path']),
                      content_type='image/png')
        
        headers = {'Authorization': f'Bearer {args.api_key}'}
        
        if verbose:
            print(f"\n[DEBUG] Calling API for task {task['id']}")
        
        # 异步 POST 请求
        async with session.post(
            args.api_url,
            headers=headers,
            data=data,
            timeout=aiohttp.ClientTimeout(total=args.timeout)
        ) as response:
            
            if verbose:
                print(f"[DEBUG] Response status: {response.status}")
            
            if response.status == 200:
                result = await response.json()
                
                image_list = result.get("data", [])
                if not image_list:
                    raise ValueError("No image data in response")
                
                url = image_list[0].get("url")
                if not url:
                    raise ValueError("No URL in response data")
                
                if verbose:
                    print(f"[DEBUG] Got image URL: {url[:50]}...")
                
                return url
            else:
                text = await response.text()
                raise Exception(f"API error {response.status}: {text}")
    
    except Exception as e:
        if verbose:
            print(f"[ERROR] API call failed: {str(e)}")
        raise
    
    finally:
        rate_limiter.release()


async def process_task(session, task, args, output_dir, rate_limiter, pbar, stats, verbose=False):
    """Process single task"""
    filename = task_to_filename(task, args.image_size)
    output_path = os.path.join(output_dir, f"{filename}_gen.png")
    
    # Skip if already generated
    if os.path.exists(output_path):
        stats['skipped'] += 1
        if verbose:
            print(f"\n[INFO] Skipped (already exists): {filename}")
        pbar.update(1)
        return
    
    try:
        if verbose:
            print(f"\n[INFO] Processing: {filename}")
        
        # Call API to get image URL
        image_url = await call_api(session, task, args, rate_limiter, verbose)
        
        if verbose:
            print(f"[INFO] Downloading image...")
        
        # Download image asynchronously
        await download_image_from_url_async(session, image_url, output_path)
        
        stats['success'] += 1
        stats['by_type'][task['mask_type']] += 1
        
        if verbose:
            print(f"[INFO] ✓ Success: {filename}")
        
        pbar.set_postfix({
            'success': stats['success'],
            'failed': stats['failed'],
            'skipped': stats['skipped']
        })
    
    except Exception as e:
        stats['failed'] += 1
        error_msg = f"Failed: {filename} - {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        
        # Save error log
        error_log_path = os.path.join(output_dir, "error_log.txt")
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
    
    finally:
        pbar.update(1)


async def main_async(args, tasks, output_dir, verbose=False):
    """Async main function"""
    print(f"\n[INFO] Starting async processing...")
    print(f"[INFO] Rate limit: {args.rate_limit} calls/second")
    print(f"[INFO] Max concurrent: {args.max_concurrent}")
    print(f"[INFO] Timeout: {args.timeout}s")
    
    # Rate limiter
    rate_limiter = RateLimiter(max_calls=args.rate_limit, time_window=1)
    
    # Statistics
    stats = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'by_type': {mt: 0 for mt in parse_mask_types(args.mask_types)}
    }
    
    # Create aiohttp session
    connector = aiohttp.TCPConnector(limit=args.max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create progress bar
        print(f"\n[INFO] Processing {len(tasks)} tasks...")
        with tqdm(total=len(tasks), desc="Processing", ncols=100) as pbar:
            # Create task queue
            semaphore = asyncio.Semaphore(args.max_concurrent)
            
            async def bounded_task(task):
                async with semaphore:
                    await process_task(session, task, args, output_dir, rate_limiter, pbar, stats, verbose)
            
            # Execute all tasks
            await asyncio.gather(*[bounded_task(task) for task in tasks])
    
    return stats


def main():
    print("="*70)
    print("VIOLIN V2 - Variation 4 Image Editing with API")
    print("="*70)
    
    # Parse arguments
    print("\n[STEP 1/6] Parsing arguments...")
    args = parse_args()
    verbose = args.verbose
    
    if verbose:
        print("[DEBUG] Arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
    
    # Parse mask types
    print("\n[STEP 2/6] Parsing mask types...")
    mask_types = parse_mask_types(args.mask_types)
    print(f"[INFO] Processing mask types: {mask_types}")
    
    # Generate output directory with resolution
    print("\n[STEP 3/6] Setting up output directory...")
    output_dir = generate_output_dir_name(args.model_name, mask_types, args.save_root, args.max_samples, args.image_size)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or create sample list
    print("\n[STEP 4/6] Loading/creating sample list...")
    tasks = load_or_create_sample_list(output_dir, args.base_dir, mask_types, args.max_samples, args.sample_seed, verbose)
    
    # Show example filename format
    if tasks and verbose:
        example_filename = task_to_filename(tasks[0], args.image_size)
        print(f"\n[DEBUG] Example output filename: {example_filename}_gen.png")
    
    # Count already generated
    print("\n[STEP 5/6] Checking existing results...")
    print("[INFO] Scanning for already generated images...")
    already_generated = 0
    for i, task in enumerate(tasks):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Checked {i + 1}/{len(tasks)} tasks...", end='\r')
        if check_already_generated(output_dir, task, args.image_size):
            already_generated += 1
    
    if verbose:
        print()  # New line
    
    print(f"[INFO] ✓ Already generated: {already_generated}/{len(tasks)}")
    print(f"[INFO] ✓ Remaining to process: {len(tasks) - already_generated}")
    
    if already_generated == len(tasks):
        print("\n[INFO] All tasks already completed! Nothing to do.")
        return
    
    # Run async processing
    print("\n[STEP 6/6] Running async processing...")
    start_time = datetime.now()
    print(f"[INFO] Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        stats = asyncio.run(main_async(args, tasks, output_dir, verbose))
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Fatal error: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*70)
    print(f"Total tasks:        {len(tasks)}")
    print(f"Success:            {stats['success']} ✓")
    print(f"Failed:             {stats['failed']} ✗")
    print(f"Skipped (existed):  {stats['skipped']} ⊝")
    print(f"\nBreakdown by mask type:")
    for mt in mask_types:
        print(f"  {mt:12s}: {stats['by_type'][mt]} generated")
    print(f"\nTime elapsed:       {elapsed}")
    print(f"Average per task:   {elapsed.total_seconds() / len(tasks):.2f}s")
    print(f"\nOutput directory:   {output_dir}")
    
    if stats['failed'] > 0:
        error_log = os.path.join(output_dir, "error_log.txt")
        print(f"Error log:          {error_log}")
    
    print("="*70)


if __name__ == "__main__":
    main()