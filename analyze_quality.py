import os
import json
import argparse
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import shutil
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze image generation quality and find worst cases')
    
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory containing generated images (e.g., outputs/model_V1-2-3_256x256/)')
    parser.add_argument('--base_dir', type=str, default='VIOLIN_v2/data',
                        help='Base directory containing ground truth images')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of worst cases to identify per variation (default: 5)')
    parser.add_argument('--metrics', type=str, default='all',
                        help='Metrics to use: all, psnr, ssim, lpips, or combination like "psnr,ssim" (default: all)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory for analysis results (auto-generated if not specified)')
    parser.add_argument('--copy_images', action='store_true',
                        help='Copy worst case images to output directory')
    
    return parser.parse_args()


def load_image_as_array(image_path):
    """Load image and convert to numpy array (RGB, 0-255)"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, channel_axis=2, data_range=255)


def calculate_lpips(img1, img2, lpips_model):
    """Calculate LPIPS between two images"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img1_tensor = transform(Image.fromarray(img1)).unsqueeze(0)
    img2_tensor = transform(Image.fromarray(img2)).unsqueeze(0)
    
    if torch.cuda.is_available():
        img1_tensor = img1_tensor.cuda()
        img2_tensor = img2_tensor.cuda()
    
    with torch.no_grad():
        distance = lpips_model(img1_tensor, img2_tensor)
    
    return distance.item()


def calculate_color_variance(img):
    """
    Calculate color variance for pure color images
    Lower variance = more uniform = better for Variation 1
    """
    return np.std(img)


def detect_edges(img):
    """
    Detect edges in image (simple Sobel-like approach)
    For pure color images, fewer edges = better
    """
    gray = np.mean(img, axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_strength = np.sum(gx) + np.sum(gy)
    return edge_strength / (img.shape[0] * img.shape[1])


def analyze_variation_specific(img_gen, img_gt, variation):
    """
    Calculate variation-specific quality metrics
    
    Returns dict with custom metrics for each variation type
    """
    metrics = {}
    
    if variation == 1:  # Pure color
        # For pure color, we want low variance and low edge strength
        metrics['color_variance'] = calculate_color_variance(img_gen)
        metrics['edge_strength'] = detect_edges(img_gen)
        metrics['gt_variance'] = calculate_color_variance(img_gt)
        
    elif variation == 2:  # Gradient
        # For gradients, we want smooth transitions
        metrics['color_variance'] = calculate_color_variance(img_gen)
        metrics['edge_strength'] = detect_edges(img_gen)
        
    elif variation == 3:  # Geometric shapes
        # For shapes, edge detection is important
        metrics['edge_strength'] = detect_edges(img_gen)
        
    return metrics


def normalize_ground_truth_path(gt_path, base_dir):
    """
    Normalize ground truth path to avoid duplication
    
    Args:
        gt_path: Original ground truth path from sample_list.json
        base_dir: Base directory from args
    
    Returns:
        Normalized absolute path
    """
    # Remove duplicate base_dir if exists
    if gt_path.count(base_dir) > 1:
        # e.g., "VIOLIN_v2/data/VIOLIN_v2/data/Variation_1/..." 
        # -> "VIOLIN_v2/data/Variation_1/..."
        gt_path = gt_path.replace(f"{base_dir}/{base_dir}/", f"{base_dir}/")
    
    # If path already contains base_dir, use it directly
    if gt_path.startswith(base_dir):
        return gt_path
    
    # Otherwise, join with base_dir
    return os.path.join(base_dir, gt_path)


def load_sample_list(result_dir, base_dir):
    """Load sample_list.json from result directory"""
    sample_list_path = os.path.join(result_dir, 'sample_list.json')
    
    if not os.path.exists(sample_list_path):
        raise FileNotFoundError(f"sample_list.json not found in {result_dir}")
    
    with open(sample_list_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Normalize ground truth paths
    for sample in data['samples']:
        if 'ground_truth' in sample:
            sample['ground_truth'] = normalize_ground_truth_path(sample['ground_truth'], base_dir)
    
    return data['samples']


def get_generated_image_path(result_dir, task, width, height):
    """Get path to generated image"""
    var = task['variation']
    task_id = task['id']
    filename = f"V{var}_{task_id}_{width}x{height}_gen.png"
    return os.path.join(result_dir, filename)


def parse_resolution_from_dirname(dirname):
    """Extract resolution from directory name (e.g., model_V1-2-3_256x256)"""
    parts = dirname.split('_')
    for part in parts:
        if 'x' in part and part.replace('x', '').replace('-', '').isdigit():
            w, h = part.split('x')
            return int(w), int(h)
    return 256, 256  # Default


def main():
    args = parse_args()
    
    # Parse metrics
    if args.metrics == 'all':
        use_metrics = ['psnr', 'ssim', 'lpips']
    else:
        use_metrics = [m.strip().lower() for m in args.metrics.split(',')]
    
    print(f"Using metrics: {', '.join(use_metrics)}")
    
    # Load LPIPS model if needed
    lpips_model = None
    if 'lpips' in use_metrics:
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            lpips_model = lpips_model.cuda()
        lpips_model.eval()
    
    # Load sample list
    print(f"Loading sample list from {args.result_dir}...")
    tasks = load_sample_list(args.result_dir, args.base_dir)
    print(f"Found {len(tasks)} tasks")
    
    # Extract resolution from directory name
    dirname = os.path.basename(args.result_dir.rstrip('/'))
    width, height = parse_resolution_from_dirname(dirname)
    print(f"Detected resolution: {width}x{height}")
    
    # Group tasks by variation
    tasks_by_variation = defaultdict(list)
    for task in tasks:
        tasks_by_variation[task['variation']].append(task)
    
    print(f"Variations found: {sorted(tasks_by_variation.keys())}")
    
    # Analyze each variation
    all_results = {}
    missing_files_summary = defaultdict(list)
    
    for variation in sorted(tasks_by_variation.keys()):
        print(f"\n{'='*60}")
        print(f"Analyzing Variation {variation} ({len(tasks_by_variation[variation])} images)...")
        print(f"{'='*60}")
        
        variation_results = []
        missing_count = 0
        
        for task in tqdm(tasks_by_variation[variation], desc=f"V{variation}"):
            gen_path = get_generated_image_path(args.result_dir, task, width, height)
            gt_path = task['ground_truth']
            
            # Check if files exist
            if not os.path.exists(gen_path):
                missing_count += 1
                missing_files_summary[variation].append({
                    'type': 'generated',
                    'path': gen_path,
                    'task_id': task['id']
                })
                continue
            
            if not os.path.exists(gt_path):
                missing_count += 1
                missing_files_summary[variation].append({
                    'type': 'ground_truth',
                    'path': gt_path,
                    'task_id': task['id']
                })
                continue
            
            # Load images
            try:
                img_gen = load_image_as_array(gen_path)
                img_gt = load_image_as_array(gt_path)
            except Exception as e:
                print(f"Error loading images for {task['id']}: {e}")
                continue
            
            # Resize if needed
            if img_gen.shape != img_gt.shape:
                img_gt = np.array(Image.fromarray(img_gt).resize((img_gen.shape[1], img_gen.shape[0])))
            
            # Calculate metrics
            result = {
                'task_id': task['id'],
                'variation': variation,
                'prompt': task['prompt'],
                'gen_path': gen_path,
                'gt_path': gt_path,
                'metrics': {}
            }
            
            if 'psnr' in use_metrics:
                result['metrics']['psnr'] = calculate_psnr(img_gen, img_gt)
            
            if 'ssim' in use_metrics:
                result['metrics']['ssim'] = calculate_ssim(img_gen, img_gt)
            
            if 'lpips' in use_metrics:
                result['metrics']['lpips'] = calculate_lpips(img_gen, img_gt, lpips_model)
            
            # Add variation-specific metrics
            var_metrics = analyze_variation_specific(img_gen, img_gt, variation)
            result['metrics'].update(var_metrics)
            
            variation_results.append(result)
        
        # Report missing files
        if missing_count > 0:
            print(f"⚠️  Warning: {missing_count} files missing for Variation {variation}")
        
        # Check if we have valid results
        if len(variation_results) == 0:
            print(f"❌ No valid results found for Variation {variation}")
            all_results[f'variation_{variation}'] = {
                'total_images': len(tasks_by_variation[variation]),
                'valid_images': 0,
                'missing_files': missing_count,
                'worst_cases': {},
                'statistics': {},
                'error': 'No valid images found (all files missing or corrupted)'
            }
            continue
        
        # Sort by different metrics to find worst cases
        worst_cases = {}
        
        if 'psnr' in use_metrics:
            # Lower PSNR = worse
            sorted_by_psnr = sorted(variation_results, key=lambda x: x['metrics']['psnr'])
            worst_cases['psnr'] = sorted_by_psnr[:args.top_k]
        
        if 'ssim' in use_metrics:
            # Lower SSIM = worse
            sorted_by_ssim = sorted(variation_results, key=lambda x: x['metrics']['ssim'])
            worst_cases['ssim'] = sorted_by_ssim[:args.top_k]
        
        if 'lpips' in use_metrics:
            # Higher LPIPS = worse
            sorted_by_lpips = sorted(variation_results, key=lambda x: x['metrics']['lpips'], reverse=True)
            worst_cases['lpips'] = sorted_by_lpips[:args.top_k]
        
        # Variation-specific worst cases
        if variation == 1:  # Pure color
            # Higher variance = worse for pure color
            if variation_results and 'color_variance' in variation_results[0]['metrics']:
                sorted_by_variance = sorted(variation_results, key=lambda x: x['metrics']['color_variance'], reverse=True)
                worst_cases['color_variance'] = sorted_by_variance[:args.top_k]
            
            # Higher edge strength = worse for pure color
            if variation_results and 'edge_strength' in variation_results[0]['metrics']:
                sorted_by_edges = sorted(variation_results, key=lambda x: x['metrics']['edge_strength'], reverse=True)
                worst_cases['edge_strength'] = sorted_by_edges[:args.top_k]
        
        all_results[f'variation_{variation}'] = {
            'total_images': len(tasks_by_variation[variation]),
            'valid_images': len(variation_results),
            'missing_files': missing_count,
            'worst_cases': worst_cases,
            'statistics': calculate_statistics(variation_results, use_metrics)
        }
    
    # Generate output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        result_dirname = os.path.basename(args.result_dir.rstrip('/'))
        output_dir = os.path.join('analysis_results', f"{result_dirname}_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save missing files report
    if missing_files_summary:
        save_missing_files_report(output_dir, missing_files_summary)
    
    # Save analysis results
    save_analysis_results(output_dir, all_results, args, dirname)
    
    # Copy worst case images if requested
    if args.copy_images:
        copy_worst_case_images(output_dir, all_results, width, height)
    
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    if missing_files_summary:
        print(f"⚠️  Missing files report saved to: {output_dir}/missing_files.txt")
    print(f"{'='*60}")


def calculate_statistics(results, use_metrics):
    """Calculate statistics for metrics"""
    if not results:
        return {}
    
    stats = {}
    
    for metric in use_metrics:
        values = [r['metrics'][metric] for r in results if metric in r['metrics']]
        if values:
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
    
    # Add variation-specific metrics
    if results and 'color_variance' in results[0]['metrics']:
        values = [r['metrics']['color_variance'] for r in results]
        stats['color_variance'] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    if results and 'edge_strength' in results[0]['metrics']:
        values = [r['metrics']['edge_strength'] for r in results]
        stats['edge_strength'] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    return stats


def save_missing_files_report(output_dir, missing_files_summary):
    """Save report of missing files"""
    report_path = os.path.join(output_dir, 'missing_files.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Missing Files Report\n")
        f.write(f"{'='*60}\n\n")
        
        for variation in sorted(missing_files_summary.keys()):
            f.write(f"Variation {variation}: {len(missing_files_summary[variation])} missing files\n")
            f.write(f"{'-'*60}\n")
            
            for item in missing_files_summary[variation]:
                f.write(f"  Task ID: {item['task_id']}\n")
                f.write(f"  Type: {item['type']}\n")
                f.write(f"  Path: {item['path']}\n\n")


def save_analysis_results(output_dir, all_results, args, result_dirname):
    """Save analysis results to JSON and text files"""
    
    # Save detailed JSON
    json_path = os.path.join(output_dir, 'analysis_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Detailed results saved to: {json_path}")
    
    # Save human-readable summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Quality Analysis Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Result Directory: {result_dirname}\n")
        f.write(f"Metrics Used: {args.metrics}\n")
        f.write(f"Top K: {args.top_k}\n")
        f.write(f"\n")
        
        for var_key in sorted(all_results.keys()):
            var_data = all_results[var_key]
            var_num = var_key.split('_')[1]
            
            f.write(f"\n{'='*60}\n")
            f.write(f"Variation {var_num}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total images: {var_data['total_images']}\n")
            f.write(f"Valid images: {var_data.get('valid_images', 0)}\n")
            if var_data.get('missing_files', 0) > 0:
                f.write(f"⚠️  Missing files: {var_data['missing_files']}\n")
            f.write(f"\n")
            
            # Check for errors
            if 'error' in var_data:
                f.write(f"❌ Error: {var_data['error']}\n")
                continue
            
            # Statistics
            if var_data.get('statistics'):
                f.write(f"Statistics:\n")
                f.write(f"{'-'*60}\n")
                for metric, stats in var_data['statistics'].items():
                    f.write(f"{metric.upper()}:\n")
                    for stat_name, value in stats.items():
                        f.write(f"  {stat_name}: {value:.4f}\n")
                    f.write(f"\n")
            
            # Worst cases
            if var_data.get('worst_cases'):
                f.write(f"\nWorst Cases:\n")
                f.write(f"{'-'*60}\n")
                for metric_name, cases in var_data['worst_cases'].items():
                    f.write(f"\n{metric_name.upper()} (top {len(cases)}):\n")
                    for i, case in enumerate(cases, 1):
                        f.write(f"  {i}. Task ID: {case['task_id']}\n")
                        f.write(f"     Prompt: {case['prompt'][:80]}...\n" if len(case['prompt']) > 80 else f"     Prompt: {case['prompt']}\n")
                        f.write(f"     Metrics:\n")
                        for m, v in case['metrics'].items():
                            f.write(f"       {m}: {v:.4f}\n")
                        f.write(f"\n")
    
    print(f"📄 Summary saved to: {summary_path}")


def copy_worst_case_images(output_dir, all_results, width, height):
    """Copy worst case images to output directory for visual inspection"""
    
    print(f"\n📸 Copying worst case images...")
    
    for var_key, var_data in all_results.items():
        # Skip if no valid results
        if 'error' in var_data or not var_data.get('worst_cases'):
            continue
        
        var_num = var_key.split('_')[1]
        
        for metric_name, cases in var_data['worst_cases'].items():
            metric_dir = os.path.join(output_dir, f'V{var_num}_{metric_name}')
            os.makedirs(metric_dir, exist_ok=True)
            
            for i, case in enumerate(cases, 1):
                # Copy generated image
                gen_src = case['gen_path']
                gen_dst = os.path.join(metric_dir, f"rank{i}_gen_{case['task_id']}.png")
                shutil.copy2(gen_src, gen_dst)
                
                # Copy ground truth
                gt_src = case['gt_path']
                gt_dst = os.path.join(metric_dir, f"rank{i}_gt_{case['task_id']}.png")
                shutil.copy2(gt_src, gt_dst)
                
                # Create side-by-side comparison
                create_comparison_image(gen_src, gt_src, 
                                       os.path.join(metric_dir, f"rank{i}_compare_{case['task_id']}.png"),
                                       case)
    
    print(f"✅ Images copied to {output_dir}")


def create_comparison_image(gen_path, gt_path, output_path, case):
    """Create side-by-side comparison image with metrics"""
    img_gen = Image.open(gen_path).convert('RGB')
    img_gt = Image.open(gt_path).convert('RGB')
    
    # Resize to same size
    if img_gen.size != img_gt.size:
        img_gt = img_gt.resize(img_gen.size)
    
    width, height = img_gen.size
    
    # Create canvas (side by side + text area)
    text_height = 120
    canvas = Image.new('RGB', (width * 2, height + text_height), 'white')
    
    # Paste images
    canvas.paste(img_gen, (0, 0))
    canvas.paste(img_gt, (width, 0))
    
    # Add text (using PIL drawing)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Add labels
    draw.text((10, height + 5), "Generated", fill='black', font=font)
    draw.text((width + 10, height + 5), "Ground Truth", fill='black', font=font)
    
    # Add metrics
    y_offset = height + 25
    draw.text((10, y_offset), f"Task ID: {case['task_id']}", fill='black', font=font)
    y_offset += 15
    
    for metric, value in case['metrics'].items():
        draw.text((10, y_offset), f"{metric}: {value:.4f}", fill='black', font=font)
        y_offset += 15
        if y_offset > height + text_height - 10:
            break
    
    canvas.save(output_path)


if __name__ == "__main__":
    main()