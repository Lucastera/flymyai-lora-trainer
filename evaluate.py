#!/usr/bin/env python3
"""
VIOLIN Evaluation Script
Evaluate generated images against ground truth using color and shape metrics.
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys

# Add metrics directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metrics'))

from color_metric import Color_metrics_from_img_path
from shape_metric import Shape_metrics_from_img_path


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VIOLIN generation results')
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='Generated images directory (e.g., outputs/FLUX.2-klein-4B_V1-2-3_512x512)')
    parser.add_argument('--model_name', type=str, default='',
                        help='Model name for "Open Source" column (default: auto from gen_dir)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results output directory (default: results)')
    return parser.parse_args()


def extract_model_name(gen_dir):
    """Extract model name from directory path"""
    dir_name = Path(gen_dir).name
    parts = dir_name.split('_')
    model_parts = []
    for part in parts:
        if part.startswith('V') or 'x' in part:
            break
        model_parts.append(part)
    return '_'.join(model_parts) if model_parts else dir_name


def main():
    args = parse_args()
    
    gen_dir = Path(args.gen_dir)
    results_dir = Path(args.results_dir)
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model name
    model_name = args.model_name if args.model_name else extract_model_name(args.gen_dir)
    
    # Load sample list
    sample_list_path = gen_dir / 'sample_list.json'
    if not sample_list_path.exists():
        print(f"❌ Error: {sample_list_path} not found!")
        return
    
    with open(sample_list_path, 'r', encoding='utf-8') as f:
        sample_info = json.load(f)
    
    tasks = sample_info['samples']
    print(f"📊 Loaded {len(tasks)} tasks from sample_list.json")
    
    # Separate results by task type
    color_results = []  # Var 1 + 2
    shape_results = []  # Var 3
    
    for task in tqdm(tasks, desc="Evaluating"):
        var = task['variation']
        task_id = task['id']
        
        # Find generated image
        gen_files = list(gen_dir.glob(f"V{var}_{task_id}_*_gen.png"))
        if not gen_files:
            print(f"⚠️  Warning: Generated image not found for {task_id}")
            continue
        
        gen_path = str(gen_files[0])
        gt_path = task['ground_truth']
        
        if not os.path.exists(gt_path):
            print(f"⚠️  Warning: Ground truth not found: {gt_path}")
            continue
        
        # Compute metrics
        try:
            if var == 1:  # Pure color
                metrics = Color_metrics_from_img_path(gen_path, gt_path)
                color_results.append(metrics)
            elif var == 2:  # Gradient
                metrics = Color_metrics_from_img_path(gen_path, gt_path, is_multi_block=True)
                color_results.append(metrics)
            elif var == 3:  # Geometry
                metrics = Shape_metrics_from_img_path(gen_path, gt_path)
                shape_results.append(metrics)
        except Exception as e:
            print(f"⚠️  Error processing {task_id}: {e}")
            continue
    
    if not color_results and not shape_results:
        print("❌ No results to save!")
        return
    
    # Generate result filename prefix from gen_dir name
    result_prefix = gen_dir.name
    
    # ========== Color Metrics (Var 1 + 2) ==========
    if color_results:
        color_df = pd.DataFrame(color_results)
        
        # Calculate mean for each metric (using actual returned field names)
        color_summary = {
            'Open Source': model_name,
            'rgb-ed': color_df['d_rgb_ed'].mean(),
            'lab-00': color_df['d_lab_00'].mean(),
            'std': color_df['d_sd'].mean(),
            'hf-ratio': color_df['d_hf'].mean(),
            'ced': color_df['d_ced'].mean(),
        }
        # Calculate overall mean
        color_summary['mean'] = (
            color_summary['rgb-ed'] + 
            color_summary['lab-00'] + 
            color_summary['std'] + 
            color_summary['hf-ratio'] + 
            color_summary['ced']
        ) / 5
        
        # Create DataFrame with specific column order
        color_final_df = pd.DataFrame([color_summary], columns=[
            'Open Source', 'rgb-ed', 'lab-00', 'std', 'hf-ratio', 'ced', 'mean'
        ])
        
        # Save Color results
        color_csv_path = results_dir / f'{result_prefix}_color.csv'
        color_txt_path = results_dir / f'{result_prefix}_color.txt'
        
        color_final_df.to_csv(color_csv_path, index=False, float_format='%.4f')
        
        with open(color_txt_path, 'w', encoding='utf-8') as f:
            f.write("颜色评估结果 (Variation 1 + 2)\n")
            f.write("="*80 + "\n\n")
            f.write(color_final_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
            f.write("\n\n")
            f.write(f"Total samples: {len(color_results)}\n")
        
        print(f"\n✅ Color results saved:")
        print(f"   CSV: {color_csv_path}")
        print(f"   TXT: {color_txt_path}")
        
        # Print to console
        print("\n" + "="*80)
        print("📊 颜色评估结果 (Variation 1 + 2)")
        print("="*80)
        print(color_final_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print(f"\nTotal samples: {len(color_results)}")
    
    # ========== Shape Metrics (Var 3) ==========
    if shape_results:
        shape_df = pd.DataFrame(shape_results)
        
        # Calculate mean for each metric (using actual returned field names)
        shape_summary = {
            'Open Source': model_name,
            'iou': shape_df['d_iou'].mean(),
            'size': shape_df['d_size'].mean(),
            'shape': shape_df['d_shape'].mean(),
            'dist': shape_df['d_dist'].mean(),
            'purity': shape_df['d_purity'].mean(),
        }
        # Calculate overall mean
        shape_summary['mean'] = (
            shape_summary['iou'] + 
            shape_summary['size'] + 
            shape_summary['shape'] + 
            shape_summary['dist'] + 
            shape_summary['purity']
        ) / 5
        
        # Create DataFrame with specific column order
        shape_final_df = pd.DataFrame([shape_summary], columns=[
            'Open Source', 'iou', 'size', 'shape', 'dist', 'purity', 'mean'
        ])
        
        # Save Shape results
        shape_csv_path = results_dir / f'{result_prefix}_shape.csv'
        shape_txt_path = results_dir / f'{result_prefix}_shape.txt'
        
        shape_final_df.to_csv(shape_csv_path, index=False, float_format='%.4f')
        
        with open(shape_txt_path, 'w', encoding='utf-8') as f:
            f.write("形状评估结果 (Variation 3)\n")
            f.write("="*80 + "\n\n")
            f.write(shape_final_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
            f.write("\n\n")
            f.write(f"Total samples: {len(shape_results)}\n")
        
        print(f"\n✅ Shape results saved:")
        print(f"   CSV: {shape_csv_path}")
        print(f"   TXT: {shape_txt_path}")
        
        # Print to console
        print("\n" + "="*80)
        print("📊 形状评估结果 (Variation 3)")
        print("="*80)
        print(shape_final_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print(f"\nTotal samples: {len(shape_results)}")
    
    print("\n" + "="*80)
    print(f"📁 All results saved to: {results_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()