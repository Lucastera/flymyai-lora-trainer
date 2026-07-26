#!/usr/bin/env python3
"""
VIOLIN Multi-Seed Shape Evaluation Script
汇总 4 模型 x 3 seed 的 Shape 指标 (Variation 3)，输出为单份 txt + csv
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
from shape_metric import Shape_metrics_from_img_path


# ============ 配置区：按需修改 ============
# (huggingface 模型名, 表格中显示名称)，顺序即表格行顺序
MODELS = [
    ("black-forest-labs/FLUX.1-dev", "Flux.1"),
    ("black-forest-labs/FLUX.2-klein-4B", "Flux.2"),
    ("Qwen/Qwen-Image", "Qwen-Image"),
    ("Tongyi-MAI/Z-Image", "Z-Image"),
]
SEEDS = [42, 0, 123]
VARIATIONS = [3]          # 只评估 Variation 3 (Shape)
WIDTH = 512
HEIGHT = 512
METRIC_COLS = ['iou', 'size', 'shape', 'dist', 'purity']


def parse_args():
    parser = argparse.ArgumentParser(description='汇总多模型多 seed 的 VIOLIN Shape 评估结果')
    parser.add_argument('--base_dir', type=str, default='/data4/kuan/outputs',
                        help='存放各模型生成结果目录的根路径')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='汇总报告输出目录 (默认 results)')
    parser.add_argument('--output_name', type=str, default='violin_shape_summary',
                        help='输出文件名前缀 (不含扩展名)')
    return parser.parse_args()


def build_gen_dir(base_dir, model_name, seed):
    """按 inference_multi_volin_v2.py 的命名规则拼出生成结果目录"""
    model_short = model_name.split('/')[-1]
    v_str = "V" + "-".join(map(str, VARIATIONS))
    resolution_str = f"{WIDTH}x{HEIGHT}"
    dir_name = f"{model_short}_{v_str}_{resolution_str}_seed{seed}"
    return Path(base_dir) / dir_name


def evaluate_one_dir(gen_dir):
    """对单个生成目录计算 Shape 指标均值，失败返回 None"""
    sample_list_path = gen_dir / 'sample_list.json'
    if not sample_list_path.exists():
        print(f"⚠️  未找到 sample_list.json: {sample_list_path}")
        return None

    with open(sample_list_path, 'r', encoding='utf-8') as f:
        sample_info = json.load(f)

    tasks = sample_info['samples']
    shape_results = []

    for task in tqdm(tasks, desc=f"Evaluating {gen_dir.name}", leave=False):
        if task['variation'] != 3:
            continue

        task_id = task['id']
        gen_files = list(gen_dir.glob(f"V3_{task_id}_*_gen.png"))
        if not gen_files:
            print(f"⚠️  未找到生成图: {task_id} ({gen_dir.name})")
            continue

        gt_path = task['ground_truth']
        if not os.path.exists(gt_path):
            print(f"⚠️  未找到真值图: {gt_path}")
            continue

        try:
            metrics = Shape_metrics_from_img_path(str(gen_files[0]), gt_path)
            shape_results.append(metrics)
        except Exception as e:
            print(f"⚠️  计算指标出错 {task_id}: {e}")
            continue

    if not shape_results:
        print(f"❌ {gen_dir.name} 无有效结果")
        return None

    df = pd.DataFrame(shape_results)
    summary = {
        'iou': df['d_iou'].mean(),
        'size': df['d_size'].mean(),
        'shape': df['d_shape'].mean(),
        'dist': df['d_dist'].mean(),
        'purity': df['d_purity'].mean(),
    }
    summary['mean'] = sum(summary[c] for c in METRIC_COLS) / len(METRIC_COLS)
    return summary


def build_table_df(results_for_this_table, order_names):
    """按指定模型顺序构建一张表 (DataFrame)，缺失的模型留空"""
    rows = []
    for disp_name in order_names:
        row = {'Open Source': disp_name}
        metrics = results_for_this_table.get(disp_name)
        if metrics is None:
            for c in METRIC_COLS + ['mean']:
                row[c] = None
        else:
            row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows, columns=['Open Source'] + METRIC_COLS + ['mean'])


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    order_names = [disp for _, disp in MODELS]

    # all_results[seed][disp_name] = {iou, size, shape, dist, purity, mean}
    all_results = {seed: {} for seed in SEEDS}

    for model_name, disp_name in MODELS:
        for seed in SEEDS:
            gen_dir = build_gen_dir(base_dir, model_name, seed)
            print(f"\n📂 处理: {disp_name}  seed={seed}  -> {gen_dir}")
            summary = evaluate_one_dir(gen_dir)
            if summary is not None:
                all_results[seed][disp_name] = summary

    # ---- 计算 Seed Avg：对每个模型，3 个 seed 的均值再取一次平均 ----
    avg_results = {}
    for disp_name in order_names:
        seed_summaries = [all_results[seed][disp_name]
                           for seed in SEEDS if disp_name in all_results[seed]]
        if not seed_summaries:
            continue
        avg_df = pd.DataFrame(seed_summaries)
        avg_results[disp_name] = {c: avg_df[c].mean() for c in METRIC_COLS + ['mean']}

    # ---- 组装 4 张表 ----
    seed_tables = {seed: build_table_df(all_results[seed], order_names) for seed in SEEDS}
    avg_table = build_table_df(avg_results, order_names)

    # ---- 写 TXT (Tab 分隔，贴近原始表格样式) ----
    txt_path = results_dir / f"{args.output_name}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        for seed in SEEDS:
            f.write(f"Seed = {seed}\n")
            f.write(seed_tables[seed].to_csv(sep='\t', index=False,
                                              float_format='%.4f', na_rep=''))
            f.write("\n")
        f.write("Seed Avg\n")
        f.write(avg_table.to_csv(sep='\t', index=False,
                                  float_format='%.4f', na_rep=''))

    # ---- 写 CSV (逗号分隔) ----
    csv_path = results_dir / f"{args.output_name}.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        for seed in SEEDS:
            f.write(f"Seed = {seed}\n")
            f.write(seed_tables[seed].to_csv(index=False,
                                              float_format='%.4f', na_rep=''))
            f.write("\n")
        f.write("Seed Avg\n")
        f.write(avg_table.to_csv(index=False, float_format='%.4f', na_rep=''))

    print(f"\n✅ 汇总结果已保存:")
    print(f"   TXT: {txt_path}")
    print(f"   CSV: {csv_path}")


if __name__ == "__main__":
    main()