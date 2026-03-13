#!/usr/bin/env python3
"""
生成Baseline对比实验汇总报告
"""
import json
import numpy as np
from pathlib import Path
import pandas as pd

def load_metrics(pattern, seeds):
    """加载指定seeds的指标"""
    metrics_list = []
    for seed in seeds:
        file = Path(pattern.format(seed=seed))
        if file.exists():
            with open(file) as f:
                metrics_list.append(json.load(f))
    return metrics_list

def compute_mean_std(metrics_list, keys):
    """计算mean±std"""
    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            result[f'{key}_mean'] = np.mean(values)
            result[f'{key}_std'] = np.std(values)
    return result

# Seeds配置
seeds_5 = [42, 52, 62, 72, 82]
seeds_9 = [42, 52, 62, 72, 82, 1024, 2024, 2025, 9999]

print("=" * 80)
print("Baseline对比实验汇总报告")
print("=" * 80)
print()

# Layer1 TSTR结果
print("### Layer1 TSTR (生成式模型)")
print()

# TSDiff
tsdiff_metrics = load_metrics('outputs/b2_baseline/tstr/tsdiff_seed{seed}_tstr_metrics.json', seeds_9)
tsdiff_summary = compute_mean_std(tsdiff_metrics, ['auroc', 'auprc', 'f1'])
print(f"TSDiff (9 seeds):")
print(f"  AUROC: {tsdiff_summary['auroc_mean']:.4f} ± {tsdiff_summary['auroc_std']:.4f}")
print(f"  AUPRC: {tsdiff_summary['auprc_mean']:.4f} ± {tsdiff_summary['auprc_std']:.4f}")
print(f"  F1:    {tsdiff_summary['f1_mean']:.4f} ± {tsdiff_summary['f1_std']:.4f}")
print()

# STaSy
stasy_metrics = load_metrics('outputs/b2_baseline/tstr/stasy_seed{seed}_tstr_metrics.json', seeds_9)
stasy_summary = compute_mean_std(stasy_metrics, ['auroc', 'auprc', 'f1'])
print(f"STaSy (9 seeds):")
print(f"  AUROC: {stasy_summary['auroc_mean']:.4f} ± {stasy_summary['auroc_std']:.4f}")
print(f"  AUPRC: {stasy_summary['auprc_mean']:.4f} ± {stasy_summary['auprc_std']:.4f}")
print(f"  F1:    {stasy_summary['f1_mean']:.4f} ± {stasy_summary['f1_std']:.4f}")
print()

# Layer2结果
print("### Layer2 (轨迹预测)")
print()

# iTransformer
itrans_metrics = load_metrics('outputs/b2_baseline/layer2/iTransformer_seed{seed}_layer2_metrics.json', seeds_5)
itrans_summary = compute_mean_std(itrans_metrics, ['trajectory_mse', 'trajectory_mae', 'valid_coverage'])
print(f"iTransformer (5 seeds):")
print(f"  Traj MSE: {itrans_summary['trajectory_mse_mean']:.2f} ± {itrans_summary['trajectory_mse_std']:.2f}")
print(f"  Traj MAE: {itrans_summary['trajectory_mae_mean']:.2f} ± {itrans_summary['trajectory_mae_std']:.2f}")
print(f"  Coverage: {itrans_summary['valid_coverage_mean']:.4f} ± {itrans_summary['valid_coverage_std']:.4f}")
print()

print("=" * 80)
print("报告生成完成")
print("=" * 80)
