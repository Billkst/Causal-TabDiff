#!/usr/bin/env python3
"""
重新生成所有 baseline 表格 - 修复版本
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

SEEDS = [42, 52, 62, 72, 82]

def find_optimal_threshold(y_true, y_pred_proba):
    """使用 Youden's J statistic 找到最优阈值"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]

def compute_metrics_with_optimal_threshold(y_true, y_pred_proba):
    """使用最优阈值计算指标"""
    if len(np.unique(y_true)) < 2:
        return None
    
    threshold = find_optimal_threshold(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return {
        'auroc': float(roc_auc_score(y_true, y_pred_proba)),
        'auprc': float(average_precision_score(y_true, y_pred_proba)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'optimal_threshold': float(threshold)
    }

def regenerate_itransformer_metrics():
    """重新生成 iTransformer 的 metrics（使用最优阈值）"""
    print("\n=== 重新评估 iTransformer（使用最优阈值）===")
    
    for seed in SEEDS:
        # 查找 prediction 文件
        pred_files = [
            f'outputs/tslib_models/itransformer_seed{seed}_predictions.npz',
            f'outputs/b2_baseline/layer1/itransformer_seed{seed}_predictions.npz'
        ]
        
        pred_file = None
        for f in pred_files:
            if os.path.exists(f):
                pred_file = f
                break
        
        if not pred_file:
            print(f"  ⚠️  Seed {seed}: 未找到 prediction 文件")
            continue
        
        data = np.load(pred_file)
        if 'test_y_true' in data:
            y_true = data['test_y_true']
            y_pred = data['test_y_pred']
        elif 'y_true' in data:
            y_true = data['y_true']
            y_pred = data['y_pred']
        else:
            print(f"  ⚠️  Seed {seed}: 无法找到 y_true/y_pred")
            continue
        
        metrics = compute_metrics_with_optimal_threshold(y_true, y_pred)
        if metrics:
            # 保存新的 metrics
            output_dir = 'outputs/b2_baseline/layer1'
            os.makedirs(output_dir, exist_ok=True)
            
            # 读取原始 metrics 并更新
            metrics_file = f'{output_dir}/iTransformer_seed{seed}_metrics.json'
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        original_metrics = json.load(f)
                    original_metrics.update(metrics)
                    metrics = original_metrics
                except:
                    pass
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"  ✓ Seed {seed}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}, Threshold={metrics['optimal_threshold']:.4f}")

def add_layer2_readout_metrics():
    """为 Layer2 表添加 2-year readout 分类指标"""
    print("\n=== 补充 Layer2 的 2-year readout 指标 ===")
    
    models = ['iTransformer', 'TimeXer', 'SSSD', 'SurvTraj']
    
    for model in models:
        for seed in SEEDS:
            # 查找 layer2 npz 文件
            npz_files = [
                f'outputs/tslib_layer2/{model.lower()}_seed{seed}_layer2.npz',
                f'outputs/b2_baseline/layer2/{model.lower()}_seed{seed}_layer2.npz'
            ]
            
            npz_file = None
            for f in npz_files:
                if os.path.exists(f):
                    npz_file = f
                    break
            
            if not npz_file:
                continue
            
            data = np.load(npz_file)
            y_pred = data['y_pred']
            y_true = data['y_true']
            
            # 提取 2-year readout（第 0 列）
            if len(y_pred.shape) == 2:
                y_pred_2year = y_pred[:, 0]
                y_true_2year = y_true[:, 0]
            else:
                continue
            
            # 计算分类指标
            readout_metrics = compute_metrics_with_optimal_threshold(y_true_2year, y_pred_2year)
            if readout_metrics:
                # 保存到 metrics 文件
                metrics_file = f'outputs/b2_baseline/layer2/{model}_seed{seed}_layer2_readout_metrics.json'
                with open(metrics_file, 'w') as f:
                    json.dump(readout_metrics, f, indent=2)
                
                print(f"  ✓ {model} seed {seed}: 2-year AUROC={readout_metrics['auroc']:.4f}")

if __name__ == '__main__':
    print("=" * 80)
    print("B2 Baseline 表格修复脚本")
    print("=" * 80)
    
    regenerate_itransformer_metrics()
    add_layer2_readout_metrics()
    
    print("\n✓ 修复完成")
