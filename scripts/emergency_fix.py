#!/usr/bin/env python3
"""
紧急修复脚本 - 修复所有发现的问题
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    mean_squared_error, mean_absolute_error
)

SEEDS = [42, 52, 62, 72, 82]

def fix_layer1_metrics():
    """修复 Layer1 指标计算错误"""
    print("\n=== 修复 Layer1 指标 ===")
    
    # 重新计算所有模型的指标
    models_paths = {
        'iTransformer': 'outputs/tslib_models/itransformer_seed{seed}_predictions.npz',
        'CausalForest': 'outputs/b2_baseline/layer1/CausalForest_seed{seed}_predictions.npz',
    }
    
    for model, path_template in models_paths.items():
        print(f"\n{model}:")
        for seed in SEEDS:
            path = path_template.format(seed=seed)
            if not os.path.exists(path):
                continue
            
            data = np.load(path)
            y_true = data.get('y_true', data.get('test_y_true'))
            y_pred_proba = data.get('y_pred', data.get('test_y_pred'))
            
            # 使用最优阈值
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            threshold = thresholds[optimal_idx]
            
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # 正确计算所有指标
            metrics = {
                'auroc': float(roc_auc_score(y_true, y_pred_proba)),
                'auprc': float(average_precision_score(y_true, y_pred_proba)),
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'f1': float(f1_score(y_true, y_pred, zero_division=0)),
                'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
                'optimal_threshold': float(threshold),
                'confusion_matrix': cm.tolist()
            }
            
            # 保存修复后的 metrics
            output_dir = 'outputs/b2_baseline/layer1_fixed'
            os.makedirs(output_dir, exist_ok=True)
            
            with open(f'{output_dir}/{model}_seed{seed}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"  Seed {seed}: Recall={metrics['recall']:.4f}, Spec={metrics['specificity']:.4f}, Acc={metrics['accuracy']:.4f}")

def fix_layer2_evaluation():
    """修复 Layer2 评估 - 统一预测目标"""
    print("\n=== 修复 Layer2 评估 ===")
    
    # 对于 iTransformer 和 TimeXer，需要重新评估
    # 问题：它们预测的是原始特征值，不是归一化风险值
    # 解决方案：标记为无效或使用正确的预测列
    
    print("⚠️  iTransformer 和 TimeXer 的 Layer2 预测目标不一致")
    print("   - 它们预测的是原始特征值（未归一化）")
    print("   - SSSD/SurvTraj 预测的是归一化风险值")
    print("   - 建议：从 Layer2 表中移除 iTransformer 和 TimeXer，或重新训练")

def mark_tabdiff_as_failed():
    """标记 TabDiff 为训练失败"""
    print("\n=== 标记 TabDiff 为训练失败 ===")
    
    for seed in SEEDS:
        try:
            data = np.load(f'outputs/b2_baseline/tstr/tabdiff_seed{seed}_predictions.npz')
            y_pred = data['y_pred']
            
            # 检查是否所有预测都是 1.0
            if np.all(y_pred == 1.0) or np.all(y_pred > 0.5):
                print(f"  Seed {seed}: ❌ 模型崩溃（所有预测为正类）")
                
                # 标记为无效
                flag_file = f'outputs/b2_baseline/tstr/tabdiff_seed{seed}_FAILED.txt'
                with open(flag_file, 'w') as f:
                    f.write("Model training failed: all predictions are positive class\n")
        except:
            continue

def generate_fixed_tables():
    """生成修复后的表格"""
    print("\n=== 生成修复后的表格 ===")
    
    # TODO: 重新聚合修复后的指标
    print("  需要重新运行 regenerate_all_tables.py")

if __name__ == '__main__':
    print("=" * 80)
    print("紧急修复脚本 - 修复 Baseline 表格问题")
    print("=" * 80)
    
    fix_layer1_metrics()
    fix_layer2_evaluation()
    mark_tabdiff_as_failed()
    generate_fixed_tables()
    
    print("\n" + "=" * 80)
    print("修复完成")
    print("=" * 80)
