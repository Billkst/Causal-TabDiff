#!/usr/bin/env python3
"""
B2 Baseline 文件对账清单生成脚本
"""
import os
import json
import glob
from collections import defaultdict

SEEDS = [42, 52, 62, 72, 82]

def check_file_exists(path):
    return "✓" if os.path.exists(path) else "✗"

def generate_reconciliation_report():
    report = []
    report.append("=" * 80)
    report.append("B2 Baseline 文件对账清单")
    report.append("=" * 80)
    
    # Layer1 对账
    report.append("\n### Layer1 Baseline 对账")
    layer1_models = ['CausalForest', 'iTransformer', 'tabsyn', 'tabdiff', 'survtraj', 'sssd']
    
    for model in layer1_models:
        report.append(f"\n**{model}**:")
        for seed in SEEDS:
            pred_file = f"outputs/b2_baseline/layer1/{model.lower()}_seed{seed}_predictions.npz"
            if not os.path.exists(pred_file):
                pred_file = f"outputs/b2_baseline/tstr/{model.lower()}_seed{seed}_predictions.npz"
            if not os.path.exists(pred_file):
                pred_file = f"outputs/tslib_models/{model.lower()}_seed{seed}_predictions.npz"
            
            metrics_file = f"outputs/b2_baseline/layer1/{model}_seed{seed}_metrics.json"
            if not os.path.exists(metrics_file):
                metrics_file = f"outputs/b2_baseline/tstr/{model.lower()}_seed{seed}_metrics.json"
            
            plots_dir = f"outputs/b2_baseline/layer1/{model}_seed{seed}"
            if not os.path.exists(plots_dir):
                plots_dir = f"outputs/b2_baseline/tstr/{model.lower()}_seed{seed}"
            
            report.append(f"  Seed {seed}: pred={check_file_exists(pred_file)} "
                         f"metrics={check_file_exists(metrics_file)} "
                         f"plots={check_file_exists(plots_dir)}")
    
    # Layer2 对账
    report.append("\n\n### Layer2 Baseline 对账")
    layer2_models = ['iTransformer', 'TimeXer', 'SSSD', 'SurvTraj']
    
    for model in layer2_models:
        report.append(f"\n**{model}**:")
        for seed in SEEDS:
            pred_file = f"outputs/tslib_layer2/{model.lower()}_seed{seed}_layer2.npz"
            if not os.path.exists(pred_file):
                pred_file = f"outputs/b2_baseline/layer2/{model.lower()}_seed{seed}_layer2.npz"
            
            metrics_file = f"outputs/b2_baseline/layer2/{model}_seed{seed}_layer2_metrics.json"
            
            report.append(f"  Seed {seed}: pred={check_file_exists(pred_file)} "
                         f"metrics={check_file_exists(metrics_file)}")
    
    # 效率表对账
    report.append("\n\n### 效率数据对账")
    all_models = layer1_models + layer2_models
    
    for model in all_models:
        report.append(f"\n**{model}**:")
        for seed in SEEDS:
            eff_file = f"outputs/b2_baseline/layer1/{model.lower()}_efficiency_seed{seed}.json"
            if not os.path.exists(eff_file):
                eff_file = f"outputs/b2_baseline/layer2/{model.lower()}_efficiency_seed{seed}.json"
            if not os.path.exists(eff_file):
                eff_file = f"outputs/tslib_models/{model.lower()}_efficiency_seed{seed}.json"
            if not os.path.exists(eff_file):
                eff_file = f"outputs/tslib_layer2/{model.lower()}_efficiency_seed{seed}.json"
            
            report.append(f"  Seed {seed}: {check_file_exists(eff_file)}")
    
    # 汇总表对账
    report.append("\n\n### 汇总表对账")
    tables = [
        'outputs/b2_baseline/summaries/baseline_main_table.csv',
        'outputs/b2_baseline/summaries/baseline_layer2_table.csv',
        'outputs/b2_baseline/summaries/baseline_tstr_table.csv',
        'outputs/b2_baseline/summaries/baseline_efficiency_table.csv'
    ]
    
    for table in tables:
        status = check_file_exists(table)
        report.append(f"{status} {os.path.basename(table)}")
    
    # 统计
    report.append("\n\n### 统计摘要")
    total_predictions = len(glob.glob("outputs/b2_baseline/**/*.npz", recursive=True))
    total_metrics = len(glob.glob("outputs/b2_baseline/**/*_metrics.json", recursive=True))
    total_plots = len(glob.glob("outputs/b2_baseline/**/*.png", recursive=True))
    total_efficiency = len(glob.glob("outputs/**/efficiency*.json", recursive=True))
    
    report.append(f"- 预测文件: {total_predictions}")
    report.append(f"- 指标文件: {total_metrics}")
    report.append(f"- 图表文件: {total_plots}")
    report.append(f"- 效率文件: {total_efficiency}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

if __name__ == '__main__':
    report = generate_reconciliation_report()
    print(report)
    
    with open('outputs/b2_baseline/reconciliation_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✓ 对账清单已保存到: outputs/b2_baseline/reconciliation_report.txt")
