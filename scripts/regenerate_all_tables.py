#!/usr/bin/env python3
"""
重新生成所有 baseline 表格 - 最终版本
"""
import os
import json
import numpy as np
import pandas as pd
import glob

SEEDS = [42, 52, 62, 72, 82]

LAYER1_METRICS = ['auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity', 
                  'npv', 'accuracy', 'balanced_accuracy', 'mcc', 'brier_score',
                  'calibration_intercept', 'calibration_slope']

def aggregate_metrics(model_name, search_paths, metric_pattern):
    """聚合单个模型的多 seed 结果"""
    model_metrics = {m: [] for m in LAYER1_METRICS}
    
    for seed in SEEDS:
        found = False
        for search_path in search_paths:
            metric_file = os.path.join(search_path, metric_pattern.format(model=model_name, seed=seed))
            if os.path.exists(metric_file):
                try:
                    with open(metric_file, 'r') as f:
                        data = json.load(f)
                    for metric in LAYER1_METRICS:
                        if metric in data:
                            model_metrics[metric].append(data[metric])
                    found = True
                    break
                except:
                    continue
        
        if not found:
            print(f"  ⚠️  {model_name} seed {seed}: 未找到 metrics")
    
    if not model_metrics['auroc']:
        return None
    
    row = {'model': model_name}
    for metric in LAYER1_METRICS:
        values = model_metrics[metric]
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0
            row[metric] = f"{mean:.4f} ± {std:.4f}"
        else:
            row[metric] = "N/A"
    
    return row

def generate_layer1_direct_table():
    """生成 Layer1 直接预测表"""
    print("\n=== 生成 Layer1 直接预测表 ===")
    
    models = ['CausalForest', 'iTransformer', 'TSDiff', 'STaSy']
    search_paths = [
        'outputs/b2_baseline/layer1',
        'outputs/b2_baseline/tstr',
        'outputs/b2_baseline/tsdiff_stasy',
        'outputs/tslib_models'
    ]
    
    rows = []
    for model in models:
        row = aggregate_metrics(model, search_paths, '{model}_seed{seed}_metrics.json')
        if not row:
            row = aggregate_metrics(model.lower(), search_paths, '{model}_seed{seed}_metrics.json')
        if row:
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = 'outputs/b2_baseline/summaries/baseline_layer1_direct.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ 已保存: {output_file}")
        print(df[['model', 'auroc', 'auprc', 'f1']].to_string(index=False))
    
    return df if rows else None

def generate_layer1_tstr_table():
    """生成 Layer1 TSTR 表（生成式模型）"""
    print("\n=== 生成 Layer1 TSTR 表（生成式模型）===")
    
    models = ['tabsyn', 'tabdiff', 'survtraj', 'sssd']
    search_paths = [
        'outputs/b2_baseline/tstr',
        'outputs/tstr_baselines',
        'outputs/b2_baseline/layer1'
    ]
    
    rows = []
    for model in models:
        row = aggregate_metrics(model, search_paths, '{model}_seed{seed}_metrics.json')
        if row:
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = 'outputs/b2_baseline/summaries/baseline_layer1_tstr.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ 已保存: {output_file}")
        print(df[['model', 'auroc', 'auprc', 'f1']].to_string(index=False))
    
    return df if rows else None

def generate_layer2_table():
    """生成 Layer2 表（包含 trajectory 和 2-year readout 指标）"""
    print("\n=== 生成 Layer2 表 ===")
    
    models = ['iTransformer', 'TimeXer', 'SSSD', 'SurvTraj']
    
    rows = []
    for model in models:
        traj_metrics = {'trajectory_mse': [], 'trajectory_mae': [], 'valid_coverage': []}
        readout_metrics = {'readout_auroc': [], 'readout_auprc': [], 'readout_f1': []}
        
        for seed in SEEDS:
            # 尝试多种文件名格式
            traj_files = [
                f'outputs/b2_baseline/layer2/{model}_seed{seed}_layer2_metrics.json',
                f'outputs/b2_baseline/layer2/{model.capitalize()}_seed{seed}_layer2_metrics.json',
                f'outputs/b2_baseline/layer2/{model.lower()}_seed{seed}_layer2_metrics.json'
            ]
            
            for traj_file in traj_files:
                if os.path.exists(traj_file):
                    with open(traj_file, 'r') as f:
                        data = json.load(f)
                    traj_metrics['trajectory_mse'].append(data.get('trajectory_mse', np.nan))
                    traj_metrics['trajectory_mae'].append(data.get('trajectory_mae', np.nan))
                    traj_metrics['valid_coverage'].append(data.get('valid_coverage', np.nan))
                    break
            
            readout_files = [
                f'outputs/b2_baseline/layer2/{model}_seed{seed}_layer2_readout_metrics.json',
                f'outputs/b2_baseline/layer2/{model.capitalize()}_seed{seed}_layer2_readout_metrics.json',
                f'outputs/b2_baseline/layer2/{model.lower()}_seed{seed}_layer2_readout_metrics.json'
            ]
            
            for readout_file in readout_files:
                if os.path.exists(readout_file):
                    with open(readout_file, 'r') as f:
                        data = json.load(f)
                    readout_metrics['readout_auroc'].append(data.get('auroc', np.nan))
                    readout_metrics['readout_auprc'].append(data.get('auprc', np.nan))
                    readout_metrics['readout_f1'].append(data.get('f1', np.nan))
                    break
        
        if traj_metrics['trajectory_mse']:
            row = {'model': model}
            for key, values in {**traj_metrics, **readout_metrics}.items():
                if values:
                    mean = np.nanmean(values)
                    std = np.nanstd(values, ddof=1) if len(values) > 1 else 0
                    row[key] = f"{mean:.4f} ± {std:.4f}"
                else:
                    row[key] = "N/A"
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = 'outputs/b2_baseline/summaries/baseline_layer2.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ 已保存: {output_file}")
        print(df.to_string(index=False))
    
    return df if rows else None

def generate_efficiency_table():
    """生成效率表（计算 mean ± std）"""
    print("\n=== 生成效率表 ===")
    
    all_models = ['CausalForest', 'iTransformer', 'TSDiff', 'STaSy',
                  'tabsyn', 'tabdiff', 'survtraj', 'sssd',
                  'iTransformer_layer2', 'TimeXer', 'SSSD_layer2', 'SurvTraj_layer2']
    
    rows = []
    for model in all_models:
        eff_data = {'total_training_wall_clock_sec': [], 'peak_gpu_memory_mb': [], 
                    'total_params': [], 'trainable_params': []}
        
        for seed in SEEDS:
            eff_files = glob.glob(f'outputs/**/{model.lower()}_efficiency_seed{seed}.json', recursive=True)
            if eff_files:
                with open(eff_files[0], 'r') as f:
                    data = json.load(f)
                for key in eff_data.keys():
                    if key in data:
                        eff_data[key].append(data[key])
        
        if any(eff_data.values()):
            row = {'model': model}
            for key, values in eff_data.items():
                if values:
                    mean = np.mean(values)
                    std = np.std(values, ddof=1) if len(values) > 1 else 0
                    row[key] = f"{mean:.2f} ± {std:.2f}"
                else:
                    row[key] = "N/A"
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = 'outputs/b2_baseline/summaries/baseline_efficiency.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ 已保存: {output_file}")
        print(df.to_string(index=False))
    
    return df if rows else None

if __name__ == '__main__':
    print("=" * 80)
    print("重新生成所有 Baseline 表格")
    print("=" * 80)
    
    generate_layer1_direct_table()
    generate_layer1_tstr_table()
    generate_layer2_table()
    generate_efficiency_table()
    
    print("\n" + "=" * 80)
    print("✓ 所有表格生成完成")
    print("=" * 80)
