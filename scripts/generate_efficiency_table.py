#!/usr/bin/env python3
"""
生成 baseline_efficiency_table.csv
"""
import os
import json
import pandas as pd
import glob

SEEDS = [42, 52, 62, 72, 82]

def collect_efficiency_data():
    """收集所有效率数据"""
    all_data = []
    
    # Layer1 models
    layer1_models = ['CausalForest', 'iTransformer', 'TSDiff', 'STaSy', 
                     'TabSyn', 'TabDiff', 'SurvTraj', 'SSSD']
    
    for model in layer1_models:
        for seed in SEEDS:
            # 尝试多个可能的路径
            paths = [
                f'outputs/b2_baseline/layer1/{model.lower()}_efficiency_seed{seed}.json',
                f'outputs/b2_baseline/tstr/{model.lower()}_efficiency_seed{seed}.json',
                f'outputs/tslib_models/{model.lower()}_efficiency_seed{seed}.json',
                f'outputs/tstr_baselines/{model.lower()}_efficiency_seed{seed}.json',
                f'outputs/b2_baseline/tsdiff_stasy/{model.lower()}_efficiency_seed{seed}.json',
            ]
            
            for path in paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    data['model'] = model
                    data['seed'] = seed
                    data['layer'] = 'layer1'
                    all_data.append(data)
                    break
    
    # Layer2 models
    layer2_models = ['iTransformer', 'TimeXer', 'SSSD', 'SurvTraj']
    
    for model in layer2_models:
        for seed in SEEDS:
            paths = [
                f'outputs/b2_baseline/layer2/{model.lower()}_efficiency_seed{seed}.json',
                f'outputs/tslib_layer2/{model.lower()}_efficiency_seed{seed}.json',
            ]
            
            for path in paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    data['model'] = f'{model}_layer2'
                    data['seed'] = seed
                    data['layer'] = 'layer2'
                    all_data.append(data)
                    break
    
    return all_data

def generate_efficiency_table():
    """生成效率表"""
    print("=== 生成 baseline_efficiency_table.csv ===")
    
    all_data = collect_efficiency_data()
    
    if not all_data:
        print("⚠️  无可用效率数据")
        return
    
    df = pd.DataFrame(all_data)
    
    # 选择关键列
    columns = ['model', 'seed', 'layer', 'total_params', 'trainable_params',
               'total_training_wall_clock_sec', 'average_epoch_time_sec',
               'total_training_gpu_hours', 'total_training_cpu_hours',
               'peak_gpu_memory_mb', 'peak_cpu_ram_mb',
               'device_type']
    
    # 只保留存在的列
    columns = [c for c in columns if c in df.columns]
    df = df[columns]
    
    output_dir = 'outputs/b2_baseline/summaries'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'baseline_efficiency_table.csv')
    df.to_csv(output_file, index=False)
    
    print(f"✓ 已保存: {output_file}")
    print(f"✓ 收集到 {len(all_data)} 条效率记录")
    print(df.to_string(index=False))

if __name__ == '__main__':
    generate_efficiency_table()
