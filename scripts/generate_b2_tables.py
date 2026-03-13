"""
B2 Baseline 统一表格生成脚本
生成四张必需表格: baseline_main_table.csv, baseline_layer2_table.csv, 
baseline_tstr_table.csv, baseline_efficiency_table.csv
"""
import os
import json
import numpy as np
import pandas as pd
import glob

SEEDS = [42, 52, 62, 72, 82]
METRICS = ['auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity',
           'npv', 'accuracy', 'balanced_accuracy', 'mcc', 'brier_score',
           'calibration_intercept', 'calibration_slope']


def aggregate_model_results(model_name, results_dir, metric_file_pattern):
    """聚合单个模型的多 seed 结果"""
    model_metrics = {m: [] for m in METRICS}
    
    for seed in SEEDS:
        metric_file = os.path.join(results_dir, metric_file_pattern.format(model=model_name, seed=seed))
        if not os.path.exists(metric_file):
            continue
        
        with open(metric_file, 'r') as f:
            data = json.load(f)
        
        for metric in METRICS:
            if metric in data:
                model_metrics[metric].append(data[metric])
    
    if not model_metrics['auroc']:
        return None
    
    row = {'model': model_name}
    for metric in METRICS:
        values = model_metrics[metric]
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0
            row[metric] = f"{mean:.4f} ± {std:.4f}"
        else:
            row[metric] = "N/A"
    
    return row


def generate_main_table(output_dir):
    """生成 Layer1 主结果表"""
    print("\n=== 生成 baseline_main_table.csv ===")
    
    models = ['CausalForest', 'iTransformer', 'TSDiff', 'STaSy', 
              'tabsyn', 'tabdiff', 'survtraj', 'sssd']
    
    rows = []
    for model in models:
        row = aggregate_model_results(model, 'outputs/b2_baseline/layer1', '{model}_seed{seed}_metrics.json')
        if not row:
            row = aggregate_model_results(model, 'outputs/b2_baseline/tstr', '{model}_seed{seed}_metrics.json')
        if not row:
            row = aggregate_model_results(model.lower(), 'outputs/tstr_baselines', '{model}_seed{seed}_metrics.json')
        if row:
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = os.path.join(output_dir, 'baseline_main_table.csv')
        df.to_csv(output_file, index=False)
        print(f"✓ 已保存: {output_file}")
        print(df.to_string(index=False))
    else:
        print("⚠️  无可用结果")


def generate_layer2_table(output_dir):
    """生成 Layer2 结果表"""
    print("\n=== 生成 baseline_layer2_table.csv ===")
    
    models = ['iTransformer', 'TimeXer', 'SSSD', 'SurvTraj']
    rows = []
    
    for model in models:
        model_data = {'model': model}
        traj_mse, traj_mae, coverage = [], [], []
        
        for seed in SEEDS:
            file = f'outputs/b2_baseline/layer2/{model}_seed{seed}_layer2_metrics.json'
            if not os.path.exists(file):
                file = f'outputs/b2_baseline/layer2/{model.lower()}_seed{seed}_layer2_metrics.json'
            if os.path.exists(file):
                with open(file, 'r') as f:
                    data = json.load(f)
                traj_mse.append(data.get('trajectory_mse', np.nan))
                traj_mae.append(data.get('trajectory_mae', np.nan))
                coverage.append(data.get('valid_coverage', np.nan))
        
        if traj_mse:
            model_data['trajectory_mse'] = f"{np.mean(traj_mse):.4f} ± {np.std(traj_mse, ddof=1):.4f}"
            model_data['trajectory_mae'] = f"{np.mean(traj_mae):.4f} ± {np.std(traj_mae, ddof=1):.4f}"
            model_data['valid_coverage'] = f"{np.mean(coverage):.4f} ± {np.std(coverage, ddof=1):.4f}"
            rows.append(model_data)
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = os.path.join(output_dir, 'baseline_layer2_table.csv')
        df.to_csv(output_file, index=False)
        print(f"✓ 已保存: {output_file}")
    else:
        print("⚠️  无可用结果")


def generate_tstr_table(output_dir):
    """生成 TSTR 结果表"""
    print("\n=== 生成 baseline_tstr_table.csv ===")
    
    models = ['tabsyn', 'tabdiff', 'survtraj', 'sssd']
    rows = []
    
    for model in models:
        row = aggregate_model_results(model, 'outputs/b2_baseline/tstr', '{model}_seed{seed}_metrics.json')
        if row:
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = os.path.join(output_dir, 'baseline_tstr_table.csv')
        df.to_csv(output_file, index=False)
        print(f"✓ 已保存: {output_file}")
    else:
        print("⚠️  无可用结果")


def generate_efficiency_table(output_dir):
    """生成效率结果表"""
    print("\n=== 生成 baseline_efficiency_table.csv ===")
    
    models = ['CausalForest', 'iTransformer', 'TimeXer', 'TSDiff', 'STaSy',
              'TabSyn_strict', 'TabDiff_strict', 'SurvTraj_strict', 'SSSD_strict']
    
    rows = []
    for model in models:
        eff_metrics = []
        for seed in SEEDS:
            file = f'outputs/b2_baseline/efficiency/{model}_seed{seed}_efficiency.json'
            if os.path.exists(file):
                with open(file, 'r') as f:
                    eff_metrics.append(json.load(f))
        
        if eff_metrics:
            row = {'model': model}
            for key in ['total_training_wall_clock_sec', 'average_epoch_time_sec',
                       'inference_latency_ms_per_sample', 'peak_gpu_memory_mb', 'peak_cpu_ram_mb']:
                values = [m.get(key, np.nan) for m in eff_metrics if key in m]
                if values:
                    row[key] = f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}"
                else:
                    row[key] = "N/A"
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = os.path.join(output_dir, 'baseline_efficiency_table.csv')
        df.to_csv(output_file, index=False)
        print(f"✓ 已保存: {output_file}")
    else:
        print("⚠️  无可用结果")


def main():
    output_dir = 'outputs/b2_baseline/summaries'
    os.makedirs(output_dir, exist_ok=True)
    
    generate_main_table(output_dir)
    generate_layer2_table(output_dir)
    generate_tstr_table(output_dir)
    generate_efficiency_table(output_dir)
    
    print("\n" + "="*60)
    print("✓ 所有表格生成完成")
    print("="*60)


if __name__ == '__main__':
    main()
