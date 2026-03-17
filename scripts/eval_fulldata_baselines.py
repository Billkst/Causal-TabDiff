#!/usr/bin/env python3
"""
全量数据 Baseline 统一评估脚本
从预测 .npz 文件生成指标 JSON，然后汇总成正式表。

用法:
    python -u scripts/eval_fulldata_baselines.py \
        --input_dir outputs/fulldata_baselines \
        --output_dir outputs/fulldata_baselines/formal_runs
"""
import sys
import os
import json
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from evaluation.metrics import compute_all_metrics, find_optimal_threshold
from evaluation.plots import generate_all_plots

SEEDS = [42, 52, 62, 72, 82]


def fmt(values):
    """格式化 mean ± std"""
    clean = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
            if np.isfinite(fv):
                clean.append(fv)
        except (ValueError, TypeError):
            continue
    vals = clean
    if not vals:
        return 'N/A'
    arr = np.array(vals, dtype=float)
    return f"{arr.mean():.4f} +/- {arr.std():.4f}"


# ============================================================
# Phase 1: 从 predictions.npz → metrics.json
# ============================================================

def evaluate_layer1_prediction(pred_file, output_dir, model_name):
    """评估 Layer1 Direct 预测文件"""
    data = np.load(pred_file)

    val_y_true = data['val_y_true'].flatten()
    val_y_pred = data['val_y_pred'].flatten()
    test_y_true = data['test_y_true'].flatten()
    test_y_pred = data['test_y_pred'].flatten()

    print(f"  Val:  {len(val_y_true)} samples, {int(val_y_true.sum())} positive", flush=True)
    print(f"  Test: {len(test_y_true)} samples, {int(test_y_true.sum())} positive", flush=True)

    # 验证集 F1 最优阈值
    threshold, f1_val = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
    print(f"  Optimal threshold: {threshold:.4f} (Val F1: {f1_val:.4f})", flush=True)

    # 测试集固定阈值评估
    metrics = compute_all_metrics(test_y_true, test_y_pred, threshold=threshold)
    metrics['threshold'] = float(threshold)
    metrics['val_f1'] = float(f1_val)
    metrics['n_test'] = int(len(test_y_true))
    metrics['n_test_positive'] = int(test_y_true.sum())

    print(f"  AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f} | F1={metrics['f1']:.4f}", flush=True)

    # 保存指标
    os.makedirs(output_dir, exist_ok=True)
    metrics_serializable = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in metrics.items() if k != 'confusion_matrix'
    }
    if 'confusion_matrix' in metrics:
        metrics_serializable['confusion_matrix'] = metrics['confusion_matrix'].tolist()

    metrics_file = os.path.join(output_dir, f'{model_name}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    # 保存可视化
    test_y_pred_binary = (test_y_pred >= threshold).astype(int)
    try:
        plot_prefix = os.path.join(output_dir, model_name)
        generate_all_plots(test_y_true, test_y_pred, test_y_pred_binary, plot_prefix)
    except Exception as e:
        print(f"  [WARN] Plot generation failed: {e}", flush=True)

    return metrics


def evaluate_layer2_prediction(pred_file, output_dir, model_name):
    """评估 Layer2 Trajectory 预测文件"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    data = np.load(pred_file)
    val_y_pred = data['val_y_pred']
    val_y_true = data['val_y_true']
    val_y_mask = data['val_y_mask']
    test_y_pred = data['test_y_pred']
    test_y_true = data['test_y_true']
    test_y_mask = data['test_y_mask']

    # 处理 3D output
    if len(val_y_pred.shape) == 3:
        val_y_pred = val_y_pred[:, :, 0]
    if len(test_y_pred.shape) == 3:
        test_y_pred = test_y_pred[:, :, 0]

    # 轨迹指标
    min_len = min(test_y_pred.shape[1], test_y_true.shape[1], test_y_mask.shape[1])
    tp = test_y_pred[:, :min_len]
    tt = test_y_true[:, :min_len]
    tm = test_y_mask[:, :min_len]

    valid_pred = tp[tm > 0]
    valid_true = tt[tm > 0]

    finite_mask = np.isfinite(valid_pred) & np.isfinite(valid_true)
    valid_pred = valid_pred[finite_mask]
    valid_true = valid_true[finite_mask]

    traj_metrics = {
        'trajectory_mse': float(mean_squared_error(valid_true, valid_pred)) if len(valid_pred) > 0 else float('nan'),
        'trajectory_mae': float(mean_absolute_error(valid_true, valid_pred)) if len(valid_pred) > 0 else float('nan'),
        'valid_coverage': float(tm.sum() / tm.size) if tm.size > 0 else 0.0,
        'n_nan_filtered': int((~finite_mask).sum()),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{model_name}_layer2_metrics.json'), 'w') as f:
        json.dump(traj_metrics, f, indent=2)

    # Readout 分类指标: 从轨迹读出 2-year risk
    pred_2year = np.nan_to_num(test_y_pred[:, :2], nan=0.0, posinf=1e6, neginf=-1e6).mean(axis=1)
    pred_2year = np.clip(pred_2year, -500, 500)
    pred_2year = 1.0 / (1.0 + np.exp(-pred_2year))
    true_2year = (test_y_true[:, :2].sum(axis=1) > 0).astype(int)

    val_pred_2year = np.nan_to_num(val_y_pred[:, :2], nan=0.0, posinf=1e6, neginf=-1e6).mean(axis=1)
    val_pred_2year = np.clip(val_pred_2year, -500, 500)
    val_pred_2year = 1.0 / (1.0 + np.exp(-val_pred_2year))
    val_true_2year = (val_y_true[:, :2].sum(axis=1) > 0).astype(int)

    threshold, val_f1 = find_optimal_threshold(val_true_2year, val_pred_2year, metric='f1')
    readout_metrics = compute_all_metrics(true_2year, pred_2year, threshold=threshold)
    readout_metrics['val_optimal_f1'] = float(val_f1)
    readout_metrics['threshold'] = float(threshold)

    readout_serializable = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in readout_metrics.items() if k != 'confusion_matrix'
    }
    if 'confusion_matrix' in readout_metrics:
        readout_serializable['confusion_matrix'] = readout_metrics['confusion_matrix'].tolist()

    with open(os.path.join(output_dir, f'{model_name}_layer2_readout_metrics.json'), 'w') as f:
        json.dump(readout_serializable, f, indent=2)

    print(f"  Traj MSE={traj_metrics['trajectory_mse']:.4f} | MAE={traj_metrics['trajectory_mae']:.4f} | "
          f"Readout AUROC={readout_metrics['auroc']:.4f}", flush=True)

    return traj_metrics, readout_metrics


def evaluate_tstr_prediction(pred_file, output_dir, model_name):
    """评估 TSTR 预测文件"""
    data = np.load(pred_file)

    # train_generative_strict.py 保存格式
    val_y_true = data['val_y_true'].flatten()
    val_y_pred = data['val_y_pred'].flatten()
    test_y_true = data['test_y_true'].flatten()
    test_y_pred = data['test_y_pred'].flatten()

    threshold, f1_val = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
    metrics = compute_all_metrics(test_y_true, test_y_pred, threshold=threshold)
    metrics['threshold'] = float(threshold)
    metrics['val_f1'] = float(f1_val)

    print(f"  AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f} | F1={metrics['f1']:.4f}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    metrics_serializable = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in metrics.items() if k != 'confusion_matrix'
    }
    if 'confusion_matrix' in metrics:
        metrics_serializable['confusion_matrix'] = metrics['confusion_matrix'].tolist()

    with open(os.path.join(output_dir, f'{model_name}_tstr_metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    return metrics


# ============================================================
# Phase 2: 汇总生成正式表
# ============================================================

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def collect_metric(formal_dir, subdir, pattern, key, seeds=SEEDS):
    vals = []
    for seed in seeds:
        path = os.path.join(formal_dir, subdir, pattern.format(seed=seed))
        data = load_json(path)
        if data and key in data and data[key] is not None:
            vals.append(data[key])
    return vals


def generate_summary_tables(formal_dir, summary_dir):
    """生成四张正式汇总表 + 协议一致性检查"""
    os.makedirs(summary_dir, exist_ok=True)
    metric_cols = ['auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity',
                   'npv', 'accuracy', 'balanced_accuracy', 'mcc',
                   'brier_score', 'calibration_intercept', 'calibration_slope']

    # --- Layer1 Direct ---
    print("\n生成 baseline_layer1_direct.csv ...", flush=True)
    layer1_models = {
        'CausalForest': 'CausalForest',
        'iTransformer': 'iTransformer',
        'TSDiff': 'TSDiff',
        'STaSy': 'STaSy',
    }
    rows = []
    for display_name, stem in layer1_models.items():
        row = {'model': display_name}
        for col in metric_cols:
            vals = collect_metric(formal_dir, 'layer1', f'{stem}_seed{{seed}}_metrics.json', col)
            row[col] = fmt(vals)
        row['seeds_used'] = ','.join(map(str, SEEDS))
        row['protocol'] = 'val_f1_threshold_then_fixed_test'
        rows.append(row)
    layer1_df = pd.DataFrame(rows)
    layer1_df.to_csv(os.path.join(summary_dir, 'baseline_layer1_direct.csv'), index=False)

    # --- Layer1 TSTR ---
    print("生成 baseline_layer1_tstr.csv ...", flush=True)
    tstr_models = ['tabsyn', 'tabdiff', 'survtraj', 'sssd', 'tsdiff', 'stasy']
    rows = []
    for model in tstr_models:
        row = {'model': model}
        for col in metric_cols:
            vals = collect_metric(formal_dir, 'tstr', f'{model}_seed{{seed}}_tstr_metrics.json', col)
            row[col] = fmt(vals)
        # 检查是否有失败 seed
        failures = []
        for seed in SEEDS:
            fail_path = os.path.join(formal_dir, 'tstr', f'{model}_seed{seed}_FAILED.txt')
            if os.path.exists(fail_path):
                with open(fail_path) as fh:
                    failures.append(fh.read().strip())
        row['status'] = 'failed' if failures else 'ok'
        row['failure_reason'] = failures[0] if failures else ''
        row['synthetic_sample_size'] = 'train_set_size'
        row['seeds_used'] = ','.join(map(str, SEEDS))
        row['protocol'] = 'val_f1_threshold_then_fixed_test'
        rows.append(row)
    tstr_df = pd.DataFrame(rows)
    tstr_df.to_csv(os.path.join(summary_dir, 'baseline_layer1_tstr.csv'), index=False)

    # --- Layer2 ---
    print("生成 baseline_layer2.csv ...", flush=True)
    layer2_models = {
        'iTransformer': 'iTransformer',
        'TimeXer': 'TimeXer',
        'SSSD': 'SSSD',
        'SurvTraj': 'SurvTraj',
    }
    rows = []
    for display_name, stem in layer2_models.items():
        row = {'model': display_name}
        for traj_key in ['trajectory_mse', 'trajectory_mae', 'valid_coverage']:
            vals = collect_metric(formal_dir, 'layer2', f'{stem}_seed{{seed}}_layer2_metrics.json', traj_key)
            row[traj_key] = fmt(vals)
        for read_key in ['auroc', 'auprc', 'f1', 'precision', 'recall', 'brier_score',
                         'calibration_intercept', 'calibration_slope']:
            vals = collect_metric(formal_dir, 'layer2', f'{stem}_seed{{seed}}_layer2_readout_metrics.json', read_key)
            row[f'readout_{read_key}'] = fmt(vals)
        row['seeds_used'] = ','.join(map(str, SEEDS))
        row['protocol'] = 'val_f1_threshold_then_fixed_test'
        rows.append(row)
    layer2_df = pd.DataFrame(rows)
    layer2_df.to_csv(os.path.join(summary_dir, 'baseline_layer2.csv'), index=False)

    # --- Efficiency ---
    print("生成 baseline_efficiency.csv ...", flush=True)
    eff_sources = {
        'CausalForest': ('layer1', 'causal_forest_efficiency_seed{seed}.json'),
        'iTransformer': ('layer1', 'itransformer_efficiency_seed{seed}.json'),
        'TSDiff': ('layer1', 'tsdiff_efficiency_seed{seed}.json'),
        'STaSy': ('layer1', 'stasy_efficiency_seed{seed}.json'),
        'TimeXer': ('layer2', 'timexer_efficiency_seed{seed}.json'),
        'TabSyn': ('tstr', 'tabsyn_efficiency_seed{seed}.json'),
        'TabDiff': ('tstr', 'tabdiff_efficiency_seed{seed}.json'),
        'SurvTraj': ('tstr', 'survtraj_efficiency_seed{seed}.json'),
        'SSSD': ('tstr', 'sssd_efficiency_seed{seed}.json'),
    }
    rows = []
    for model_name, (subdir, pattern) in eff_sources.items():
        row = {'model': model_name}
        for key in ['total_params', 'trainable_params', 'inference_latency_ms_per_sample',
                     'throughput_samples_per_sec', 'total_training_wall_clock_sec', 'peak_gpu_memory_mb']:
            vals = []
            for seed in SEEDS:
                path = os.path.join(formal_dir, subdir, pattern.format(seed=seed))
                data = load_json(path)
                if data and key in data:
                    vals.append(data[key])
            row[key] = fmt(vals)
        rows.append(row)
    eff_df = pd.DataFrame(rows)
    eff_df.to_csv(os.path.join(summary_dir, 'baseline_efficiency.csv'), index=False)

    # --- 协议一致性检查 ---
    print("生成 baseline_protocol_consistency_check.csv ...", flush=True)
    check_rows = []

    # Layer1 Direct
    for model_name, stem in layer1_models.items():
        for seed in SEEDS:
            pred_path = os.path.join(formal_dir, 'layer1', f'{stem}_seed{seed}_predictions.npz')
            metrics_path = os.path.join(formal_dir, 'layer1', f'{stem}_seed{seed}_metrics.json')
            has_pred = os.path.exists(pred_path)
            has_metrics = os.path.exists(metrics_path)
            has_val = has_test = False
            threshold = None
            if has_pred:
                data = np.load(pred_path)
                has_val = 'val_y_true' in data.files and 'val_y_pred' in data.files
                has_test = 'test_y_true' in data.files and 'test_y_pred' in data.files
            if has_metrics:
                m = load_json(metrics_path)
                if m:
                    threshold = m.get('threshold')
            check_rows.append({
                'task': 'layer1_direct', 'model': model_name, 'seed': seed,
                'has_val_predictions': has_val, 'has_test_predictions': has_test,
                'metrics_file_exists': has_metrics, 'threshold': threshold,
                'formal_protocol': bool(has_val and has_test and has_metrics)
            })

    # Layer1 TSTR
    for model in tstr_models:
        for seed in SEEDS:
            pred_path = os.path.join(formal_dir, 'tstr', f'{model}_seed{seed}_predictions.npz')
            metrics_path = os.path.join(formal_dir, 'tstr', f'{model}_seed{seed}_tstr_metrics.json')
            has_pred = os.path.exists(pred_path)
            has_metrics = os.path.exists(metrics_path)
            has_val = has_test = False
            threshold = None
            if has_pred:
                data = np.load(pred_path)
                has_val = 'val_y_true' in data.files and 'val_y_pred' in data.files
                has_test = 'test_y_true' in data.files and 'test_y_pred' in data.files
            if has_metrics:
                m = load_json(metrics_path)
                if m:
                    threshold = m.get('threshold')
            check_rows.append({
                'task': 'layer1_tstr', 'model': model, 'seed': seed,
                'has_val_predictions': has_val, 'has_test_predictions': has_test,
                'metrics_file_exists': has_metrics, 'threshold': threshold,
                'formal_protocol': bool(has_val and has_test and has_metrics)
            })

    # Layer2
    for model_name, stem in layer2_models.items():
        for seed in SEEDS:
            pred_path = os.path.join(formal_dir, 'layer2', f'{stem}_seed{seed}_layer2.npz')
            metrics_path = os.path.join(formal_dir, 'layer2', f'{stem}_seed{seed}_layer2_readout_metrics.json')
            has_pred = os.path.exists(pred_path)
            has_metrics = os.path.exists(metrics_path)
            has_val = has_test = False
            threshold = None
            if has_pred:
                data = np.load(pred_path)
                has_val = 'val_y_true' in data.files or 'val_y_pred' in data.files
                has_test = 'test_y_true' in data.files or 'test_y_pred' in data.files
            if has_metrics:
                m = load_json(metrics_path)
                if m:
                    threshold = m.get('threshold')
            check_rows.append({
                'task': 'layer2', 'model': model_name, 'seed': seed,
                'has_val_predictions': has_val, 'has_test_predictions': has_test,
                'metrics_file_exists': has_metrics, 'threshold': threshold,
                'formal_protocol': bool(has_val and has_test and has_metrics)
            })

    check_df = pd.DataFrame(check_rows)
    check_df.to_csv(os.path.join(summary_dir, 'baseline_protocol_consistency_check.csv'), index=False)

    pass_rate = check_df['formal_protocol'].mean()
    print(f"\n协议一致性通过率: {pass_rate:.4f}", flush=True)

    return layer1_df, tstr_df, layer2_df, eff_df, check_df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='全量数据 Baseline 统一评估')
    parser.add_argument('--input_dir', type=str, default='outputs/fulldata_baselines',
                        help='训练输出根目录')
    parser.add_argument('--output_dir', type=str, default='outputs/fulldata_baselines/formal_runs',
                        help='正式评估结果输出目录')
    args = parser.parse_args()

    input_dir = args.input_dir
    formal_dir = args.output_dir

    print("=" * 60, flush=True)
    print("全量数据 Baseline 统一评估", flush=True)
    print(f"输入目录: {input_dir}", flush=True)
    print(f"输出目录: {formal_dir}", flush=True)
    print("=" * 60, flush=True)

    # --- Layer1 Direct 评估 ---
    print("\n=== Layer1 Direct 评估 ===", flush=True)
    layer1_dir = os.path.join(input_dir, 'layer1')
    layer1_formal = os.path.join(formal_dir, 'layer1')

    layer1_models = {
        'causal_forest': 'CausalForest',
        'itransformer': 'iTransformer',
        'tsdiff': 'TSDiff',
        'stasy': 'STaSy',
    }

    for file_stem, display_name in layer1_models.items():
        for seed in SEEDS:
            pred_file = os.path.join(layer1_dir, f'{file_stem}_seed{seed}_predictions.npz')
            if not os.path.exists(pred_file):
                print(f"  [SKIP] {display_name} seed={seed}: 预测文件不存在", flush=True)
                continue
            print(f"\n评估 {display_name} seed={seed}", flush=True)
            evaluate_layer1_prediction(pred_file, layer1_formal, f'{display_name}_seed{seed}')

            # 复制效率文件
            eff_src = os.path.join(layer1_dir, f'{file_stem}_efficiency_seed{seed}.json')
            eff_dst = os.path.join(layer1_formal, f'{file_stem}_efficiency_seed{seed}.json')
            if os.path.exists(eff_src) and not os.path.exists(eff_dst):
                import shutil
                os.makedirs(os.path.dirname(eff_dst), exist_ok=True)
                shutil.copy2(eff_src, eff_dst)

    # --- Layer1 TSTR 评估 ---
    print("\n=== Layer1 TSTR 评估 ===", flush=True)
    tstr_dir = os.path.join(input_dir, 'tstr')
    tstr_formal = os.path.join(formal_dir, 'tstr')

    tstr_models = ['tabsyn', 'tabdiff', 'survtraj', 'sssd', 'tsdiff', 'stasy']
    for model in tstr_models:
        for seed in SEEDS:
            pred_file = os.path.join(tstr_dir, f'{model}_seed{seed}_predictions.npz')
            if not os.path.exists(pred_file):
                print(f"  [SKIP] {model} seed={seed}: 预测文件不存在", flush=True)
                continue
            print(f"\n评估 {model} TSTR seed={seed}", flush=True)
            evaluate_tstr_prediction(pred_file, tstr_formal, f'{model}_seed{seed}')

            # 复制效率文件
            eff_src = os.path.join(tstr_dir, f'{model}_efficiency_seed{seed}.json')
            eff_dst = os.path.join(tstr_formal, f'{model}_efficiency_seed{seed}.json')
            if os.path.exists(eff_src) and not os.path.exists(eff_dst):
                import shutil
                os.makedirs(os.path.dirname(eff_dst), exist_ok=True)
                shutil.copy2(eff_src, eff_dst)

    # --- Layer2 评估 ---
    print("\n=== Layer2 Trajectory 评估 ===", flush=True)
    layer2_dir = os.path.join(input_dir, 'layer2')
    layer2_formal = os.path.join(formal_dir, 'layer2')

    layer2_models = {
        'itransformer': 'iTransformer',
        'timexer': 'TimeXer',
        'sssd': 'SSSD',
        'survtraj': 'SurvTraj',
    }
    for file_stem, display_name in layer2_models.items():
        for seed in SEEDS:
            pred_file = os.path.join(layer2_dir, f'{file_stem}_seed{seed}_layer2.npz')
            if not os.path.exists(pred_file):
                print(f"  [SKIP] {display_name} seed={seed}: 预测文件不存在", flush=True)
                continue
            print(f"\n评估 {display_name} Layer2 seed={seed}", flush=True)
            evaluate_layer2_prediction(pred_file, layer2_formal, f'{display_name}_seed{seed}')

            # 复制效率文件
            eff_src = os.path.join(layer2_dir, f'{file_stem}_efficiency_seed{seed}.json')
            eff_dst = os.path.join(layer2_formal, f'{file_stem}_efficiency_seed{seed}.json')
            if os.path.exists(eff_src) and not os.path.exists(eff_dst):
                import shutil
                os.makedirs(os.path.dirname(eff_dst), exist_ok=True)
                shutil.copy2(eff_src, eff_dst)

    # --- 生成汇总表 ---
    print("\n=== 生成汇总表 ===", flush=True)
    summary_dir = os.path.join(input_dir, 'summaries')
    layer1_df, tstr_df, layer2_df, eff_df, check_df = generate_summary_tables(formal_dir, summary_dir)

    # --- 最终报告 ---
    print("\n" + "=" * 60, flush=True)
    print("全量数据 Baseline 评估完成", flush=True)
    print(f"汇总表目录: {summary_dir}/", flush=True)
    all_pass = bool(check_df['formal_protocol'].all())
    if all_pass:
        print("✓ 所有正式协议检查通过", flush=True)
    else:
        n_fail = (~check_df['formal_protocol']).sum()
        print(f"⚠ {n_fail} 项协议检查未通过", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
