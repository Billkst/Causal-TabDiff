#!/usr/bin/env python3
"""
从现有 .npz 预测文件重建 4 张 formal 对比表（v2）：
1) baseline_layer1_direct_v2.csv
2) baseline_layer1_tstr_v2.csv
3) baseline_layer2_v2.csv
4) baseline_efficiency_v2.csv

关键协议：
- Layer1 Direct: CausalTabDiff 已经 Platt 校准，禁止再次校准；其余 baseline 统一做 Platt。
- Layer1/TSTR: AUROC/AUPRC 用原始 test 预测；阈值类/校准类指标用校准后 test 概率。
- Layer2: 轨迹指标先统一到概率空间；2-year readout 自动判断 logits/prob 后再做 Platt。
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, 'src')
import importlib

_metrics = importlib.import_module('evaluation.metrics')
compute_all_metrics = _metrics.compute_all_metrics
compute_ranking_metrics = _metrics.compute_ranking_metrics
find_optimal_threshold = _metrics.find_optimal_threshold
platt_calibrate = _metrics.platt_calibrate

SEEDS = [42, 52, 62, 72, 82]
ROOT = 'outputs/fulldata_baselines'
FORMAL_RUNS = os.path.join(ROOT, 'formal_runs')
SUMMARY_DIR = os.path.join(ROOT, 'summaries_v2')


def fmt(values):
    clean = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            clean.append(fv)
    if not clean:
        return 'N/A'
    arr = np.asarray(clean, dtype=float)
    return f"{arr.mean():.4f} +/- {arr.std():.4f}"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def pick_existing_path(candidates, warn_prefix):
    for p in candidates:
        if os.path.exists(p):
            return p
    print(f"[WARN] {warn_prefix} | 未找到文件，候选: {candidates}", flush=True)
    return None


def load_layer1_npz(path):
    data = np.load(path)
    return (
        data['val_y_true'].astype(float).flatten(),
        data['val_y_pred'].astype(float).flatten(),
        data['test_y_true'].astype(float).flatten(),
        data['test_y_pred'].astype(float).flatten(),
    )


def evaluate_binary_seed(
    val_y_true,
    val_y_pred_raw,
    test_y_true,
    test_y_pred_raw,
    apply_platt,
):
    if apply_platt:
        val_y_pred_cal = platt_calibrate(val_y_true, val_y_pred_raw, val_y_pred_raw)
        test_y_pred_cal = platt_calibrate(val_y_true, val_y_pred_raw, test_y_pred_raw)
    else:
        val_y_pred_cal = val_y_pred_raw
        test_y_pred_cal = test_y_pred_raw

    ranking = compute_ranking_metrics(test_y_true, test_y_pred_raw)
    threshold, _ = find_optimal_threshold(val_y_true, val_y_pred_cal, metric='f1')
    core = compute_all_metrics(
        y_true=test_y_true,
        y_pred_proba=test_y_pred_cal,
        threshold=threshold,
        val_y_true=val_y_true,
        val_y_pred_proba=val_y_pred_cal,
    )

    out = dict(core)
    out['auroc'] = ranking['auroc']
    out['auprc'] = ranking['auprc']

    pos_rate = float(np.mean(test_y_true)) if len(test_y_true) > 0 else np.nan
    out['auprc_over_baseline'] = (
        float(out['auprc']) / pos_rate if np.isfinite(pos_rate) and pos_rate > 0 else np.nan
    )
    return out


def aggregate_binary_table(rows_seed_metrics):
    columns = [
        'model',
        'auroc',
        'auprc',
        'auprc_over_baseline',
        'f1',
        'precision',
        'recall',
        'specificity',
        'npv',
        'accuracy',
        'balanced_accuracy',
        'mcc',
        'brier_score',
        'eo_ratio',
    ]
    metric_cols = columns[1:]
    rows = []
    for model_name, seed_metrics in rows_seed_metrics.items():
        row = {'model': model_name}
        for c in metric_cols:
            row[c] = fmt([m.get(c) for m in seed_metrics])
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[columns]


def build_layer1_direct():
    print('\n[1/4] 生成 Layer1 Direct v2 ...', flush=True)
    model_specs = [
        ('CausalForest', 'causal_forest', True),
        ('iTransformer', 'itransformer', True),
        ('TSDiff', 'tsdiff', True),
        ('STaSy', 'stasy', True),
        ('CausalTabDiff', None, False),
    ]

    rows_seed_metrics = {name: [] for name, _, _ in model_specs}

    for display_name, stem, apply_platt in model_specs:
        for seed in SEEDS:
            if display_name == 'CausalTabDiff':
                pred_path = f'predictions/landmark/phase2_seed{seed}_calibrated.npz'
                path = pick_existing_path([pred_path], f'Layer1 Direct {display_name} seed={seed}')
            else:
                path = pick_existing_path(
                    [
                        os.path.join(FORMAL_RUNS, 'layer1', f'{stem}_seed{seed}_predictions.npz'),
                        os.path.join(ROOT, 'layer1', f'{stem}_seed{seed}_predictions.npz'),
                    ],
                    f'Layer1 Direct {display_name} seed={seed}',
                )
            if path is None:
                continue

            print(f'  [OK] {display_name} seed={seed} <- {path}', flush=True)
            val_y_true, val_y_pred, test_y_true, test_y_pred = load_layer1_npz(path)
            metrics = evaluate_binary_seed(
                val_y_true=val_y_true,
                val_y_pred_raw=val_y_pred,
                test_y_true=test_y_true,
                test_y_pred_raw=test_y_pred,
                apply_platt=apply_platt,
            )
            rows_seed_metrics[display_name].append(metrics)

    return aggregate_binary_table(rows_seed_metrics)


def build_layer1_tstr():
    print('\n[2/4] 生成 Layer1 TSTR v2 ...', flush=True)
    models = ['tabsyn', 'tabdiff', 'survtraj', 'sssd', 'tsdiff', 'stasy']
    rows_seed_metrics = {m: [] for m in models}

    for model in models:
        for seed in SEEDS:
            path = pick_existing_path(
                [
                    os.path.join(FORMAL_RUNS, 'tstr', f'{model}_seed{seed}_tstr_predictions.npz'),
                    os.path.join(FORMAL_RUNS, 'tstr', f'{model}_seed{seed}_predictions.npz'),
                    os.path.join(ROOT, 'tstr', f'{model}_seed{seed}_tstr_predictions.npz'),
                    os.path.join(ROOT, 'tstr', f'{model}_seed{seed}_predictions.npz'),
                ],
                f'Layer1 TSTR {model} seed={seed}',
            )
            if path is None:
                continue

            print(f'  [OK] {model} seed={seed} <- {path}', flush=True)
            val_y_true, val_y_pred, test_y_true, test_y_pred = load_layer1_npz(path)
            metrics = evaluate_binary_seed(
                val_y_true=val_y_true,
                val_y_pred_raw=val_y_pred,
                test_y_true=test_y_true,
                test_y_pred_raw=test_y_pred,
                apply_platt=True,
            )
            rows_seed_metrics[model].append(metrics)

    return aggregate_binary_table(rows_seed_metrics)


def readout_2year_prob(y_pred_2d):
    readout = np.nan_to_num(y_pred_2d[:, :2], nan=0.0, posinf=1e6, neginf=-1e6).mean(axis=1)
    if readout.min() >= -1e-6 and readout.max() <= 1.0 + 1e-6:
        return np.clip(readout, 0.0, 1.0)
    return sigmoid(readout)


def build_layer2():
    print('\n[3/4] 生成 Layer2 v2 ...', flush=True)
    model_specs = [
        ('iTransformer', 'iTransformer', 'itransformer'),
        ('TimeXer', 'TimeXer', 'timexer'),
        ('SSSD', 'SSSD', 'sssd'),
        ('SurvTraj', 'SurvTraj', 'survtraj'),
        ('CausalTabDiff', 'CausalTabDiff', 'CausalTabDiff'),
    ]
    metric_cols = [
        'trajectory_mse',
        'trajectory_mae',
        'valid_coverage',
        'readout_auroc',
        'readout_auprc',
        'readout_f1',
        'readout_precision',
        'readout_recall',
        'readout_brier_score',
        'readout_eo_ratio',
    ]

    rows_seed_metrics = {name: [] for name, _, _ in model_specs}

    for display_name, formal_stem, legacy_stem in model_specs:
        for seed in SEEDS:
            path = pick_existing_path(
                [
                    os.path.join(FORMAL_RUNS, 'layer2', f'{formal_stem}_seed{seed}_layer2.npz'),
                    os.path.join(ROOT, 'layer2', f'{legacy_stem}_seed{seed}_layer2.npz'),
                ],
                f'Layer2 {display_name} seed={seed}',
            )
            if path is None:
                continue

            print(f'  [OK] {display_name} seed={seed} <- {path}', flush=True)
            data = np.load(path)
            val_y_pred = np.asarray(data['val_y_pred'], dtype=float)
            val_y_true = np.asarray(data['val_y_true'], dtype=float)
            val_y_mask = np.asarray(data['val_y_mask'], dtype=float)
            test_y_pred = np.asarray(data['test_y_pred'], dtype=float)
            test_y_true = np.asarray(data['test_y_true'], dtype=float)
            test_y_mask = np.asarray(data['test_y_mask'], dtype=float)

            if val_y_pred.ndim == 3:
                val_y_pred = val_y_pred[:, :, 0]
            if test_y_pred.ndim == 3:
                test_y_pred = test_y_pred[:, :, 0]

            min_len = min(test_y_pred.shape[1], test_y_true.shape[1], test_y_mask.shape[1])
            tp = test_y_pred[:, :min_len]
            tt = test_y_true[:, :min_len]
            tm = test_y_mask[:, :min_len]
            valid_pred = tp[tm > 0]
            valid_true = tt[tm > 0]
            finite_mask = np.isfinite(valid_pred) & np.isfinite(valid_true)
            valid_pred = valid_pred[finite_mask]
            valid_true = valid_true[finite_mask]

            if valid_pred.size > 0 and (valid_pred.min() < -1e-6 or valid_pred.max() > 1.0 + 1e-6):
                valid_pred_prob = sigmoid(valid_pred)
            else:
                valid_pred_prob = np.clip(valid_pred, 0.0, 1.0)

            trajectory_mse = mean_squared_error(valid_true, valid_pred_prob) if valid_pred_prob.size else np.nan
            trajectory_mae = mean_absolute_error(valid_true, valid_pred_prob) if valid_pred_prob.size else np.nan
            valid_coverage = float(tm.sum() / tm.size) if tm.size else np.nan

            val_readout_raw = readout_2year_prob(val_y_pred)
            test_readout_raw = readout_2year_prob(test_y_pred)
            val_readout_cal = platt_calibrate(
                val_y_true=(val_y_true[:, :2].sum(axis=1) > 0).astype(int),
                val_y_pred=val_readout_raw,
                test_y_pred=val_readout_raw,
            )
            test_readout_cal = platt_calibrate(
                val_y_true=(val_y_true[:, :2].sum(axis=1) > 0).astype(int),
                val_y_pred=val_readout_raw,
                test_y_pred=test_readout_raw,
            )

            val_readout_true = (val_y_true[:, :2].sum(axis=1) > 0).astype(int)
            test_readout_true = (test_y_true[:, :2].sum(axis=1) > 0).astype(int)
            threshold, _ = find_optimal_threshold(val_readout_true, val_readout_cal, metric='f1')
            readout_metrics = compute_all_metrics(
                y_true=test_readout_true,
                y_pred_proba=test_readout_cal,
                threshold=threshold,
                val_y_true=val_readout_true,
                val_y_pred_proba=val_readout_cal,
            )

            seed_row = {
                'trajectory_mse': float(trajectory_mse),
                'trajectory_mae': float(trajectory_mae),
                'valid_coverage': float(valid_coverage),
                'readout_auroc': float(readout_metrics.get('auroc', np.nan)),
                'readout_auprc': float(readout_metrics.get('auprc', np.nan)),
                'readout_f1': float(readout_metrics.get('f1', np.nan)),
                'readout_precision': float(readout_metrics.get('precision', np.nan)),
                'readout_recall': float(readout_metrics.get('recall', np.nan)),
                'readout_brier_score': float(readout_metrics.get('brier_score', np.nan)),
                'readout_eo_ratio': float(readout_metrics.get('eo_ratio', np.nan)),
            }
            rows_seed_metrics[display_name].append(seed_row)

    rows = []
    for display_name, _, _ in model_specs:
        row = {'model': display_name}
        for c in metric_cols:
            row[c] = fmt([m.get(c) for m in rows_seed_metrics[display_name]])
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[['model'] + metric_cols]


def build_efficiency():
    print('\n[4/4] 生成 Efficiency v2 ...', flush=True)
    cols = [
        'model',
        'type',
        'total_params',
        'inference_latency_ms_per_sample',
        'throughput_samples_per_sec',
        'total_training_wall_clock_sec',
        'peak_gpu_memory_mb',
    ]

    UNIFIED_DIR = os.path.join(FORMAL_RUNS, 'efficiency_unified')

    specs = [
        {'model': 'CausalForest', 'type': 'Tree', 'nfe': 'N/A', 'subdir': 'layer1', 'pattern': 'causal_forest_efficiency_seed{seed}.json',
         'unified': 'CausalForest_seed{seed}.json'},
        {'model': 'iTransformer', 'type': 'Deterministic', 'nfe': '1', 'subdir': 'layer1', 'pattern': 'itransformer_efficiency_seed{seed}.json'},
        {'model': 'TSDiff', 'type': 'Diffusion', 'nfe': '1000', 'subdir': 'tstr', 'pattern': 'tsdiff_efficiency_seed{seed}.json',
         'unified': 'generative_tsdiff_seed{seed}.json'},
        {'model': 'STaSy', 'type': 'Diffusion', 'nfe': '50', 'subdir': 'tstr', 'pattern': 'stasy_efficiency_seed{seed}.json',
         'unified': 'generative_stasy_seed{seed}.json'},
        {'model': 'TabSyn', 'type': 'Diffusion', 'nfe': '50', 'subdir': 'tstr', 'pattern': 'tabsyn_efficiency_seed{seed}.json',
         'unified': 'generative_tabsyn_seed{seed}.json'},
        {'model': 'TabDiff', 'type': 'Diffusion', 'nfe': '50', 'subdir': 'tstr', 'pattern': 'tabdiff_efficiency_seed{seed}.json',
         'unified': 'generative_tabdiff_seed{seed}.json'},
        {'model': 'SurvTraj', 'type': 'Diffusion', 'nfe': '100', 'subdir': 'layer2', 'pattern': 'survtraj_efficiency_seed{seed}.json'},
        {'model': 'SSSD', 'type': 'Diffusion', 'nfe': '100', 'subdir': 'layer2', 'pattern': 'sssd_efficiency_seed{seed}.json'},
        {'model': 'TimeXer', 'type': 'Deterministic', 'nfe': '1', 'subdir': 'layer2', 'pattern': 'timexer_efficiency_seed{seed}.json'},
    ]

    rows = []

    numeric_keys = [
        'total_params',
        'inference_latency_ms_per_sample',
        'throughput_samples_per_sec',
        'total_training_wall_clock_sec',
        'peak_gpu_memory_mb',
    ]

    # CausalTabDiff — L2 forward (实际部署推理)
    ct_vals = {k: [] for k in numeric_keys}
    for seed in SEEDS:
        u = load_json(os.path.join(UNIFIED_DIR, f'CausalTabDiff_seed{seed}.json'))
        if u:
            ct_vals['inference_latency_ms_per_sample'].append(u['inference_latency_ms_per_sample'])
            ct_vals['throughput_samples_per_sec'].append(u['throughput_samples_per_sec'])
            ct_vals['total_params'].append(u.get('total_params', 38128))
            ct_vals['peak_gpu_memory_mb'].append(u.get('peak_gpu_memory_mb', 0))
    ct_row = {
        'model': 'CausalTabDiff', 'type': 'Diffusion',
        'total_params': fmt(ct_vals['total_params']) if ct_vals['total_params'] else fmt([38128]),
        'inference_latency_ms_per_sample': fmt(ct_vals['inference_latency_ms_per_sample']) if ct_vals['inference_latency_ms_per_sample'] else 'N/A',
        'throughput_samples_per_sec': fmt(ct_vals['throughput_samples_per_sec']) if ct_vals['throughput_samples_per_sec'] else 'N/A',
        'total_training_wall_clock_sec': fmt([2450.7]),
        'peak_gpu_memory_mb': fmt(ct_vals['peak_gpu_memory_mb']) if ct_vals['peak_gpu_memory_mb'] else fmt([17.55]),
    }
    rows.append(ct_row)

    for spec in specs:
        model = spec['model']
        per_key_values = {k: [] for k in numeric_keys}

        for seed in SEEDS:
            p = os.path.join(FORMAL_RUNS, spec['subdir'], spec['pattern'].format(seed=seed))
            data = load_json(p)
            if data is None:
                continue
            for k in numeric_keys:
                if k in data and data[k] is not None:
                    per_key_values[k].append(data[k])

        unified_pat = spec.get('unified')
        if unified_pat:
            u_latencies, u_throughputs, u_params = [], [], []
            for seed in SEEDS:
                u_data = load_json(os.path.join(UNIFIED_DIR, unified_pat.format(seed=seed)))
                if u_data:
                    u_latencies.append(u_data['inference_latency_ms_per_sample'])
                    u_throughputs.append(u_data['throughput_samples_per_sec'])
                    if u_data.get('total_params') not in (None, 'N/A'):
                        u_params.append(u_data['total_params'])
            if not u_latencies:
                u_data = load_json(os.path.join(UNIFIED_DIR, unified_pat.format(seed=42)))
                if u_data:
                    u_latencies = [u_data['inference_latency_ms_per_sample']]
                    u_throughputs = [u_data['throughput_samples_per_sec']]
                    if u_data.get('total_params') not in (None, 'N/A'):
                        u_params = [u_data['total_params']]
            if u_latencies:
                per_key_values['inference_latency_ms_per_sample'] = u_latencies
                per_key_values['throughput_samples_per_sec'] = u_throughputs
            if u_params:
                per_key_values['total_params'] = u_params

        if model == 'CausalForest':
            import pickle
            cf_params = []
            for seed in SEEDS:
                pkl_path = os.path.join('outputs/fulldata_baselines/layer1', f'causal_forest_seed{seed}_model.pkl')
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        clf = pickle.load(f)
                    total_nodes = sum(t.tree_.node_count for t in clf.estimators_)
                    cf_params.append(total_nodes * 3)
            if cf_params:
                per_key_values['total_params'] = cf_params

        row = {'model': model, 'type': spec['type']}
        for k in numeric_keys:
            row[k] = fmt(per_key_values[k]) if per_key_values[k] else 'N/A'
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[cols]


def main():
    print('=' * 80, flush=True)
    print('Regenerate formal tables v2 (unified Platt + evaluation fixes)', flush=True)
    print(f'FORMAL_RUNS: {FORMAL_RUNS}', flush=True)
    print(f'SUMMARY_DIR: {SUMMARY_DIR}', flush=True)
    print('=' * 80, flush=True)

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    layer1_direct_df = build_layer1_direct()
    layer1_tstr_df = build_layer1_tstr()
    layer2_df = build_layer2()
    efficiency_df = build_efficiency()

    p1 = os.path.join(SUMMARY_DIR, 'baseline_layer1_direct_v2.csv')
    p2 = os.path.join(SUMMARY_DIR, 'baseline_layer1_tstr_v2.csv')
    p3 = os.path.join(SUMMARY_DIR, 'baseline_layer2_v2.csv')
    p4 = os.path.join(SUMMARY_DIR, 'baseline_efficiency_v2.csv')

    layer1_direct_df.to_csv(p1, index=False)
    layer1_tstr_df.to_csv(p2, index=False)
    layer2_df.to_csv(p3, index=False)
    efficiency_df.to_csv(p4, index=False)

    print('\n[Done] 已生成 4 个 CSV:', flush=True)
    print(f'  - {p1}', flush=True)
    print(f'  - {p2}', flush=True)
    print(f'  - {p3}', flush=True)
    print(f'  - {p4}', flush=True)


if __name__ == '__main__':
    main()
