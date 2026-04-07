"""
聚合 submission_v3 的消融实验和参数讨论结果。
生成 CSV/MD 汇总文件。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_ranking_metrics,
    find_optimal_threshold,
    compute_threshold_metrics,
)

SUBMISSION_ROOT = PROJECT_ROOT / 'outputs' / 'submission_v3'
ABLATION_DIR = SUBMISSION_ROOT / 'ablation_layer2'
PARAM_DIR = SUBMISSION_ROOT / 'param_discussion_layer2'


def load_run_metrics(npz_path, eff_path):
    d = np.load(npz_path)
    val_y_true = d['val_y_true']
    val_y_pred = d['val_y_pred']
    test_y_true = d['test_y_true']
    test_y_pred = d['test_y_pred']

    val_m = compute_ranking_metrics(val_y_true, val_y_pred)
    test_m = compute_ranking_metrics(test_y_true, test_y_pred)

    thr, _ = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
    test_bin = (test_y_pred >= thr).astype(int)
    tm = compute_threshold_metrics(test_y_true, test_bin)

    eff = {}
    if eff_path.exists():
        with open(eff_path) as f:
            eff = json.load(f)

    return {
        'val_auprc': float(val_m['auprc']),
        'val_auroc': float(val_m['auroc']),
        'test_auprc': float(test_m['auprc']),
        'test_auroc': float(test_m['auroc']),
        'test_f1': float(tm['f1']),
        'test_precision': float(tm['precision']),
        'test_recall': float(tm['recall']),
        'test_specificity': float(tm['specificity']),
        'test_balanced_accuracy': float(tm['balanced_accuracy']),
        'test_mcc': float(tm['mcc']),
        'optimal_threshold': float(thr),
        'total_params': eff.get('total_params', eff.get('num_parameters_total')),
        'peak_gpu_memory_mb': eff.get('peak_gpu_memory_mb', eff.get('gpu_memory_mb_peak')),
        'train_wall_clock_sec': eff.get('total_training_wall_clock_sec', eff.get('train_wall_clock_sec')),
        'inference_latency_ms': eff.get('inference_latency_ms_per_sample'),
    }


def aggregate_ablation():
    print('=== Aggregating Ablation Results ===', flush=True)
    variants = ['Full_Model', 'wo_Dual_Attention', 'wo_Oversampling', 'wo_Discriminator']
    rows = []
    missing = []

    for v in variants:
        npz = ABLATION_DIR / f'{v}_seed42_official.npz'
        eff = ABLATION_DIR / f'{v}_seed42_official_efficiency.json'
        if not npz.exists():
            missing.append(v)
            continue
        m = load_run_metrics(npz, eff)
        m['variant'] = v
        m['seed'] = 42
        rows.append(m)
        print(f'  {v}: AUPRC={m["test_auprc"]:.4f} AUROC={m["test_auroc"]:.4f} F1={m["test_f1"]:.4f} MCC={m["test_mcc"]:.4f}', flush=True)

    if missing:
        print(f'  ⚠️ Missing: {missing}', flush=True)

    df = pd.DataFrame(rows)
    cols_order = ['variant', 'seed', 'test_auprc', 'test_auroc', 'test_f1', 'test_precision',
                  'test_recall', 'test_specificity', 'test_balanced_accuracy', 'test_mcc',
                  'val_auprc', 'val_auroc', 'optimal_threshold',
                  'total_params', 'peak_gpu_memory_mb', 'train_wall_clock_sec', 'inference_latency_ms']
    df = df[[c for c in cols_order if c in df.columns]]

    detail_csv = SUBMISSION_ROOT / 'ablation_layer2_detail.csv'
    summary_csv = SUBMISSION_ROOT / 'ablation_layer2_summary.csv'
    summary_md = SUBMISSION_ROOT / 'ablation_layer2_summary.md'

    df.to_csv(detail_csv, index=False)
    df.to_csv(summary_csv, index=False)
    print(f'  Saved: {detail_csv}', flush=True)

    full_model_auprc = df.loc[df['variant'] == 'Full_Model', 'test_auprc'].values[0] if 'Full_Model' in df['variant'].values else None

    with open(summary_md, 'w') as f:
        f.write('# Layer2 消融实验汇总\n\n')
        f.write(f'Seed: 42 | GPU: RTX 3090 | Protocol: Two-stage (pretrain 50 + finetune 200)\n\n')
        f.write('## 结果表\n\n')
        f.write('| Variant | Test AUPRC | Test AUROC | Test F1 | Test MCC | Params | GPU Mem (MB) | Train Time (min) |\n')
        f.write('|---------|-----------|-----------|---------|----------|--------|-------------|------------------|\n')
        for _, row in df.iterrows():
            train_min = f'{row["train_wall_clock_sec"]/60:.1f}' if pd.notna(row.get('train_wall_clock_sec')) else 'N/A'
            gpu_mem = f'{row["peak_gpu_memory_mb"]:.1f}' if pd.notna(row.get('peak_gpu_memory_mb')) else 'N/A'
            f.write(f'| {row["variant"]} | {row["test_auprc"]:.4f} | {row["test_auroc"]:.4f} | '
                    f'{row["test_f1"]:.4f} | {row["test_mcc"]:.4f} | {row.get("total_params", "N/A")} | '
                    f'{gpu_mem} | {train_min} |\n')

        f.write('\n## 分析\n\n')
        if full_model_auprc is not None:
            most_hurt = df.loc[df['variant'] != 'Full_Model'].sort_values('test_auprc').iloc[0]
            drop = full_model_auprc - most_hurt['test_auprc']
            f.write(f'- Full Model Test AUPRC: {full_model_auprc:.4f}\n')
            f.write(f'- 最大 AUPRC 下降: {most_hurt["variant"]} (AUPRC={most_hurt["test_auprc"]:.4f}, '
                    f'下降 {drop:.4f}, 相对下降 {drop/full_model_auprc*100:.1f}%)\n')
            for _, row in df.iterrows():
                if row['variant'] == 'Full_Model':
                    continue
                d = full_model_auprc - row['test_auprc']
                f.write(f'- {row["variant"]}: AUPRC 变化 {d:+.4f} ({d/full_model_auprc*100:+.1f}%)\n')

    print(f'  Saved: {summary_md}', flush=True)
    return df


def aggregate_param_discussion():
    print('\n=== Aggregating Parameter Discussion Results ===', flush=True)
    params_config = {
        'oversample_ratio': [1.0, 2.5, 5.0, 7.5, 10.0],
        'disc_weight': [0.0, 0.01, 0.05, 0.1, 0.2],
        'focal_gamma': [0.0, 1.0, 2.0, 3.0, 4.0],
    }
    official_defaults = {'oversample_ratio': 5.0, 'disc_weight': 0.5, 'focal_gamma': 2.0}

    rows = []
    missing = []

    for param_name, values in params_config.items():
        for val in values:
            val_str = str(val)
            npz = PARAM_DIR / param_name / f'{param_name}_{val_str}_seed42_official.npz'
            eff = PARAM_DIR / param_name / f'{param_name}_{val_str}_seed42_official_efficiency.json'

            if not npz.exists():
                npz_raw = PARAM_DIR / param_name / f'{param_name}_{val_str}_seed42.npz'
                eff_raw = PARAM_DIR / param_name / f'{param_name}_{val_str}_seed42_efficiency.json'
                if npz_raw.exists():
                    npz = npz_raw
                    eff = eff_raw
                else:
                    missing.append(f'{param_name}={val_str}')
                    continue

            m = load_run_metrics(npz, eff)
            m['param_name'] = param_name
            m['param_value'] = val
            m['seed'] = 42
            m['is_default'] = (val == official_defaults.get(param_name))
            rows.append(m)
            marker = ' ★' if m['is_default'] else ''
            print(f'  {param_name}={val_str}: AUPRC={m["test_auprc"]:.4f} AUROC={m["test_auroc"]:.4f} F1={m["test_f1"]:.4f}{marker}', flush=True)

    if missing:
        print(f'  ⚠️ Missing: {missing}', flush=True)

    df = pd.DataFrame(rows)
    cols_order = ['param_name', 'param_value', 'is_default', 'seed',
                  'test_auprc', 'test_auroc', 'test_f1', 'test_precision',
                  'test_recall', 'test_specificity', 'test_balanced_accuracy', 'test_mcc',
                  'val_auprc', 'val_auroc', 'optimal_threshold',
                  'total_params', 'peak_gpu_memory_mb', 'train_wall_clock_sec', 'inference_latency_ms']
    df = df[[c for c in cols_order if c in df.columns]]

    detail_csv = SUBMISSION_ROOT / 'param_discussion_layer2_detail.csv'
    summary_csv = SUBMISSION_ROOT / 'param_discussion_layer2_summary.csv'
    summary_md = SUBMISSION_ROOT / 'param_discussion_layer2_summary.md'

    df.to_csv(detail_csv, index=False)
    df.to_csv(summary_csv, index=False)
    print(f'  Saved: {detail_csv}', flush=True)

    with open(summary_md, 'w') as f:
        f.write('# Layer2 参数讨论汇总\n\n')
        f.write(f'Seed: 42 | GPU: RTX 3090 | Protocol: Two-stage (pretrain 50 + finetune 200)\n')
        f.write(f'One-factor-at-a-time sweep, other params fixed at official defaults.\n\n')

        for param_name in params_config:
            pdf = df[df['param_name'] == param_name].sort_values('param_value')
            if pdf.empty:
                continue

            default_val = official_defaults.get(param_name)
            f.write(f'## {param_name} (default={default_val})\n\n')
            f.write(f'| Value | Test AUPRC | Test AUROC | Test F1 | Test MCC | Note |\n')
            f.write(f'|-------|-----------|-----------|---------|----------|------|\n')

            best_auprc_row = pdf.loc[pdf['test_auprc'].idxmax()]
            for _, row in pdf.iterrows():
                note = ''
                if row.get('is_default'):
                    note = '★ default'
                if row['param_value'] == best_auprc_row['param_value']:
                    note += ' ◆ best' if note else '◆ best'
                f.write(f'| {row["param_value"]} | {row["test_auprc"]:.4f} | {row["test_auroc"]:.4f} | '
                        f'{row["test_f1"]:.4f} | {row["test_mcc"]:.4f} | {note} |\n')

            auprc_range = pdf['test_auprc'].max() - pdf['test_auprc'].min()
            auprc_mean = pdf['test_auprc'].mean()
            sensitivity = auprc_range / auprc_mean * 100 if auprc_mean > 0 else 0

            f.write(f'\nAUPRC range: {pdf["test_auprc"].min():.4f} ~ {pdf["test_auprc"].max():.4f} ')
            f.write(f'(Δ={auprc_range:.4f}, sensitivity={sensitivity:.1f}%)\n')
            f.write(f'Best: {param_name}={best_auprc_row["param_value"]} (AUPRC={best_auprc_row["test_auprc"]:.4f})\n\n')

        f.write('## 总结\n\n')
        sensitivities = {}
        for param_name in params_config:
            pdf = df[df['param_name'] == param_name]
            if not pdf.empty:
                auprc_range = pdf['test_auprc'].max() - pdf['test_auprc'].min()
                auprc_mean = pdf['test_auprc'].mean()
                sensitivities[param_name] = auprc_range / auprc_mean * 100 if auprc_mean > 0 else 0

        if sensitivities:
            most_sensitive = max(sensitivities, key=sensitivities.get)
            least_sensitive = min(sensitivities, key=sensitivities.get)
            f.write(f'- 最敏感参数: {most_sensitive} (sensitivity={sensitivities[most_sensitive]:.1f}%)\n')
            f.write(f'- 最不敏感参数: {least_sensitive} (sensitivity={sensitivities[least_sensitive]:.1f}%)\n')
            for p, s in sorted(sensitivities.items(), key=lambda x: -x[1]):
                f.write(f'- {p}: sensitivity={s:.1f}%\n')

    print(f'  Saved: {summary_md}', flush=True)
    return df


def verify_completeness():
    print('\n=== Verification ===', flush=True)
    ablation_expected = ['Full_Model', 'wo_Dual_Attention', 'wo_Oversampling', 'wo_Discriminator']
    param_expected = {
        'oversample_ratio': [1.0, 2.5, 5.0, 7.5, 10.0],
        'disc_weight': [0.0, 0.01, 0.05, 0.1, 0.2],
        'focal_gamma': [0.0, 1.0, 2.0, 3.0, 4.0],
    }

    ok = 0
    fail = 0

    for v in ablation_expected:
        npz = ABLATION_DIR / f'{v}_seed42_official.npz'
        eff = ABLATION_DIR / f'{v}_seed42_official_efficiency.json'
        if npz.exists() and eff.exists():
            ok += 1
        else:
            print(f'  ❌ MISSING ablation: {v}', flush=True)
            fail += 1

    for param, values in param_expected.items():
        for val in values:
            val_str = str(val)
            npz = PARAM_DIR / param / f'{param}_{val_str}_seed42_official.npz'
            npz_raw = PARAM_DIR / param / f'{param}_{val_str}_seed42.npz'
            if npz.exists() or npz_raw.exists():
                ok += 1
            else:
                print(f'  ❌ MISSING param: {param}={val_str}', flush=True)
                fail += 1

    total = ok + fail
    print(f'\n  Total: {total}/19 | OK: {ok} | Missing: {fail}', flush=True)
    return fail == 0


if __name__ == '__main__':
    all_complete = verify_completeness()
    if not all_complete:
        print('\n⚠️ Not all runs complete. Aggregating available results only.', flush=True)

    abl_df = aggregate_ablation()
    param_df = aggregate_param_discussion()

    print('\n=== Done ===', flush=True)
    if all_complete:
        print('✅ All 19 runs verified and aggregated.', flush=True)
    else:
        print('⚠️ Partial aggregation. Re-run after all experiments complete.', flush=True)
