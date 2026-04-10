"""
聚合 submission_v3 消融 + 参数讨论结果。
读取 REGISTRY.json，验证 P0 配置，生成 CSV/MD 汇总。
"""
import json
import math
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
    val_m = compute_ranking_metrics(d['val_y_true'], d['val_y_pred'])
    test_m = compute_ranking_metrics(d['test_y_true'], d['test_y_pred'])
    thr, _ = find_optimal_threshold(d['val_y_true'], d['val_y_pred'], metric='f1')
    test_bin = (d['test_y_pred'] >= thr).astype(int)
    tm = compute_threshold_metrics(d['test_y_true'], test_bin)

    eff = {}
    if Path(eff_path).exists():
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


def verify_shared_default(registry, param_dir):
    p0 = registry['shared_default']
    eff_path = param_dir / p0['efficiency_path']

    if not eff_path.exists():
        print(f'FATAL: P0 efficiency file not found: {eff_path}', flush=True)
        sys.exit(1)

    with open(eff_path) as f:
        eff = json.load(f)

    expected = p0['config']
    errors = []
    for key, expected_val in expected.items():
        if key not in eff:
            errors.append(f'  MISSING: "{key}" not found in efficiency.json')
            continue

        actual_val = eff[key]

        if isinstance(expected_val, bool):
            if actual_val is not expected_val:
                errors.append(f'  MISMATCH: {key} expected={expected_val!r} actual={actual_val!r}')
        elif isinstance(expected_val, float):
            if not isinstance(actual_val, (int, float)):
                errors.append(f'  TYPE_ERROR: {key} expected float, got {type(actual_val).__name__}={actual_val!r}')
            elif not math.isclose(float(actual_val), expected_val, rel_tol=1e-9):
                errors.append(f'  MISMATCH: {key} expected={expected_val} actual={actual_val}')
        elif isinstance(expected_val, int):
            if not isinstance(actual_val, int) or isinstance(actual_val, bool):
                errors.append(f'  TYPE_ERROR: {key} expected int, got {type(actual_val).__name__}={actual_val!r}')
            elif actual_val != expected_val:
                errors.append(f'  MISMATCH: {key} expected={expected_val} actual={actual_val}')
        else:
            if actual_val != expected_val:
                errors.append(f'  MISMATCH: {key} expected={expected_val!r} actual={actual_val!r}')

    if errors:
        print('FATAL: Shared default P0 config verification FAILED:', flush=True)
        for e in errors:
            print(e, flush=True)
        sys.exit(1)

    print(f'  P0 config verified: fg={eff["focal_gamma"]} os={eff["oversample_ratio"]} '
          f'dw={eff["disc_weight"]} uw={eff["use_uw"]} seed={eff["seed"]}', flush=True)


def resolve_npz_path(param_dir, param_name, value, registry):
    """Resolve physical npz/eff path, using REGISTRY for shared default mapping."""
    physical_dir = param_dir / param_name
    val_str = str(value)
    npz = physical_dir / f'{param_name}_{val_str}_seed42.npz'
    eff = physical_dir / f'{param_name}_{val_str}_seed42_efficiency.json'

    if npz.exists():
        return npz, eff

    for mapping in registry['shared_default'].get('display_mappings', []):
        if mapping['param'] == param_name and math.isclose(mapping['value'], value, rel_tol=1e-9):
            p0_npz = param_dir / registry['shared_default']['physical_path']
            p0_eff = param_dir / registry['shared_default']['efficiency_path']
            if p0_npz.exists():
                return p0_npz, p0_eff

    return npz, eff


def verify_physical_assets(registry, param_dir):
    print('\n=== Verifying Physical Assets ===', flush=True)
    expected_count = registry['counts']['total_physical_assets']
    actual_npz = list(param_dir.rglob('*.npz'))
    actual_eff = list(param_dir.rglob('*_efficiency.json'))

    symlinks = list(param_dir.rglob('*'))
    symlink_count = sum(1 for p in symlinks if p.is_symlink())
    if symlink_count > 0:
        print(f'FATAL: Found {symlink_count} symlinks in {param_dir}. Not allowed.', flush=True)
        sys.exit(1)

    non_registry = [f for f in param_dir.rglob('*') if f.is_file() and f.name != 'REGISTRY.json'
                    and not f.name.endswith('.npz') and not f.name.endswith('_efficiency.json')]
    if non_registry:
        print(f'  WARNING: unexpected files: {[str(f) for f in non_registry]}', flush=True)

    if len(actual_npz) != expected_count:
        print(f'FATAL: Expected {expected_count} npz files, found {len(actual_npz)}', flush=True)
        for r in registry['runs']:
            npz_path = param_dir / r['physical_dir'] / f'{r["param"]}_{r["value"]}_seed42.npz'
            if not npz_path.exists():
                print(f'  MISSING: {npz_path}', flush=True)
        sys.exit(1)

    if len(actual_eff) != expected_count:
        print(f'FATAL: Expected {expected_count} efficiency.json files, found {len(actual_eff)}', flush=True)
        sys.exit(1)

    print(f'  Physical assets: {len(actual_npz)} npz + {len(actual_eff)} efficiency.json = {expected_count} OK', flush=True)
    print(f'  Symlinks: 0 (verified)', flush=True)


def aggregate_ablation():
    print('\n=== Aggregating Ablation Results ===', flush=True)
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


def aggregate_param_discussion(registry):
    print('\n=== Aggregating Parameter Discussion Results ===', flush=True)

    params_display = {
        'focal_gamma': [0.0, 1.0, 2.0, 3.0, 4.0],
        'oversample_ratio': [1.0, 2.5, 5.0, 7.5, 10.0],
        'disc_weight': [0.0, 0.01, 0.05, 0.1, 0.2],
    }
    official_defaults = {'oversample_ratio': 5.0, 'focal_gamma': 2.0}

    p0_npz = PARAM_DIR / registry['shared_default']['physical_path']
    p0_eff = PARAM_DIR / registry['shared_default']['efficiency_path']
    p0_metrics = load_run_metrics(p0_npz, p0_eff)

    rows = []
    missing = []

    for param_name, values in params_display.items():
        for val in values:
            npz, eff = resolve_npz_path(PARAM_DIR, param_name, val, registry)

            if not npz.exists():
                missing.append(f'{param_name}={val}')
                continue

            m = load_run_metrics(npz, eff)
            m['param_name'] = param_name
            m['param_value'] = val
            m['seed'] = 42
            m['is_default'] = (val == official_defaults.get(param_name))

            is_p0_mapped = not npz.parent.name == param_name or npz.name != f'{param_name}_{val}_seed42.npz'
            source_tag = ''
            if is_p0_mapped:
                source_tag = ' [shared default P0]'

            rows.append(m)
            marker = ' ★' if m['is_default'] else ''
            print(f'  {param_name}={val}: AUPRC={m["test_auprc"]:.4f} AUROC={m["test_auroc"]:.4f}{marker}{source_tag}', flush=True)

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
        f.write(f'Official defaults: focal_gamma=2.0, oversample_ratio=5.0, disc_weight=0.5, use_uw=True\n\n')
        f.write(f'Real training runs: {registry["counts"]["total_real_training_runs"]} | '
                f'Physical assets: {registry["counts"]["total_physical_assets"]} | '
                f'Display rows: {registry["counts"]["total_display_rows"]}\n\n')

        for param_name in ['focal_gamma', 'oversample_ratio', 'disc_weight']:
            pdf = df[df['param_name'] == param_name].sort_values('param_value')
            f.write(f'## {param_name}\n\n')

            if param_name == 'disc_weight':
                f.write(f'> Note: Official default disc_weight=0.5 (shared baseline P0): '
                        f'AUPRC={p0_metrics["test_auprc"]:.4f}, AUROC={p0_metrics["test_auroc"]:.4f}\n\n')

            f.write('| Value | Test AUPRC | Test AUROC | Test F1 | Test MCC | Note |\n')
            f.write('|-------|-----------|-----------|---------|----------|------|\n')
            for _, row in pdf.iterrows():
                note = ''
                if row.get('is_default'):
                    note = '★ default'
                f.write(f'| {row["param_value"]} | {row["test_auprc"]:.4f} | {row["test_auroc"]:.4f} | '
                        f'{row["test_f1"]:.4f} | {row["test_mcc"]:.4f} | {note} |\n')
            f.write('\n')

    print(f'  Saved: {summary_md}', flush=True)
    return df


def print_final_counts(registry):
    print('\n=== Final Counts ===', flush=True)
    print(f'  Real training runs: {registry["counts"]["total_real_training_runs"]}/14 verified', flush=True)
    print(f'  Physical assets: {registry["counts"]["total_physical_assets"]} npz + '
          f'{registry["counts"]["total_physical_assets"]} efficiency.json', flush=True)
    print(f'  Display rows: {registry["counts"]["total_display_rows"]} '
          f'(fg={registry["counts"]["display_per_param"]["focal_gamma"]} + '
          f'os={registry["counts"]["display_per_param"]["oversample_ratio"]} + '
          f'dw={registry["counts"]["display_per_param"]["disc_weight"]})', flush=True)


if __name__ == '__main__':
    registry_path = PARAM_DIR / 'REGISTRY.json'
    if not registry_path.exists():
        print(f'FATAL: REGISTRY.json not found at {registry_path}', flush=True)
        sys.exit(1)

    with open(registry_path) as f:
        registry = json.load(f)

    print('=== P0 Config Verification ===', flush=True)
    verify_shared_default(registry, PARAM_DIR)

    verify_physical_assets(registry, PARAM_DIR)

    abl_df = aggregate_ablation()
    param_df = aggregate_param_discussion(registry)

    print_final_counts(registry)

    print('\n=== Done ===', flush=True)
