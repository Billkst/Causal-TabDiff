#!/usr/bin/env python -u
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

OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'submission_v3_5seeds' / 'param_discussion_layer2'

CONFIGS = {
    'focal_gamma': ['0.0', '1.0', '2.0', '3.0', '4.0'],
    'oversample_ratio': ['1.0', '2.5', '7.5', '10.0'],
    'disc_weight': ['0.0', '0.01', '0.05', '0.1', '0.2'],
}
SEEDS = [42, 0, 1, 2, 3]

METRIC_KEYS = [
    'test_auprc', 'test_auroc', 'test_f1', 'test_precision',
    'test_recall', 'test_specificity', 'test_balanced_accuracy', 'test_mcc',
    'val_auprc', 'val_auroc', 'best_epoch', 'wall_clock_sec',
]


def load_run(param, val, seed):
    tag = f'{param}_{val}_seed{seed}'
    npz_path = OUTPUT_ROOT / param / f'{tag}.npz'
    eff_path = OUTPUT_ROOT / param / f'{tag}_efficiency.json'

    d = np.load(npz_path)
    test_m = compute_ranking_metrics(d['test_y_true'], d['test_y_pred'])
    val_m = compute_ranking_metrics(d['val_y_true'], d['val_y_pred'])
    thr, _ = find_optimal_threshold(d['val_y_true'], d['val_y_pred'], metric='f1')
    test_bin = (d['test_y_pred'] >= thr).astype(int)
    tm = compute_threshold_metrics(d['test_y_true'], test_bin)

    wall = 0.0
    if eff_path.exists():
        with open(eff_path) as f:
            eff = json.load(f)
        wall = eff.get('total_training_wall_clock_sec', eff.get('train_wall_clock_sec', 0))

    return {
        'param': param,
        'value': val,
        'seed': seed,
        'test_auprc': float(test_m['auprc']),
        'test_auroc': float(test_m['auroc']),
        'test_f1': float(tm['f1']),
        'test_precision': float(tm['precision']),
        'test_recall': float(tm['recall']),
        'test_specificity': float(tm['specificity']),
        'test_balanced_accuracy': float(tm['balanced_accuracy']),
        'test_mcc': float(tm['mcc']),
        'val_auprc': float(val_m['auprc']),
        'val_auroc': float(val_m['auroc']),
        'best_epoch': int(d['best_epoch']) if 'best_epoch' in d else -1,
        'wall_clock_sec': float(wall),
    }


def aggregate():
    all_runs = []
    for param, values in CONFIGS.items():
        for val in values:
            for seed in SEEDS:
                try:
                    run = load_run(param, val, seed)
                    all_runs.append(run)
                except Exception as e:
                    print(f'ERROR loading {param}_{val}_seed{seed}: {e}', flush=True)
                    return

    df = pd.DataFrame(all_runs)

    out_dir = OUTPUT_ROOT
    df.to_csv(out_dir / 'all_runs_raw.csv', index=False)
    print(f'Saved raw: {out_dir / "all_runs_raw.csv"} ({len(df)} rows)', flush=True)

    agg_metrics = ['test_auprc', 'test_auroc', 'test_f1', 'test_mcc',
                   'test_precision', 'test_recall', 'test_balanced_accuracy',
                   'val_auprc', 'val_auroc', 'best_epoch', 'wall_clock_sec']

    rows = []
    for param, values in CONFIGS.items():
        for val in values:
            subset = df[(df['param'] == param) & (df['value'] == val)]
            if len(subset) != len(SEEDS):
                print(f'WARNING: {param}={val} has {len(subset)}/{len(SEEDS)} seeds', flush=True)
            row = {'param': param, 'value': val, 'n_seeds': len(subset)}
            for m in agg_metrics:
                arr = subset[m].values
                row[f'{m}_mean'] = float(np.mean(arr))
                row[f'{m}_std'] = float(np.std(arr))
            rows.append(row)

    agg_df = pd.DataFrame(rows)
    agg_df.to_csv(out_dir / 'aggregated_5seeds.csv', index=False)
    print(f'Saved aggregated: {out_dir / "aggregated_5seeds.csv"} ({len(agg_df)} rows)', flush=True)

    print(f'\n{"="*90}', flush=True)
    print(f'5-SEED AGGREGATION RESULTS (mean ± std)', flush=True)
    print(f'{"="*90}', flush=True)

    for param in CONFIGS:
        param_rows = agg_df[agg_df['param'] == param]
        print(f'\n--- {param} ---', flush=True)
        print(f'{"value":>10} | {"AUPRC":>18} | {"AUROC":>18} | {"F1":>18} | {"MCC":>18}', flush=True)
        print(f'{"-"*90}', flush=True)
        for _, r in param_rows.iterrows():
            print(
                f'{r["value"]:>10} | '
                f'{r["test_auprc_mean"]:.4f} ± {r["test_auprc_std"]:.4f} | '
                f'{r["test_auroc_mean"]:.4f} ± {r["test_auroc_std"]:.4f} | '
                f'{r["test_f1_mean"]:.4f} ± {r["test_f1_std"]:.4f} | '
                f'{r["test_mcc_mean"]:.4f} ± {r["test_mcc_std"]:.4f}',
                flush=True,
            )

    summary = {
        'protocol': {
            'batch_size': 94739,
            'num_workers': 0,
            'seeds': SEEDS,
            'n_configs': sum(len(v) for v in CONFIGS.values()),
            'n_total_runs': len(df),
        },
        'results': {},
    }
    for _, r in agg_df.iterrows():
        key = f'{r["param"]}_{r["value"]}'
        summary['results'][key] = {
            'test_auprc': f'{r["test_auprc_mean"]:.4f} ± {r["test_auprc_std"]:.4f}',
            'test_auroc': f'{r["test_auroc_mean"]:.4f} ± {r["test_auroc_std"]:.4f}',
            'test_f1': f'{r["test_f1_mean"]:.4f} ± {r["test_f1_std"]:.4f}',
            'test_mcc': f'{r["test_mcc_mean"]:.4f} ± {r["test_mcc_std"]:.4f}',
        }

    with open(out_dir / 'summary_5seeds.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary: {out_dir / "summary_5seeds.json"}', flush=True)

    print(f'\n{"="*90}', flush=True)
    print(f'AGGREGATION COMPLETE: {len(df)} runs → {len(agg_df)} configs', flush=True)
    print(f'{"="*90}', flush=True)


if __name__ == '__main__':
    aggregate()
