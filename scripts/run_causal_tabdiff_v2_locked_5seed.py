import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_baselines import compute_metrics, estimate_trainable_params_m
from src.baselines.wrappers import CausalTabDiffWrapper
from src.data.data_module import NLSTDataset


SEEDS = [7, 42, 123, 1024, 2024]
TRAIN_SIZE = 8192
EVAL_SIZE = 4096
BATCH_SIZE = 256
EPOCHS = 10
DIFFUSION_STEPS = 35
SAMPLE_MODEL_SCORE_WEIGHT = 1.0
SAMPLE_GUIDANCE_SCALE = 1.5
OUTCOME_LOSS_WEIGHT = 0.35
RISK_SMOOTHNESS_WEIGHT = 0.10
CF_CONSISTENCY_WEIGHT = 0.05
USE_TRAJECTORY_RISK_HEAD = True
SAMPLE_USE_TRAJECTORY_RISK = True
OUTPUT_PATH = os.environ.get(
    'CAUSAL_TABDIFF_V2_LOCKED_5SEED_OUTPUT_PATH',
    os.path.join(PROJECT_ROOT, 'logs', 'testing', 'causal_tabdiff_v2_locked_5seed.md'),
)
BASELINE_REPORT_PATH = os.environ.get(
    'CAUSAL_TABDIFF_V2_BASELINE_REPORT_PATH',
    os.path.join(PROJECT_ROOT, 'markdown_report.md'),
)

FORMAL_GATE = {
    'TSTR_AUC': 0.58,
    'TSTR_PR_AUC': 0.04,
    'TSTR_F1': 0.05,
    'TSTR_F1_RealPrev': 0.05,
}
METRIC_DIRECTIONS = {
    'ATE_Bias': 'lower',
    'Wasserstein': 'lower',
    'CMD': 'lower',
    'TSTR_AUC': 'higher',
    'TSTR_F1': 'higher',
}

META = [
    {'name': 'race', 'type': 'categorical', 'dim': 4},
    {'name': 'cigsmok', 'type': 'categorical', 'dim': 1},
    {'name': 'gender', 'type': 'categorical', 'dim': 1},
    {'name': 'age', 'type': 'continuous', 'dim': 1},
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_semantic_real_x(batch, device):
    x_analog = batch['x'].to(device)
    cat_raw = batch['x_cat_raw'].to(device).float()
    real_x = torch.zeros((x_analog.shape[0], len(META)), device=device)

    analog_offset = 0
    cat_idx = 0
    for i_col, col_meta in enumerate(META):
        if col_meta['type'] == 'continuous':
            real_x[:, i_col:i_col + 1] = x_analog[:, -1, analog_offset:analog_offset + 1]
        else:
            real_x[:, i_col:i_col + 1] = cat_raw[:, -1, cat_idx:cat_idx + 1]
            cat_idx += 1
        analog_offset += col_meta['dim']
    return real_x


def rank_threshold(scores: np.ndarray, target_rate: float) -> float:
    q = float(np.clip(1.0 - target_rate, 0.0, 1.0))
    return float(np.quantile(scores, q))


def threshold_stats(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    y_pred = (scores >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'positive_rate': float(y_pred.mean()),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
    }


def safe_mean_std(values):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(np.mean(arr)), float(np.std(arr))


def fmt(mean_v, std_v):
    if not np.isfinite(mean_v) or not np.isfinite(std_v):
        return 'N/A'
    return f'{mean_v:.6f} ± {std_v:.6f}'


def parse_metric_cell(cell: str):
    text = cell.strip()
    if text == 'N/A':
        return float('nan')
    if '±' in text:
        text = text.split('±', 1)[0].strip()
    return float(text)


def parse_baseline_report(report_path: str):
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    table_lines = [line for line in lines if line.strip().startswith('|')]
    headers = [part.strip() for part in table_lines[0].strip('|').split('|')]
    rows = []
    for line in table_lines[2:]:
        parts = [part.strip() for part in line.strip('|').split('|')]
        if len(parts) != len(headers):
            continue
        rows.append(dict(zip(headers, parts)))
    return rows


def build_reference_frontier(report_path: str):
    rows = parse_baseline_report(report_path)
    frontier = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = [parse_metric_cell(row[metric]) for row in rows if metric in row]
        values = [v for v in values if np.isfinite(v)]
        frontier[metric] = min(values) if direction == 'lower' else max(values)
    return rows, frontier


def evaluate_seed(seed: int, dataset: NLSTDataset, device: torch.device):
    set_seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    train_idx = indices[:TRAIN_SIZE]
    eval_idx = indices[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]

    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(Subset(dataset, eval_idx.tolist()), batch_size=BATCH_SIZE, shuffle=False)

    first_batch = next(iter(train_loader))
    wrapper = CausalTabDiffWrapper(
        t_steps=first_batch['x'].shape[1],
        feature_dim=first_batch['x'].shape[2],
        diffusion_steps=DIFFUSION_STEPS,
        outcome_loss_weight=OUTCOME_LOSS_WEIGHT,
        use_trajectory_risk_head=USE_TRAJECTORY_RISK_HEAD,
        sample_use_trajectory_risk=SAMPLE_USE_TRAJECTORY_RISK,
        risk_smoothness_weight=RISK_SMOOTHNESS_WEIGHT,
        cf_consistency_weight=CF_CONSISTENCY_WEIGHT,
        sample_model_score_weight=SAMPLE_MODEL_SCORE_WEIGHT,
        sample_guidance_scale=SAMPLE_GUIDANCE_SCALE,
    )

    train_start = time.perf_counter()
    wrapper.fit(train_loader, epochs=EPOCHS, device=device, debug_mode=False)
    train_seconds = time.perf_counter() - train_start

    all_real_x, all_fake_x, all_real_y, all_fake_y, all_alpha = [], [], [], [], []
    infer_start = time.perf_counter()
    with torch.no_grad():
        for batch in eval_loader:
            alpha = batch['alpha_target'].to(device)
            fake_x, fake_y = wrapper.sample(batch_size=alpha.shape[0], alpha_target=alpha, device=device)
            real_x = build_semantic_real_x(batch, device)
            all_real_x.append(real_x)
            all_fake_x.append(fake_x)
            all_real_y.append(batch['y'].to(device))
            all_fake_y.append(fake_y)
            all_alpha.append(alpha)
    infer_seconds = time.perf_counter() - infer_start

    real_x_full = torch.cat(all_real_x, dim=0)
    fake_x_full = torch.cat(all_fake_x, dim=0)
    real_y_full = torch.cat(all_real_y, dim=0)
    fake_y_full = torch.cat(all_fake_y, dim=0)
    alpha_full = torch.cat(all_alpha, dim=0)
    base_metrics = compute_metrics(real_x_full, fake_x_full, real_y_full, fake_y_full, alpha_full)

    real_x_np = real_x_full.detach().cpu().numpy()
    fake_x_np = fake_x_full.detach().cpu().numpy()
    real_y_np = (real_y_full.detach().cpu().numpy().reshape(-1) > 0.5).astype(int)
    fake_y_np = (fake_y_full.detach().cpu().numpy().reshape(-1) > 0.5).astype(int)

    pos_count = int(np.sum(fake_y_np == 1))
    neg_count = int(np.sum(fake_y_np == 0))
    scale_pos_weight = neg_count / max(1, pos_count) if pos_count > 0 else 1.0
    clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, scale_pos_weight=scale_pos_weight)
    clf.fit(fake_x_np, fake_y_np)
    real_scores = clf.predict_proba(real_x_np)[:, 1]

    real_rate = float(real_y_np.mean())
    fake_rate = float(fake_y_np.mean())
    result = {
        'seed': seed,
        'params_m': estimate_trainable_params_m(wrapper),
        'train_seconds': float(train_seconds),
        'infer_seconds': float(infer_seconds),
        'real_y_rate': real_rate,
        'fake_y_rate': fake_rate,
        'base_metrics': base_metrics,
        'pr_auc': float(average_precision_score(real_y_np, real_scores)),
        'real_prev': threshold_stats(real_y_np, real_scores, rank_threshold(real_scores, real_rate)),
        'fake_prev': threshold_stats(real_y_np, real_scores, rank_threshold(real_scores, fake_rate)),
    }
    result['formal_gate_pass'] = all(result['base_metrics'][metric] >= threshold for metric, threshold in FORMAL_GATE.items())
    return result


def main() -> None:
    os.environ.setdefault('DATASET_METADATA_PATH', os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json'))
    os.environ.setdefault('ALPHA_TREATMENT_COLUMN', 'cigsmok')
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    baseline_rows, frontier = build_reference_frontier(BASELINE_REPORT_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NLSTDataset(data_dir=os.path.join(PROJECT_ROOT, 'data'), debug_mode=False)

    results = []
    wall_start = time.perf_counter()
    for seed in SEEDS:
        print(f'=== Running locked V2 seed {seed} ===', flush=True)
        results.append(evaluate_seed(seed, dataset, device))
    total_seconds = time.perf_counter() - wall_start

    def collect(metric_name, nested=None):
        vals = []
        for r in results:
            if nested is None:
                vals.append(r[metric_name])
            else:
                vals.append(r[metric_name][nested])
        return vals

    mean_metrics = {
        'ATE_Bias': safe_mean_std(collect('base_metrics', 'ATE_Bias'))[0],
        'Wasserstein': safe_mean_std(collect('base_metrics', 'Wasserstein'))[0],
        'CMD': safe_mean_std(collect('base_metrics', 'CMD'))[0],
        'TSTR_AUC': safe_mean_std(collect('base_metrics', 'TSTR_AUC'))[0],
        'TSTR_F1': safe_mean_std(collect('base_metrics', 'TSTR_F1'))[0],
    }
    mean_wins = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        if direction == 'lower':
            mean_wins[metric] = mean_metrics[metric] < frontier[metric]
        else:
            mean_wins[metric] = mean_metrics[metric] > frontier[metric]

    lines = []
    lines.append('# CausalTabDiff V2 Locked 5-Seed Validation')
    lines.append('')
    lines.append(f'- Generated at: {datetime.now().isoformat()}')
    lines.append(f'- Device: {device}')
    lines.append(f'- Seeds: {SEEDS}')
    lines.append(f'- Train size per seed: {TRAIN_SIZE}')
    lines.append(f'- Eval size per seed: {EVAL_SIZE}')
    lines.append(f'- Batch size: {BATCH_SIZE}')
    lines.append(f'- Epochs: {EPOCHS}')
    lines.append(f'- Diffusion steps: {DIFFUSION_STEPS}')
    lines.append(f'- Sample model score weight: {SAMPLE_MODEL_SCORE_WEIGHT}')
    lines.append(f'- Sample guidance scale: {SAMPLE_GUIDANCE_SCALE}')
    lines.append(f'- Outcome loss weight: {OUTCOME_LOSS_WEIGHT}')
    lines.append(f'- Risk smoothness weight: {RISK_SMOOTHNESS_WEIGHT}')
    lines.append(f'- Counterfactual consistency weight: {CF_CONSISTENCY_WEIGHT}')
    lines.append(f'- Treatment source: {os.environ.get("ALPHA_TREATMENT_COLUMN")}')
    lines.append(f'- Metadata path: {os.environ.get("DATASET_METADATA_PATH")}')
    lines.append(f'- Total wall time seconds: {total_seconds:.2f}')
    lines.append('')
    lines.append('## Aggregate Summary')
    lines.append('')
    for metric in ['ATE_Bias', 'Wasserstein', 'CMD', 'TSTR_AUC', 'TSTR_F1']:
        lines.append(f"- {metric}: {fmt(*safe_mean_std(collect('base_metrics', metric)))}")
    lines.append(f"- TSTR_PR_AUC: {fmt(*safe_mean_std(collect('pr_auc')))}")
    lines.append(f"- TSTR_F1_RealPrev: {fmt(*safe_mean_std(collect('real_prev', 'f1')))}")
    lines.append(f"- TSTR_F1_FakePrev: {fmt(*safe_mean_std(collect('fake_prev', 'f1')))}")
    lines.append(f"- Params(M): {fmt(*safe_mean_std(collect('params_m')))}")
    lines.append(f"- AvgInfer(ms/sample): {fmt(*safe_mean_std([1000.0 * r['infer_seconds'] / EVAL_SIZE for r in results]))}")
    lines.append(f"- Gate pass seeds: {sum(r['formal_gate_pass'] for r in results)}/{len(results)}")
    lines.append(f"- Mean hard wins vs table frontier: {sum(bool(v) for v in mean_wins.values())}/5")
    lines.append('- Mean hard metric wins: ' + ', '.join(f"{k}={'Y' if v else 'N'}" for k, v in mean_wins.items()))
    lines.append('')
    lines.append('## Current Baseline Frontier')
    lines.append('')
    for row in baseline_rows:
        lines.append(f"- {row['Model']}: ATE_Bias={row['ATE_Bias']}, Wasserstein={row['Wasserstein']}, CMD={row['CMD']}, TSTR_AUC={row['TSTR_AUC']}, TSTR_F1={row['TSTR_F1']}")
    lines.append('')
    lines.append('## Per-Seed Results')
    lines.append('')
    for r in results:
        lines.append(f"### Seed {r['seed']}")
        lines.append('')
        lines.append(f"- formal_gate_pass: {r['formal_gate_pass']}")
        lines.append(f"- train_seconds: {r['train_seconds']:.2f}")
        lines.append(f"- infer_seconds: {r['infer_seconds']:.2f}")
        lines.append(f"- params_m: {r['params_m']:.4f}")
        lines.append(f"- ATE_Bias: {r['base_metrics']['ATE_Bias']:.6f}")
        lines.append(f"- Wasserstein: {r['base_metrics']['Wasserstein']:.6f}")
        lines.append(f"- CMD: {r['base_metrics']['CMD']:.6f}")
        lines.append(f"- TSTR_AUC: {r['base_metrics']['TSTR_AUC']:.6f}")
        lines.append(f"- TSTR_F1: {r['base_metrics']['TSTR_F1']:.6f}")
        lines.append(f"- TSTR_PR_AUC: {r['pr_auc']:.6f}")
        lines.append(f"- TSTR_F1_RealPrev: {r['real_prev']['f1']:.6f}")
        lines.append(f"- TSTR_F1_FakePrev: {r['fake_prev']['f1']:.6f}")
        lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    if all(mean_wins.values()):
        lines.append('- The locked V2 configuration beats the current 5-baseline table on all hard metrics at the 5-seed mean level.')
    else:
        lines.append('- The locked V2 configuration does not yet beat the current 5-baseline table on all hard metrics at the 5-seed mean level.')
    lines.append('- This report is the deciding checkpoint for whether V2 can advance directly to ablation and parameter discussion, or whether a structural V2.1 adjustment is still required.')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Wrote locked 5-seed report to {OUTPUT_PATH}', flush=True)
    print(f"mean_hard_wins={sum(bool(v) for v in mean_wins.values())}", flush=True)
    print(f"gate_pass_seeds={sum(r['formal_gate_pass'] for r in results)}/{len(results)}", flush=True)


if __name__ == '__main__':
    main()