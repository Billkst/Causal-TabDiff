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


SEEDS = [7, 42, 2024]
TRAIN_SIZE = 8192
EVAL_SIZE = 4096
BATCH_SIZE = 256
EPOCHS = 8
DIFFUSION_STEPS = 20
GUIDANCE_SCALE_WEIGHT = 0.5
OUTCOME_LOSS_WEIGHT = 1.0
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'logs', 'testing', 'causal_tabdiff_large_controlled_pilot.md')

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


def best_f1_over_thresholds(y_true: np.ndarray, scores: np.ndarray):
    unique_scores = np.unique(scores)
    if unique_scores.size > 512:
        candidates = np.unique(np.quantile(unique_scores, np.linspace(0.0, 1.0, 512)))
    else:
        candidates = unique_scores

    best = None
    for threshold in candidates:
        current = threshold_stats(y_true, scores, float(threshold))
        if best is None or current['f1'] > best['f1'] or (current['f1'] == best['f1'] and current['recall'] > best['recall']):
            best = current
    return best


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
        sample_model_score_weight=GUIDANCE_SCALE_WEIGHT,
    )

    train_start = time.perf_counter()
    wrapper.fit(train_loader, epochs=EPOCHS, device=device, debug_mode=False)
    train_seconds = time.perf_counter() - train_start

    all_real_x, all_fake_x, all_real_y, all_fake_y, all_alpha = [], [], [], [], []
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
    clf = XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(fake_x_np, fake_y_np)
    real_scores = clf.predict_proba(real_x_np)[:, 1]

    real_rate = float(real_y_np.mean())
    fake_rate = float(fake_y_np.mean())
    result = {
        'seed': seed,
        'params_m': estimate_trainable_params_m(wrapper),
        'train_seconds': train_seconds,
        'real_y_rate': real_rate,
        'fake_y_rate': fake_rate,
        'roc_auc': float(roc_auc_score(real_y_np, real_scores)),
        'pr_auc': float(average_precision_score(real_y_np, real_scores)),
        'base_metrics': base_metrics,
        'default': threshold_stats(real_y_np, real_scores, 0.5),
        'real_prev': threshold_stats(real_y_np, real_scores, rank_threshold(real_scores, real_rate)),
        'fake_prev': threshold_stats(real_y_np, real_scores, rank_threshold(real_scores, fake_rate)),
        'oracle': best_f1_over_thresholds(real_y_np, real_scores),
    }
    return result


def main() -> None:
    os.environ.setdefault('DATASET_METADATA_PATH', os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json'))
    os.environ.setdefault('ALPHA_TREATMENT_COLUMN', 'cigsmok')
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NLSTDataset(data_dir=os.path.join(PROJECT_ROOT, 'data'), debug_mode=False)

    results = []
    wall_start = time.perf_counter()
    for seed in SEEDS:
        print(f'=== Running large controlled pilot seed {seed} ===')
        results.append(evaluate_seed(seed, dataset, device))
    total_seconds = time.perf_counter() - wall_start

    def collect(path1, path2=None):
        vals = []
        for r in results:
            v = r[path1] if path2 is None else r[path1][path2]
            vals.append(v)
        return vals

    lines = []
    lines.append('# CausalTabDiff Large Controlled Pilot')
    lines.append('')
    lines.append(f'- Generated at: {datetime.now().isoformat()}')
    lines.append(f'- Device: {device}')
    lines.append(f'- Seeds: {SEEDS}')
    lines.append(f'- Train size per seed: {TRAIN_SIZE}')
    lines.append(f'- Eval size per seed: {EVAL_SIZE}')
    lines.append(f'- Batch size: {BATCH_SIZE}')
    lines.append(f'- Epochs: {EPOCHS}')
    lines.append(f'- Diffusion steps: {DIFFUSION_STEPS}')
    lines.append(f'- Treatment source: {os.environ.get("ALPHA_TREATMENT_COLUMN")}')
    lines.append(f'- Total wall time seconds: {total_seconds:.2f}')
    lines.append('')
    lines.append('## Aggregate Summary')
    lines.append('')
    lines.append(f"- ATE_Bias: {fmt(*safe_mean_std(collect('base_metrics', 'ATE_Bias')))}")
    lines.append(f"- Wasserstein: {fmt(*safe_mean_std(collect('base_metrics', 'Wasserstein')))}")
    lines.append(f"- CMD: {fmt(*safe_mean_std(collect('base_metrics', 'CMD')))}")
    lines.append(f"- Base TSTR_AUC: {fmt(*safe_mean_std(collect('base_metrics', 'TSTR_AUC')))}")
    lines.append(f"- Base TSTR_F1 @0.5: {fmt(*safe_mean_std(collect('base_metrics', 'TSTR_F1')))}")
    lines.append(f"- PR_AUC: {fmt(*safe_mean_std(collect('pr_auc')))}")
    lines.append(f"- Real-prev F1: {fmt(*safe_mean_std(collect('real_prev', 'f1')))}")
    lines.append(f"- Fake-prev F1: {fmt(*safe_mean_std(collect('fake_prev', 'f1')))}")
    lines.append(f"- Oracle F1: {fmt(*safe_mean_std(collect('oracle', 'f1')))}")
    lines.append(f"- Seeds with base F1 > 0.05: {sum(v > 0.05 for v in collect('base_metrics', 'TSTR_F1'))}/{len(results)}")
    lines.append(f"- Seeds with real-prev F1 > 0.05: {sum(v > 0.05 for v in collect('real_prev', 'f1'))}/{len(results)}")
    lines.append('')
    lines.append('## Per-Seed Results')
    lines.append('')

    for r in results:
        lines.append(f"### Seed {r['seed']}")
        lines.append('')
        lines.append(f"- train_seconds: {r['train_seconds']:.2f}")
        lines.append(f"- params_m: {r['params_m']:.4f}")
        lines.append(f"- real_y_rate: {r['real_y_rate']:.4f}")
        lines.append(f"- fake_y_rate: {r['fake_y_rate']:.4f}")
        lines.append(f"- ATE_Bias: {r['base_metrics']['ATE_Bias']:.6f}")
        lines.append(f"- Wasserstein: {r['base_metrics']['Wasserstein']:.6f}")
        lines.append(f"- CMD: {r['base_metrics']['CMD']:.6f}")
        lines.append(f"- base_TSTR_AUC: {r['base_metrics']['TSTR_AUC']:.6f}")
        lines.append(f"- base_TSTR_F1@0.5: {r['base_metrics']['TSTR_F1']:.6f}")
        lines.append(f"- PR_AUC: {r['pr_auc']:.6f}")
        lines.append(f"- real_prev_threshold: {r['real_prev']['threshold']:.6f}")
        lines.append(f"- real_prev_F1: {r['real_prev']['f1']:.6f}")
        lines.append(f"- fake_prev_F1: {r['fake_prev']['f1']:.6f}")
        lines.append(f"- oracle_F1: {r['oracle']['f1']:.6f}")
        lines.append(f"- oracle_threshold: {r['oracle']['threshold']:.6f}")
        lines.append('')

    lines.append('## Interpretation')
    lines.append('')
    base_good = sum(v > 0.05 for v in collect('base_metrics', 'TSTR_F1'))
    real_good = sum(v > 0.05 for v in collect('real_prev', 'f1'))
    if base_good == len(results) and real_good >= 2:
        lines.append('- The larger controlled pilot shows non-trivial downstream utility across all tested seeds, but prevalence-aware gains are still not fully uniform.')
    elif base_good == 0:
        lines.append('- The larger pilot collapses at the default threshold and should not be scaled further.')
    else:
        lines.append('- The larger pilot improves confidence relative to earlier single-run results, but robustness is still incomplete and remains pilot-only evidence.')
    lines.append('- This report does not authorize full-scale training.')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Wrote large controlled pilot report to {OUTPUT_PATH}')
    print(f"base_tstr_auc_mean={safe_mean_std(collect('base_metrics', 'TSTR_AUC'))[0]:.6f}")
    print(f"base_tstr_f1_mean={safe_mean_std(collect('base_metrics', 'TSTR_F1'))[0]:.6f}")
    print(f"real_prev_f1_mean={safe_mean_std(collect('real_prev', 'f1'))[0]:.6f}")
    print(f"oracle_f1_mean={safe_mean_std(collect('oracle', 'f1'))[0]:.6f}")


if __name__ == '__main__':
    main()
