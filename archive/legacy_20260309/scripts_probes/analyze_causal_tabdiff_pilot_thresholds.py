import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_baselines import compute_metrics, estimate_trainable_params_m
from src.baselines.wrappers import CausalTabDiffWrapper
from src.data.data_module import NLSTDataset


SEED = 7
TRAIN_SIZE = 8192
EVAL_SIZE = 4096
BATCH_SIZE = 256
EPOCHS = 8
DIFFUSION_STEPS = 20
GUIDANCE_SCALE_WEIGHT = 0.5
OUTCOME_LOSS_WEIGHT = 1.0
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'logs', 'testing', 'causal_tabdiff_pilot_threshold_analysis.md')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_semantic_real_x(batch, device):
    x_analog = batch['x'].to(device)
    cat_raw = batch['x_cat_raw'].to(device).float()
    meta = [
        {'name': 'race', 'type': 'categorical', 'dim': 4},
        {'name': 'cigsmok', 'type': 'categorical', 'dim': 1},
        {'name': 'gender', 'type': 'categorical', 'dim': 1},
        {'name': 'age', 'type': 'continuous', 'dim': 1},
    ]
    real_x = torch.zeros((x_analog.shape[0], len(meta)), device=device)

    analog_offset = 0
    cat_idx = 0
    for i_col, col_meta in enumerate(meta):
        if col_meta['type'] == 'continuous':
            real_x[:, i_col:i_col + 1] = x_analog[:, -1, analog_offset:analog_offset + 1]
        else:
            real_x[:, i_col:i_col + 1] = cat_raw[:, -1, cat_idx:cat_idx + 1]
            cat_idx += 1
        analog_offset += col_meta['dim']
    return real_x


def rank_threshold(scores: np.ndarray, target_rate: float) -> float:
    target_rate = float(np.clip(target_rate, 0.0, 1.0))
    if scores.size == 0:
        return 1.0
    q = max(0.0, min(1.0, 1.0 - target_rate))
    return float(np.quantile(scores, q))


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    y_pred = (scores >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'positive_rate': float(y_pred.mean()),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'confusion': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def best_f1_over_thresholds(y_true: np.ndarray, scores: np.ndarray):
    unique_scores = np.unique(scores)
    if unique_scores.size > 512:
        quantiles = np.linspace(0.0, 1.0, 512)
        candidates = np.unique(np.quantile(unique_scores, quantiles))
    else:
        candidates = unique_scores

    best = None
    for threshold in candidates:
        current = metrics_at_threshold(y_true, scores, float(threshold))
        if best is None or current['f1'] > best['f1'] or (current['f1'] == best['f1'] and current['recall'] > best['recall']):
            best = current
    return best


def main() -> None:
    os.environ.setdefault('DATASET_METADATA_PATH', os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json'))
    os.environ.setdefault('ALPHA_TREATMENT_COLUMN', 'cigsmok')
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = NLSTDataset(data_dir=os.path.join(PROJECT_ROOT, 'data'), debug_mode=False)
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
    params_m = estimate_trainable_params_m(wrapper)

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

    auc = float(roc_auc_score(real_y_np, real_scores))
    pr_auc = float(average_precision_score(real_y_np, real_scores))
    real_rate = float(real_y_np.mean())
    fake_rate = float(fake_y_np.mean())

    at_default = metrics_at_threshold(real_y_np, real_scores, 0.5)
    at_real_prev = metrics_at_threshold(real_y_np, real_scores, rank_threshold(real_scores, real_rate))
    at_fake_prev = metrics_at_threshold(real_y_np, real_scores, rank_threshold(real_scores, fake_rate))
    best_oracle = best_f1_over_thresholds(real_y_np, real_scores)

    lines = []
    lines.append('# CausalTabDiff Pilot Threshold Analysis')
    lines.append('')
    lines.append(f'- Generated at: {datetime.now().isoformat()}')
    lines.append(f'- Seed: {SEED}')
    lines.append(f'- Device: {device}')
    lines.append(f'- Train size: {TRAIN_SIZE}')
    lines.append(f'- Eval size: {EVAL_SIZE}')
    lines.append(f'- Batch size: {BATCH_SIZE}')
    lines.append(f'- Epochs: {EPOCHS}')
    lines.append(f'- Diffusion steps: {DIFFUSION_STEPS}')
    lines.append(f'- Treatment source: {os.environ.get("ALPHA_TREATMENT_COLUMN")}')
    lines.append(f'- Params(M): {params_m:.4f}')
    lines.append(f'- Train seconds: {train_seconds:.2f}')
    lines.append(f'- Real Y rate: {real_rate:.4f}')
    lines.append(f'- Fake Y rate: {fake_rate:.4f}')
    lines.append(f'- XGB scale_pos_weight: {scale_pos_weight:.4f}')
    lines.append('')
    lines.append('## Base Metrics')
    lines.append('')
    for key, value in base_metrics.items():
        lines.append(f'- {key}: {value:.6f}')
    lines.append('')
    lines.append('## Score Diagnostics')
    lines.append('')
    lines.append(f'- ROC_AUC (recomputed): {auc:.6f}')
    lines.append(f'- PR_AUC: {pr_auc:.6f}')
    lines.append('')
    lines.append('## Threshold Sweeps')
    lines.append('')

    threshold_blocks = [
        ('Default threshold 0.5', at_default),
        ('Real-prevalence-matched threshold', at_real_prev),
        ('Fake-prevalence-matched threshold', at_fake_prev),
        ('Oracle best F1 on eval set', best_oracle),
    ]
    for title, block in threshold_blocks:
        lines.append(f'### {title}')
        lines.append('')
        lines.append(f'- threshold: {block["threshold"]:.6f}')
        lines.append(f'- predicted_positive_rate: {block["positive_rate"]:.4f}')
        lines.append(f'- F1: {block["f1"]:.6f}')
        lines.append(f'- precision: {block["precision"]:.6f}')
        lines.append(f'- recall: {block["recall"]:.6f}')
        lines.append(f'- confusion_matrix[[TN,FP],[FN,TP]]: {block["confusion"]}')
        lines.append('')

    lines.append('## Interpretation')
    lines.append('')
    if at_default['f1'] == 0.0 and at_real_prev['f1'] > 0.0:
        lines.append('- The zero F1 is primarily a thresholding/cutoff problem rather than a complete loss of ranking signal.')
    elif best_oracle['f1'] == 0.0:
        lines.append('- Even oracle threshold search cannot recover F1, suggesting the pilot lacks usable downstream label signal.')
    else:
        lines.append('- Threshold tuning recovers some F1, but the recovery should be treated as diagnostic only until validated on separate seeds/splits.')
    lines.append('- This analysis remains pilot-only and does not authorize full-scale training.')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Wrote threshold analysis report to {OUTPUT_PATH}')
    print(f'base_TSTR_AUC={base_metrics["TSTR_AUC"]:.6f}')
    print(f'base_TSTR_F1={base_metrics["TSTR_F1"]:.6f}')
    print(f'roc_auc={auc:.6f}')
    print(f'pr_auc={pr_auc:.6f}')
    print(f'default_f1={at_default["f1"]:.6f}')
    print(f'real_prev_f1={at_real_prev["f1"]:.6f}')
    print(f'fake_prev_f1={at_fake_prev["f1"]:.6f}')
    print(f'best_oracle_f1={best_oracle["f1"]:.6f}')


if __name__ == '__main__':
    main()
