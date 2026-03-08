import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_baselines import compute_metrics, estimate_trainable_params_m
from src.baselines.wrappers import CausalTabDiffWrapper
from src.data.data_module import NLSTDataset


SEED = 7
TRAIN_SIZE = 4096
EVAL_SIZE = 2048
BATCH_SIZE = 256
EPOCHS = 6
DIFFUSION_STEPS = 20
OUTCOME_LOSS_WEIGHT = 1.0
MODEL_SCORE_WEIGHT = 0.75
GUIDANCE_SCALE = 1.0
RANK_WEIGHTS = [0.0, 0.25, 0.5, 1.0]
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'logs', 'testing', 'causal_tabdiff_ranking_loss_ablation.md')

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


def evaluate(wrapper, eval_loader, device):
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
    return compute_metrics(real_x_full, fake_x_full, real_y_full, fake_y_full, alpha_full)


def main() -> None:
    os.environ.setdefault('DATASET_METADATA_PATH', os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json'))
    os.environ.setdefault('ALPHA_TREATMENT_COLUMN', 'cigsmok')
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NLSTDataset(data_dir=os.path.join(PROJECT_ROOT, 'data'), debug_mode=False)

    set_seed(SEED)
    indices = np.random.permutation(len(dataset))
    train_idx = indices[:TRAIN_SIZE]
    eval_idx = indices[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]

    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(Subset(dataset, eval_idx.tolist()), batch_size=BATCH_SIZE, shuffle=False)

    first_batch = next(iter(train_loader))
    results = []

    for rank_weight in RANK_WEIGHTS:
        set_seed(SEED)
        wrapper = CausalTabDiffWrapper(
            t_steps=first_batch['x'].shape[1],
            feature_dim=first_batch['x'].shape[2],
            diffusion_steps=DIFFUSION_STEPS,
            outcome_loss_weight=OUTCOME_LOSS_WEIGHT,
            outcome_rank_loss_weight=rank_weight,
            sample_model_score_weight=MODEL_SCORE_WEIGHT,
            sample_guidance_scale=GUIDANCE_SCALE,
        )
        train_start = time.perf_counter()
        wrapper.fit(train_loader, epochs=EPOCHS, device=device, debug_mode=False)
        train_seconds = time.perf_counter() - train_start
        metrics = evaluate(wrapper, eval_loader, device)
        results.append({
            'rank_weight': rank_weight,
            'train_seconds': train_seconds,
            'params_m': estimate_trainable_params_m(wrapper),
            'metrics': metrics,
        })
        print(
            f"rank_w={rank_weight:.2f} auc={metrics['TSTR_AUC']:.6f} pr_auc={metrics['TSTR_PR_AUC']:.6f} "
            f"f1={metrics['TSTR_F1']:.6f} real_prev_f1={metrics['TSTR_F1_RealPrev']:.6f}"
        )

    results.sort(
        key=lambda r: (
            r['metrics']['TSTR_PR_AUC'],
            r['metrics']['TSTR_AUC'],
            r['metrics']['TSTR_F1_RealPrev'],
            r['metrics']['TSTR_F1'],
        ),
        reverse=True,
    )

    lines = []
    lines.append('# CausalTabDiff Ranking Loss Ablation')
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
    lines.append(f'- Fixed sample model score weight: {MODEL_SCORE_WEIGHT}')
    lines.append(f'- Fixed guidance scale: {GUIDANCE_SCALE}')
    lines.append('')
    lines.append('## Ranked Results')
    lines.append('')

    for idx, result in enumerate(results, start=1):
        m = result['metrics']
        lines.append(f"### Rank {idx}: rank_loss_weight={result['rank_weight']:.2f}")
        lines.append('')
        lines.append(f"- train_seconds: {result['train_seconds']:.2f}")
        lines.append(f"- params_m: {result['params_m']:.4f}")
        lines.append(f"- TSTR_AUC: {m['TSTR_AUC']:.6f}")
        lines.append(f"- TSTR_PR_AUC: {m['TSTR_PR_AUC']:.6f}")
        lines.append(f"- TSTR_F1: {m['TSTR_F1']:.6f}")
        lines.append(f"- TSTR_F1_RealPrev: {m['TSTR_F1_RealPrev']:.6f}")
        lines.append(f"- TSTR_F1_FakePrev: {m['TSTR_F1_FakePrev']:.6f}")
        lines.append(f"- ATE_Bias: {m['ATE_Bias']:.6f}")
        lines.append(f"- Wasserstein: {m['Wasserstein']:.6f}")
        lines.append(f"- CMD: {m['CMD']:.6f}")
        lines.append('')

    best = results[0]
    bm = best['metrics']
    lines.append('## Recommendation')
    lines.append('')
    lines.append(
        f"- Best ranking-loss setting is rank_loss_weight={best['rank_weight']:.2f} with "
        f"TSTR_AUC={bm['TSTR_AUC']:.6f}, TSTR_PR_AUC={bm['TSTR_PR_AUC']:.6f}, "
        f"TSTR_F1={bm['TSTR_F1']:.6f}, TSTR_F1_RealPrev={bm['TSTR_F1_RealPrev']:.6f}."
    )
    if bm['TSTR_PR_AUC'] >= 0.04 and bm['TSTR_AUC'] >= 0.58 and bm['TSTR_F1_RealPrev'] >= 0.05:
        lines.append('- This is the first minimal ranking-oriented change that clears the local heuristic and should be prioritized for the next gated rehearsal.')
    else:
        lines.append('- Even with ranking loss, the signal does not yet clear the desired joint bar; additional outcome-head redesign would still be required.')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Wrote ranking loss ablation report to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
