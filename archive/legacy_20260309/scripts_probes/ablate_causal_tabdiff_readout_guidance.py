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
GRID = [
    (0.25, 1.0),
    (0.50, 1.0),
    (0.75, 1.0),
    (0.25, 2.0),
    (0.50, 2.0),
    (0.75, 2.0),
    (0.25, 3.0),
    (0.50, 3.0),
    (0.75, 3.0),
]
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'logs', 'testing', 'causal_tabdiff_readout_guidance_ablation.md')

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


def evaluate_current_sampling(wrapper, eval_loader, device):
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
        sample_model_score_weight=0.5,
        sample_guidance_scale=2.0,
    )

    train_start = time.perf_counter()
    wrapper.fit(train_loader, epochs=EPOCHS, device=device, debug_mode=False)
    train_seconds = time.perf_counter() - train_start
    params_m = estimate_trainable_params_m(wrapper)

    results = []
    for model_w, guidance_scale in GRID:
        wrapper.sample_model_score_weight = float(model_w)
        wrapper.sample_guidance_scale = float(guidance_scale)
        metrics = evaluate_current_sampling(wrapper, eval_loader, device)
        gate_like = (
            metrics['TSTR_AUC'] >= 0.58
            and metrics['TSTR_PR_AUC'] >= 0.04
            and metrics['TSTR_F1_RealPrev'] >= 0.05
        )
        results.append({
            'model_w': model_w,
            'guidance_scale': guidance_scale,
            'metrics': metrics,
            'gate_like': gate_like,
        })
        print(
            f"model_w={model_w:.2f} guidance={guidance_scale:.1f} "
            f"auc={metrics['TSTR_AUC']:.6f} pr_auc={metrics['TSTR_PR_AUC']:.6f} "
            f"f1={metrics['TSTR_F1']:.6f} real_prev_f1={metrics['TSTR_F1_RealPrev']:.6f}"
        )

    results.sort(
        key=lambda r: (
            r['gate_like'],
            r['metrics']['TSTR_PR_AUC'],
            r['metrics']['TSTR_AUC'],
            r['metrics']['TSTR_F1_RealPrev'],
        ),
        reverse=True,
    )

    lines = []
    lines.append('# CausalTabDiff Readout + Guidance Ablation')
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
    lines.append(f'- Outcome loss weight: {OUTCOME_LOSS_WEIGHT}')
    lines.append(f'- Params(M): {params_m:.4f}')
    lines.append(f'- Train seconds: {train_seconds:.2f}')
    lines.append('')
    lines.append('## Ranked Configurations')
    lines.append('')

    for idx, result in enumerate(results, start=1):
        m = result['metrics']
        lines.append(f"### Rank {idx}: model_w={result['model_w']:.2f}, guidance_scale={result['guidance_scale']:.1f}")
        lines.append('')
        lines.append(f"- gate_like_pass: {result['gate_like']}")
        lines.append(f"- TSTR_AUC: {m['TSTR_AUC']:.6f}")
        lines.append(f"- TSTR_PR_AUC: {m['TSTR_PR_AUC']:.6f}")
        lines.append(f"- TSTR_F1: {m['TSTR_F1']:.6f}")
        lines.append(f"- TSTR_F1_RealPrev: {m['TSTR_F1_RealPrev']:.6f}")
        lines.append(f"- TSTR_F1_FakePrev: {m['TSTR_F1_FakePrev']:.6f}")
        lines.append(f"- ATE_Bias: {m['ATE_Bias']:.6f}")
        lines.append(f"- Wasserstein: {m['Wasserstein']:.6f}")
        lines.append(f"- CMD: {m['CMD']:.6f}")
        lines.append('')

    top = results[0]
    top_m = top['metrics']
    lines.append('## Recommendation')
    lines.append('')
    if top['gate_like']:
        lines.append(
            f"- Best controlled candidate is model_w={top['model_w']:.2f}, guidance_scale={top['guidance_scale']:.1f}; "
            f"it clears the local gate-like heuristic with TSTR_AUC={top_m['TSTR_AUC']:.6f}, "
            f"TSTR_PR_AUC={top_m['TSTR_PR_AUC']:.6f}, TSTR_F1_RealPrev={top_m['TSTR_F1_RealPrev']:.6f}."
        )
    else:
        lines.append(
            f"- No sampled configuration clears the local gate-like heuristic. Best candidate is model_w={top['model_w']:.2f}, guidance_scale={top['guidance_scale']:.1f}, "
            f"but TSTR_AUC={top_m['TSTR_AUC']:.6f}, TSTR_PR_AUC={top_m['TSTR_PR_AUC']:.6f}, TSTR_F1_RealPrev={top_m['TSTR_F1_RealPrev']:.6f} remain below the desired joint bar."
        )
    lines.append('- This is a minimal structure/sampling optimization only; it does not authorize full-scale training by itself.')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Wrote ablation report to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
