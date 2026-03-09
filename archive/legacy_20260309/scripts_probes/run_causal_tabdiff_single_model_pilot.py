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
TRAIN_SIZE = 8192
EVAL_SIZE = 4096
BATCH_SIZE = 256
EPOCHS = 8
DIFFUSION_STEPS = 20
GUIDANCE_SCALE_WEIGHT = 0.5
OUTCOME_LOSS_WEIGHT = 1.0
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'logs', 'testing', 'causal_tabdiff_single_model_pilot.md')


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

    metrics = compute_metrics(real_x_full, fake_x_full, real_y_full, fake_y_full, alpha_full)
    real_y_rate = float((real_y_full > 0.5).float().mean().item())
    fake_y_rate = float((fake_y_full > 0.5).float().mean().item())

    lines = []
    lines.append('# CausalTabDiff Single-Model Pilot')
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
    lines.append(f'- Sample model score weight: {GUIDANCE_SCALE_WEIGHT}')
    lines.append(f'- Params(M): {params_m:.4f}')
    lines.append(f'- Train seconds: {train_seconds:.2f}')
    lines.append(f'- Inference seconds: {infer_seconds:.2f}')
    lines.append(f'- Real Y rate: {real_y_rate:.4f}')
    lines.append(f'- Fake Y rate: {fake_y_rate:.4f}')
    lines.append('')
    lines.append('## Metrics')
    lines.append('')
    for key, value in metrics.items():
        lines.append(f'- {key}: {value:.6f}')
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    lines.append('- This is a single-model pilot, not a full-scale experiment.')
    lines.append('- It intentionally remains inside the noleak + cigsmok treatment gate.')
    lines.append('- The result is only suitable for deciding whether a larger pilot is justified.')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Wrote pilot report to {OUTPUT_PATH}')
    for key, value in metrics.items():
        print(f'{key}={value:.6f}')
    print(f'real_y_rate={real_y_rate:.4f}')
    print(f'fake_y_rate={fake_y_rate:.4f}')
    print(f'params_m={params_m:.4f}')
    print(f'train_seconds={train_seconds:.2f}')
    print(f'infer_seconds={infer_seconds:.2f}')


if __name__ == '__main__':
    main()
