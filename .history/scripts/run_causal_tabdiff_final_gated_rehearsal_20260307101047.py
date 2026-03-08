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
GUIDANCE_SCALE_WEIGHT = float(os.environ.get('CAUSAL_TABDIFF_SAMPLE_MODEL_SCORE_WEIGHT', '0.5'))
OUTCOME_LOSS_WEIGHT = 1.0
OUTCOME_RANK_LOSS_WEIGHT = float(os.environ.get('CAUSAL_TABDIFF_OUTCOME_RANK_LOSS_WEIGHT', '0.0'))
SAMPLE_GUIDANCE_SCALE = float(os.environ.get('CAUSAL_TABDIFF_SAMPLE_GUIDANCE_SCALE', '2.0'))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'logs', 'testing', 'causal_tabdiff_final_gated_rehearsal.md')

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
        outcome_rank_loss_weight=OUTCOME_RANK_LOSS_WEIGHT,
        sample_model_score_weight=GUIDANCE_SCALE_WEIGHT,
        sample_guidance_scale=SAMPLE_GUIDANCE_SCALE,
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

    gate_pass = (
        metrics['TSTR_AUC'] >= 0.58
        and metrics['TSTR_F1'] >= 0.05
        and metrics['TSTR_F1_RealPrev'] >= 0.05
        and metrics['TSTR_PR_AUC'] >= 0.04
    )

    lines = []
    lines.append('# CausalTabDiff Final Gated Rehearsal')
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
    lines.append(f'- Outcome rank loss weight: {OUTCOME_RANK_LOSS_WEIGHT}')
    lines.append(f'- Sample model score weight: {GUIDANCE_SCALE_WEIGHT}')
    lines.append(f'- Sample guidance scale: {SAMPLE_GUIDANCE_SCALE}')
    lines.append(f'- Params(M): {params_m:.4f}')
    lines.append(f'- Train seconds: {train_seconds:.2f}')
    lines.append(f'- Inference seconds: {infer_seconds:.2f}')
    lines.append(f'- Real Y rate: {real_y_rate:.4f}')
    lines.append(f'- Fake Y rate: {fake_y_rate:.4f}')
    lines.append('')
    lines.append('## Formalized Metrics')
    lines.append('')
    for key, value in metrics.items():
        lines.append(f'- {key}: {value:.6f}')
    lines.append('')
    lines.append('## Gate Criteria')
    lines.append('')
    lines.append('- TSTR_AUC >= 0.58')
    lines.append('- TSTR_F1 >= 0.05')
    lines.append('- TSTR_F1_RealPrev >= 0.05')
    lines.append('- TSTR_PR_AUC >= 0.04')
    lines.append(f'- Gate decision: {"PASS" if gate_pass else "HOLD"}')
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    if gate_pass:
        lines.append('- This rehearsal clears the minimal pre-full gate under the formalized prevalence-aware metric bundle.')
        lines.append('- Full-scale training is not auto-approved, but a single-model full run can now be considered for approval.')
    else:
        lines.append('- This rehearsal does not clear the minimal pre-full gate under the formalized prevalence-aware metric bundle.')
        lines.append('- The project should remain in controlled mode rather than moving to full-scale training.')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Wrote final gated rehearsal report to {OUTPUT_PATH}')
    for key, value in metrics.items():
        print(f'{key}={value:.6f}')
    print(f'gate_decision={"PASS" if gate_pass else "HOLD"}')


if __name__ == '__main__':
    main()
