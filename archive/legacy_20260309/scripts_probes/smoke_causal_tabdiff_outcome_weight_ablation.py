import os
import random
import sys

import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baselines.wrappers import CausalTabDiffWrapper
from src.data.data_module import get_dataloader


SEEDS = [512, 1024, 42]
OUTCOME_LOSS_WEIGHTS = [1.0, 1.5, 2.0, 3.0]
EPOCHS = 3
GUIDANCE_SCALE = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(seed: int, outcome_loss_weight: float):
    set_seed(seed)
    device = torch.device('cpu')

    dataloader = get_dataloader(
        data_dir=os.path.join(PROJECT_ROOT, 'data'),
        batch_size=256,
        debug_mode=False,
    )
    empirical_y_rate = float((dataloader.dataset.y.reshape(-1) > 0.5).mean())
    first_batch = next(iter(dataloader))

    wrapper = CausalTabDiffWrapper(
        t_steps=first_batch['x'].shape[1],
        feature_dim=first_batch['x'].shape[2],
        diffusion_steps=10,
        outcome_loss_weight=outcome_loss_weight,
    )
    wrapper.fit(dataloader, epochs=EPOCHS, device=device, debug_mode=True)

    with torch.no_grad():
        alpha_low = torch.zeros((256, 1), device=device)
        alpha_high = torch.ones((256, 1), device=device)
        raw_low = wrapper.model.sample_with_guidance(256, alpha_low, guidance_scale=GUIDANCE_SCALE)
        raw_high = wrapper.model.sample_with_guidance(256, alpha_high, guidance_scale=GUIDANCE_SCALE)
        score_low = wrapper.model.predict_outcome_proba(raw_low, alpha_low).detach().cpu().numpy().reshape(-1)
        score_high = wrapper.model.predict_outcome_proba(raw_high, alpha_high).detach().cpu().numpy().reshape(-1)

    pooled_scores = np.concatenate([score_low, score_high], axis=0)
    threshold = float(np.quantile(pooled_scores, max(0.0, 1.0 - empirical_y_rate)))
    low_rate = float((score_low >= threshold).mean())
    high_rate = float((score_high >= threshold).mean())

    return {
        'seed': seed,
        'outcome_loss_weight': outcome_loss_weight,
        'score_mean_low': float(score_low.mean()),
        'score_mean_high': float(score_high.mean()),
        'low_rate': low_rate,
        'high_rate': high_rate,
        'gap': float(high_rate - low_rate),
        'direction_ok': bool(high_rate > low_rate),
    }


def main() -> None:
    os.environ.setdefault('DATASET_METADATA_PATH', os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json'))
    os.environ.setdefault('ALPHA_TREATMENT_COLUMN', 'cigsmok')

    print('=== CausalTabDiff outcome-loss-weight ablation ===')
    print(f"metadata_path={os.environ['DATASET_METADATA_PATH']}")
    print(f"alpha_column={os.environ['ALPHA_TREATMENT_COLUMN']}")
    print(f"seeds={SEEDS}")
    print(f"weights={OUTCOME_LOSS_WEIGHTS}")
    print(f"epochs_per_run={EPOCHS} debug_batches_per_epoch=2 guidance_scale={GUIDANCE_SCALE}")

    summary = {}
    for weight in OUTCOME_LOSS_WEIGHTS:
        runs = []
        for seed in SEEDS:
            result = evaluate(seed, weight)
            runs.append(result)
            print(
                'w={outcome_loss_weight:.1f} seed={seed} '
                'score_low={score_mean_low:.6f} score_high={score_mean_high:.6f} '
                'low_rate={low_rate:.4f} high_rate={high_rate:.4f} '
                'gap={gap:.4f} direction_ok={direction_ok}'.format(**result)
            )
        ok_count = sum(int(r['direction_ok']) for r in runs)
        avg_gap = float(np.mean([r['gap'] for r in runs]))
        summary[weight] = {'ok_count': ok_count, 'avg_gap': avg_gap}
        print(f'weight={weight:.1f} direction_ok_count={ok_count}/{len(runs)} avg_gap={avg_gap:.6f}')

    best = max(summary.items(), key=lambda kv: (kv[1]['ok_count'], kv[1]['avg_gap']))
    print(f'best_weight={best[0]:.1f} best_direction_ok_count={best[1]["ok_count"]}/{len(SEEDS)} best_avg_gap={best[1]["avg_gap"]:.6f}')


if __name__ == '__main__':
    main()
