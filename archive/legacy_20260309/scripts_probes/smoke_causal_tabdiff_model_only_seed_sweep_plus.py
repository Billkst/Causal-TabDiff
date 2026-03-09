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


SEEDS = [7, 42, 123, 1024, 2024]
EPOCHS = 3
GUIDANCE_SCALE = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_seed(seed: int):
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
    )
    # Still controlled: debug_mode=True => only 2 batches/epoch.
    wrapper.fit(dataloader, epochs=EPOCHS, device=device, debug_mode=True)

    with torch.no_grad():
        alpha_low = torch.zeros((256, 1), device=device)
        alpha_high = torch.ones((256, 1), device=device)
        raw_low = wrapper.model.sample_with_guidance(256, alpha_low, guidance_scale=GUIDANCE_SCALE)
        raw_high = wrapper.model.sample_with_guidance(256, alpha_high, guidance_scale=GUIDANCE_SCALE)

        model_score_low = wrapper.model.predict_outcome_proba(raw_low, alpha_low).detach().cpu().numpy().reshape(-1)
        model_score_high = wrapper.model.predict_outcome_proba(raw_high, alpha_high).detach().cpu().numpy().reshape(-1)

    pooled_model_scores = np.concatenate([model_score_low, model_score_high], axis=0)
    model_only_threshold = float(np.quantile(pooled_model_scores, max(0.0, 1.0 - empirical_y_rate)))
    model_only_low_rate = float((model_score_low >= model_only_threshold).mean())
    model_only_high_rate = float((model_score_high >= model_only_threshold).mean())

    return {
        'seed': seed,
        'cached_positive_rate': float(wrapper.y_positive_rate),
        'score_mean_low': float(model_score_low.mean()),
        'score_mean_high': float(model_score_high.mean()),
        'model_only_threshold': model_only_threshold,
        'model_only_low_rate': model_only_low_rate,
        'model_only_high_rate': model_only_high_rate,
        'direction_ok': bool(model_only_high_rate > model_only_low_rate),
        'gap': float(model_only_high_rate - model_only_low_rate),
    }


def main() -> None:
    os.environ.setdefault('DATASET_METADATA_PATH', os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json'))
    os.environ.setdefault('ALPHA_TREATMENT_COLUMN', 'cigsmok')

    print('=== CausalTabDiff model-only seed sweep plus ===')
    print(f"metadata_path={os.environ['DATASET_METADATA_PATH']}")
    print(f"alpha_column={os.environ['ALPHA_TREATMENT_COLUMN']}")
    print(f"seeds={SEEDS}")
    print(f"epochs_per_seed={EPOCHS} debug_batches_per_epoch=2 guidance_scale={GUIDANCE_SCALE}")

    results = []
    for seed in SEEDS:
        result = evaluate_seed(seed)
        results.append(result)
        print(
            'seed={seed} cached_pos={cached_positive_rate:.4f} '
            'score_low={score_mean_low:.6f} score_high={score_mean_high:.6f} '
            'low_rate={model_only_low_rate:.4f} high_rate={model_only_high_rate:.4f} '
            'gap={gap:.4f} direction_ok={direction_ok}'.format(**result)
        )

    direction_ok_count = sum(int(r['direction_ok']) for r in results)
    avg_gap = float(np.mean([r['gap'] for r in results]))
    median_gap = float(np.median([r['gap'] for r in results]))
    print(f'direction_ok_count={direction_ok_count}/{len(results)}')
    print(f'avg_gap={avg_gap:.6f}')
    print(f'median_gap={median_gap:.6f}')


if __name__ == '__main__':
    main()
