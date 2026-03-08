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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    os.environ.setdefault('DATASET_METADATA_PATH', os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json'))
    os.environ.setdefault('ALPHA_TREATMENT_COLUMN', 'cigsmok')

    set_seed(20260307)
    device = torch.device('cpu')

    dataloader = get_dataloader(
        data_dir=os.path.join(PROJECT_ROOT, 'data'),
        batch_size=256,
        debug_mode=False,
    )
    first_batch = next(iter(dataloader))
    t_steps = first_batch['x'].shape[1]
    feature_dim = first_batch['x'].shape[2]

    wrapper = CausalTabDiffWrapper(
        t_steps=t_steps,
        feature_dim=feature_dim,
        diffusion_steps=10,
    )

    print('=== CausalTabDiff wrapper real-treatment smoke ===')
    print(f"metadata_path={os.environ['DATASET_METADATA_PATH']}")
    print(f"alpha_column={os.environ['ALPHA_TREATMENT_COLUMN']}")
    print(f"dataset_alpha_source={getattr(dataloader.dataset, 'alpha_target_source', 'unknown')}")
    print(f"batch_shape_x={tuple(first_batch['x'].shape)}")
    print(f"empirical_y_rate_full={float((dataloader.dataset.y.reshape(-1) > 0.5).mean()):.4f}")
    print(f"empirical_alpha_mean_full={float(dataloader.dataset.alpha_target.mean()):.4f}")

    wrapper.fit(dataloader, epochs=1, device=device, debug_mode=True)

    print(f"cached_positive_rate={float(wrapper.y_positive_rate):.4f}")
    print(f"outcome_model_fitted={wrapper.outcome_model is not None}")
    print(f"outcome_feature_dim={wrapper.outcome_feature_dim}")

    alpha_low = torch.zeros((512, 1), device=device)
    alpha_high = torch.ones((512, 1), device=device)

    x_low, y_low = wrapper.sample(512, alpha_low, device)
    x_high, y_high = wrapper.sample(512, alpha_high, device)

    print(f"sample_low_x_shape={tuple(x_low.shape)}")
    print(f"sample_high_x_shape={tuple(x_high.shape)}")
    print(f"sample_low_y_rate={float((y_low > 0.5).float().mean().item()):.4f}")
    print(f"sample_high_y_rate={float((y_high > 0.5).float().mean().item()):.4f}")
    print(f"sample_x_mean_abs_diff={(x_low - x_high).abs().mean().item():.6f}")
    print(f"sample_y_equal={torch.allclose(y_low, y_high)}")
    print(f"sample_y_disagreement_rate={float((y_low != y_high).float().mean().item()):.4f}")

    if wrapper.outcome_model is not None:
        low_features = np.concatenate([x_low.detach().cpu().numpy(), alpha_low.detach().cpu().numpy()], axis=1)
        high_features = np.concatenate([x_high.detach().cpu().numpy(), alpha_high.detach().cpu().numpy()], axis=1)
        low_scores = wrapper.outcome_model.predict_proba(low_features)[:, 1]
        high_scores = wrapper.outcome_model.predict_proba(high_features)[:, 1]
        print(f"score_mean_low={float(low_scores.mean()):.6f}")
        print(f"score_mean_high={float(high_scores.mean()):.6f}")


if __name__ == '__main__':
    main()
