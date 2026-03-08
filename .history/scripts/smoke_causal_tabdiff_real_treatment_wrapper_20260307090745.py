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
    empirical_y_rate = float((dataloader.dataset.y.reshape(-1) > 0.5).mean())
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

    with torch.no_grad():
        raw_low = wrapper.model.sample_with_guidance(256, torch.zeros((256, 1), device=device), guidance_scale=2.0)
        raw_high = wrapper.model.sample_with_guidance(256, torch.ones((256, 1), device=device), guidance_scale=2.0)

        model_score_low = wrapper.model.predict_outcome_proba(
            raw_low,
            torch.zeros((256, 1), device=device),
        )
        model_score_high = wrapper.model.predict_outcome_proba(
            raw_high,
            torch.ones((256, 1), device=device),
        )
        print(f"model_score_mean_low={float(model_score_low.mean().item()):.6f}")
        print(f"model_score_mean_high={float(model_score_high.mean().item()):.6f}")

        pooled_model_scores = np.concatenate(
            [
                model_score_low.detach().cpu().numpy().reshape(-1),
                model_score_high.detach().cpu().numpy().reshape(-1),
            ],
            axis=0,
        )
        model_only_threshold = float(np.quantile(pooled_model_scores, max(0.0, 1.0 - empirical_y_rate)))
        model_only_low_rate = float((model_score_low.detach().cpu().numpy().reshape(-1) >= model_only_threshold).mean())
        model_only_high_rate = float((model_score_high.detach().cpu().numpy().reshape(-1) >= model_only_threshold).mean())
        print(f"model_only_threshold={model_only_threshold:.6f}")
        print(f"model_only_low_rate={model_only_low_rate:.4f}")
        print(f"model_only_high_rate={model_only_high_rate:.4f}")

        sem_low = wrapper._decode_semantic_from_analog(raw_low, device)[:, -1, :]
        sem_high = wrapper._decode_semantic_from_analog(raw_high, device)[:, -1, :]
        glue_low = wrapper._predict_outcome_score_from_semantic(sem_low, torch.zeros((256, 1), device=device))
        glue_high = wrapper._predict_outcome_score_from_semantic(sem_high, torch.ones((256, 1), device=device))

        combined_low = 0.5 * glue_low + 0.5 * model_score_low.detach().cpu().numpy().reshape(-1)
        combined_high = 0.5 * glue_high + 0.5 * model_score_high.detach().cpu().numpy().reshape(-1)
        pooled_scores = np.concatenate([combined_low, combined_high], axis=0)
        threshold = float(np.quantile(pooled_scores, max(0.0, 1.0 - empirical_y_rate)))

        pooled_low_rate = float((combined_low >= threshold).mean())
        pooled_high_rate = float((combined_high >= threshold).mean())
        print(f"pooled_threshold={threshold:.6f}")
        print(f"pooled_threshold_low_rate={pooled_low_rate:.4f}")
        print(f"pooled_threshold_high_rate={pooled_high_rate:.4f}")


if __name__ == '__main__':
    main()
