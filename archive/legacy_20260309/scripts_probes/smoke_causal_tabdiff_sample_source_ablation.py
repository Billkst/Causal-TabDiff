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


SEEDS = [7, 11, 42, 77, 123, 256, 512, 1024, 2024, 4096]
EPOCHS = 3
GUIDANCE_SCALE = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pooled_direction(score_low: np.ndarray, score_high: np.ndarray, empirical_y_rate: float):
    pooled = np.concatenate([score_low, score_high], axis=0)
    threshold = float(np.quantile(pooled, max(0.0, 1.0 - empirical_y_rate)))
    low_rate = float((score_low >= threshold).mean())
    high_rate = float((score_high >= threshold).mean())
    return {
        'threshold': threshold,
        'low_rate': low_rate,
        'high_rate': high_rate,
        'gap': float(high_rate - low_rate),
        'direction_ok': bool(high_rate > low_rate),
    }


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
    wrapper.fit(dataloader, epochs=EPOCHS, device=device, debug_mode=True)

    with torch.no_grad():
        alpha_low = torch.zeros((256, 1), device=device)
        alpha_high = torch.ones((256, 1), device=device)
        raw_low = wrapper.model.sample_with_guidance(256, alpha_low, guidance_scale=GUIDANCE_SCALE)
        raw_high = wrapper.model.sample_with_guidance(256, alpha_high, guidance_scale=GUIDANCE_SCALE)

        sem_low = wrapper._decode_semantic_from_analog(raw_low, device)[:, -1, :]
        sem_high = wrapper._decode_semantic_from_analog(raw_high, device)[:, -1, :]

        glue_low = wrapper._predict_outcome_score_from_semantic(sem_low, alpha_low)
        glue_high = wrapper._predict_outcome_score_from_semantic(sem_high, alpha_high)
        model_low = wrapper.model.predict_outcome_proba(raw_low, alpha_low).detach().cpu().numpy().reshape(-1)
        model_high = wrapper.model.predict_outcome_proba(raw_high, alpha_high).detach().cpu().numpy().reshape(-1)
        blend_low = 0.5 * glue_low + 0.5 * model_low
        blend_high = 0.5 * glue_high + 0.5 * model_high

    return {
        'seed': seed,
        'glue': pooled_direction(glue_low, glue_high, empirical_y_rate),
        'blend': pooled_direction(blend_low, blend_high, empirical_y_rate),
        'model': pooled_direction(model_low, model_high, empirical_y_rate),
    }


def summarize(results, key: str):
    ok_count = sum(int(r[key]['direction_ok']) for r in results)
    gaps = [r[key]['gap'] for r in results]
    return {
        'ok_count': ok_count,
        'avg_gap': float(np.mean(gaps)),
        'median_gap': float(np.median(gaps)),
        'min_gap': float(np.min(gaps)),
        'max_gap': float(np.max(gaps)),
    }


def main() -> None:
    os.environ.setdefault('DATASET_METADATA_PATH', os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json'))
    os.environ.setdefault('ALPHA_TREATMENT_COLUMN', 'cigsmok')

    print('=== CausalTabDiff sample-source ablation ===')
    print(f"metadata_path={os.environ['DATASET_METADATA_PATH']}")
    print(f"alpha_column={os.environ['ALPHA_TREATMENT_COLUMN']}")
    print(f"seeds={SEEDS}")
    print(f"epochs_per_seed={EPOCHS} debug_batches_per_epoch=2 guidance_scale={GUIDANCE_SCALE}")

    results = []
    for seed in SEEDS:
        result = evaluate_seed(seed)
        results.append(result)
        print(
            f"seed={seed} "
            f"glue_gap={result['glue']['gap']:.4f} glue_ok={result['glue']['direction_ok']} "
            f"blend_gap={result['blend']['gap']:.4f} blend_ok={result['blend']['direction_ok']} "
            f"model_gap={result['model']['gap']:.4f} model_ok={result['model']['direction_ok']}"
        )

    for key in ['glue', 'blend', 'model']:
        s = summarize(results, key)
        print(
            f"{key}_direction_ok_count={s['ok_count']}/{len(results)} "
            f"avg_gap={s['avg_gap']:.6f} median_gap={s['median_gap']:.6f} "
            f"min_gap={s['min_gap']:.6f} max_gap={s['max_gap']:.6f}"
        )


if __name__ == '__main__':
    main()
