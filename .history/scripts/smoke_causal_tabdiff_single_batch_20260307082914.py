import os
import random

import numpy as np
import torch

from src.data.data_module import get_dataloader
from src.models.causal_tabdiff import CausalTabDiff


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.environ.setdefault(
        'DATASET_METADATA_PATH',
        os.path.join(project_root, 'src', 'data', 'dataset_metadata_noleak.json'),
    )

    set_seed(20260307)
    device = torch.device('cpu')

    dataloader = get_dataloader(
        data_dir=os.path.join(project_root, 'data'),
        batch_size=4,
        debug_mode=True,
    )
    batch = next(iter(dataloader))

    x = batch['x'].to(device)
    y = batch['y'].to(device)
    alpha_target = batch['alpha_target'].to(device)

    print('=== CausalTabDiff 单批烟测 ===')
    print(f'metadata_path={os.environ.get("DATASET_METADATA_PATH")}')
    print(f'x.shape={tuple(x.shape)} y.shape={tuple(y.shape)} alpha.shape={tuple(alpha_target.shape)}')
    print(f'x finite={torch.isfinite(x).all().item()} y finite={torch.isfinite(y).all().item()} alpha finite={torch.isfinite(alpha_target).all().item()}')
    print(f'y_positive_rate_batch={(y > 0.5).float().mean().item():.4f}')
    print(f'max_time_slice_delta_01={(x[:, 0, :] - x[:, 1, :]).abs().max().item():.6f}')
    print(f'max_time_slice_delta_12={(x[:, 1, :] - x[:, 2, :]).abs().max().item():.6f}')

    t_steps = x.shape[1]
    feature_dim = x.shape[2]
    model = CausalTabDiff(
        t_steps=t_steps,
        feature_dim=feature_dim,
        diffusion_steps=10,
        heads=1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad()
    diff_loss, disc_loss = model(x, alpha_target)
    total_loss = diff_loss + 0.5 * disc_loss
    total_loss.backward()

    grad_norm_sq = 0.0
    params_with_grad = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_sq += float(param.grad.detach().pow(2).sum().item())
            params_with_grad += 1

    optimizer.step()

    print(f'diff_loss={diff_loss.item():.6f}')
    print(f'disc_loss={disc_loss.item():.6f}')
    print(f'total_loss={total_loss.item():.6f}')
    print(f'params_with_grad={params_with_grad}')
    print(f'global_grad_norm={(grad_norm_sq ** 0.5):.6f}')
    print(f'loss_is_finite={torch.isfinite(total_loss).item()}')

    model.eval()
    alpha_low = torch.full((2, 1), 0.1, device=device)
    alpha_high = torch.full((2, 1), 0.9, device=device)

    torch.manual_seed(999)
    out_low = model.sample_with_guidance(2, alpha_low, guidance_scale=1.0)
    torch.manual_seed(999)
    out_high = model.sample_with_guidance(2, alpha_high, guidance_scale=1.0)

    mean_abs_diff = (out_low - out_high).abs().mean().item()
    print(f'conditional_mean_abs_diff={mean_abs_diff:.6f}')
    print(f'conditional_outputs_equal={torch.allclose(out_low, out_high)}')


if __name__ == '__main__':
    main()