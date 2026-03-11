import numpy as np
import torch
from scipy.stats import wasserstein_distance


def compute_ate_bias(model, dataloader, device, alpha_low=0.0, alpha_high=1.0):
    model.eval()
    ate_estimates = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            
            alpha_0 = torch.full((x.shape[0], 1), alpha_low, device=device)
            alpha_1 = torch.full((x.shape[0], 1), alpha_high, device=device)
            
            risk_0 = model(x, alpha_0)['risk_2year'].cpu().numpy()
            risk_1 = model(x, alpha_1)['risk_2year'].cpu().numpy()
            
            ate = (risk_1 - risk_0).mean()
            ate_estimates.append(ate)
    
    return np.mean(ate_estimates)


def compute_wasserstein(real_data, generated_data):
    real_flat = real_data.flatten()
    gen_flat = generated_data.flatten()
    return wasserstein_distance(real_flat, gen_flat)


def compute_cmd(real_data, generated_data, n_moments=5):
    real_flat = real_data.flatten()
    gen_flat = generated_data.flatten()
    
    cmd = 0.0
    for k in range(1, n_moments + 1):
        real_moment = np.mean(real_flat ** k)
        gen_moment = np.mean(gen_flat ** k)
        cmd += np.abs(real_moment - gen_moment)
    
    return cmd


def evaluate_causal_and_distribution(model, dataloader, device, output_dir=None):
    ate_bias = compute_ate_bias(model, dataloader, device)
    
    print(f"\n=== Causal & Distribution Metrics ===")
    print(f"ATE Bias: {ate_bias:.4f}")
    
    metrics = {
        'ate_bias': float(ate_bias)
    }
    
    if output_dir:
        import os, json
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'causal_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics
