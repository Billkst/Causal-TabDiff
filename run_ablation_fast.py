import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from src.data.data_module_landmark import get_dataloader
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.evaluation.metrics import compute_ranking_metrics

SEEDS = [42, 52, 62, 72, 82]
EPOCHS = 30
BATCH_SIZE = 1024

CONFIGS = {
    'disc_weight': {'name': '判别器权重', 'values': [0.0, 0.25, 0.5, 0.75, 1.0], 'default': 0.5},
    'diffusion_steps': {'name': '扩散步数', 'values': [25, 50, 100, 150, 200], 'default': 100},
    'heads': {'name': '注意力头数', 'values': [1, 2, 4, 6, 8], 'default': 4},
    'traj_weight': {'name': '轨迹权重', 'values': [0.0, 0.5, 1.0, 1.5, 2.0], 'default': 1.0},
}

def train_one(seed, disc_w, diff_steps, traj_w, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader = get_dataloader('data/landmark_tables/unified_person_landmark_table.pkl', 'train', batch_size=BATCH_SIZE, seed=seed)
    val_loader = get_dataloader('data/landmark_tables/unified_person_landmark_table.pkl', 'val', batch_size=BATCH_SIZE, seed=seed)
    
    model = CausalTabDiffTrajectory(3, 15, diff_steps, 7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    best_auprc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            x = batch['x'].to(device)
            alpha = batch['landmark'].float().to(device)
            y = batch['y_2year'].to(device)
            traj_t = batch['trajectory_target'].to(device)
            traj_m = batch['trajectory_valid_mask'].to(device)
            
            optimizer.zero_grad()
            out = model(x, alpha)
            loss = out['diff_loss'] + disc_w * out['disc_loss'] + traj_w * model.compute_trajectory_loss(out['trajectory'], traj_t, traj_m) + F.binary_cross_entropy(out['risk_2year'], y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                alpha = batch['landmark'].float().to(device)
                out = model(x, alpha)
                y_true.append(batch['y_2year'].cpu().numpy())
                y_pred.append(out['risk_2year'].cpu().numpy())
        
        y_true = np.concatenate(y_true).flatten()
        y_pred = np.concatenate(y_pred).flatten()
        auprc = compute_ranking_metrics(y_true, y_pred)['auprc']
        if not np.isnan(auprc) and auprc > best_auprc:
            best_auprc = auprc
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | AUPRC {auprc:.4f} | Best {best_auprc:.4f}", flush=True)
    
    return best_auprc

def run(ablation_type):
    device = torch.device('cuda')
    os.makedirs('logs/ablations', exist_ok=True)
    
    cfg = CONFIGS[ablation_type]
    log = open(f'logs/ablations/{ablation_type}.log', 'w', buffering=1)
    log.write(f"=== {cfg['name']} 消融 ===\n测试值: {cfg['values']}\n\n")
    
    results = {}
    for val in cfg['values']:
        results[str(val)] = []
        for seed in SEEDS:
            defaults = {k: v['default'] for k, v in CONFIGS.items()}
            defaults[ablation_type] = val
            
            msg = f"{ablation_type}={val} | Seed={seed}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            
            auprc = train_one(seed, defaults['disc_weight'], defaults['diffusion_steps'], defaults['traj_weight'], device)
            results[str(val)].append(auprc)
            
            msg = f"  AUPRC={auprc:.4f}\n"
            print(msg, flush=True)
            log.write(msg)
            log.flush()
    
    log.write("\n=== 汇总 ===\n")
    for val in cfg['values']:
        auprcs = results[str(val)]
        mean = np.mean(auprcs)
        std = np.std(auprcs)
        msg = f"{ablation_type}={val} | Mean={mean:.4f}±{std:.4f}\n"
        print(msg, flush=True)
        log.write(msg)
    
    log.close()
    with open(f'logs/ablations/{ablation_type}_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_type', required=True, choices=list(CONFIGS.keys()))
    run(parser.parse_args().ablation_type)
