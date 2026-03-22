#!/usr/bin/env python
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, '/home/UserData/ljx/Project_2/Causal-TabDiff')
from src.data.data_module_landmark import get_dataloader
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.evaluation.metrics import compute_ranking_metrics

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

ablation_type = sys.argv[1]
device = torch.device('cuda')

configs = {
    'disc_weight': [0.0, 0.25, 0.5, 0.75, 1.0],
    'diffusion_steps': [25, 50, 100, 150, 200],
    'traj_weight': [0.0, 0.5, 1.0, 1.5, 2.0],
}

os.makedirs('logs/ablations', exist_ok=True)
log = open(f'logs/ablations/{ablation_type}.log', 'w', buffering=1)
results = {}

for val in configs[ablation_type]:
    results[str(val)] = []
    for seed in [42, 52, 62, 72, 82]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        msg = f"{ablation_type}={val} seed={seed}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        
        train_loader = get_dataloader('data/landmark_tables/unified_person_landmark_table.pkl', 'train', 4096, seed)
        val_loader = get_dataloader('data/landmark_tables/unified_person_landmark_table.pkl', 'val', 4096, seed)
        
        disc_w = val if ablation_type == 'disc_weight' else 0.5
        diff_s = val if ablation_type == 'diffusion_steps' else 100
        traj_w = val if ablation_type == 'traj_weight' else 1.0
        
        model = CausalTabDiffTrajectory(3, 15, diff_s, 7).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best = 0.0
        for ep in tqdm(range(10), desc=f"{ablation_type}={val} seed={seed}"):
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                out = model(batch['x'].to(device, non_blocking=True), batch['landmark'].float().to(device, non_blocking=True))
                loss = out['diff_loss'] + disc_w * out['disc_loss'] + traj_w * model.compute_trajectory_loss(out['trajectory'], batch['trajectory_target'].to(device, non_blocking=True), batch['trajectory_valid_mask'].to(device, non_blocking=True)) + F.binary_cross_entropy(out['risk_2year'], batch['y_2year'].to(device, non_blocking=True))
                loss.backward()
                opt.step()
            
            if (ep + 1) % 5 == 0 or ep == 9:
                model.eval()
                yt, yp = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        out = model(batch['x'].to(device, non_blocking=True), batch['landmark'].float().to(device, non_blocking=True))
                        yt.append(batch['y_2year'].cpu().numpy())
                        yp.append(out['risk_2year'].cpu().numpy())
                
                auprc = compute_ranking_metrics(np.concatenate(yt).flatten(), np.concatenate(yp).flatten())['auprc']
                if not np.isnan(auprc) and auprc > best:
                    best = auprc
        
        results[str(val)].append(best)
        log.write(f"  Best={best:.4f}\n\n")
        log.flush()

log.write("\n=== Summary ===\n")
for val in configs[ablation_type]:
    auprcs = results[str(val)]
    log.write(f"{ablation_type}={val} Mean={np.mean(auprcs):.4f}±{np.std(auprcs):.4f}\n")
log.close()

with open(f'logs/ablations/{ablation_type}_results.json', 'w') as f:
    json.dump(results, f, indent=2)
