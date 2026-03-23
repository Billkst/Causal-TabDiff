"""
快速验证修复：disc_weight=[0.0, 1.0, 2.0] × 2 seeds × 15 epochs
预计耗时：1.5小时
"""
import os
import json
import time
import torch
import numpy as np
import torch.nn.functional as F
from src.data.data_module_landmark import load_and_split_data, create_dataloaders
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.evaluation.metrics import compute_ranking_metrics

SEEDS = [42, 52]
DISC_WEIGHTS = [0.0, 1.0, 2.0]
EPOCHS = 15
BATCH_SIZE = 4096
TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'

os.makedirs('outputs/quick_verify', exist_ok=True)
log_file = open('logs/quick_verify_fix.log', 'w', buffering=1)

def log_print(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

results = {}

for disc_w in DISC_WEIGHTS:
    results[disc_w] = []
    
    for seed in SEEDS:
        log_print(f"\n{'='*70}")
        log_print(f"disc_weight={disc_w} | seed={seed}")
        log_print(f"{'='*70}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_df, val_df, test_df, landmark_to_idx = load_and_split_data(TABLE_PATH, seed=seed)
        train_loader, val_loader, _ = create_dataloaders(
            train_df, val_df, test_df, landmark_to_idx, batch_size=BATCH_SIZE, num_workers=4
        )
        
        model = CausalTabDiffTrajectory(
            t_steps=3, feature_dim=15, diffusion_steps=100, trajectory_len=7
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        best_val_auprc = 0.0
        
        for epoch in range(EPOCHS):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                x = batch['x'].to(device)
                alpha = batch['landmark'].float().to(device)
                y_2year = batch['y_2year'].to(device)
                traj_target = batch['trajectory_target'].to(device)
                traj_mask = batch['trajectory_valid_mask'].to(device)
                
                optimizer.zero_grad()
                outputs = model(x, alpha)
                
                loss = (outputs['diff_loss'] 
                       + disc_w * outputs['disc_loss']
                       + model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask)
                       + F.binary_cross_entropy(outputs['risk_2year'], y_2year))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 验证
            model.eval()
            val_y_true, val_y_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['x'].to(device)
                    alpha = batch['landmark'].float().to(device)
                    y_2year = batch['y_2year'].to(device)
                    outputs = model(x, alpha)
                    val_y_true.append(y_2year.cpu().numpy())
                    val_y_pred.append(outputs['risk_2year'].cpu().numpy())
            
            val_y_true = np.concatenate(val_y_true).flatten()
            val_y_pred = np.concatenate(val_y_pred).flatten()
            metrics = compute_ranking_metrics(val_y_true, val_y_pred)
            val_auprc = metrics['auprc'] if not np.isnan(metrics['auprc']) else 0.0
            
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
            
            elapsed = time.time() - t0
            log_print(f"Epoch {epoch+1}/{EPOCHS} | TrainLoss {total_loss/len(train_loader):.4f} | "
                     f"ValAUPRC {val_auprc:.4f} | Best {best_val_auprc:.4f} | Time {elapsed:.1f}s")
        
        results[disc_w].append(best_val_auprc)
        log_print(f"Final: disc_weight={disc_w} | seed={seed} | BestValAUPRC={best_val_auprc:.4f}")

log_print(f"\n{'='*70}")
log_print("验证结果汇总")
log_print(f"{'='*70}")

for disc_w in DISC_WEIGHTS:
    auprcs = results[disc_w]
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)
    log_print(f"disc_weight={disc_w:3.1f} | AUPRC {mean_auprc:.4f}±{std_auprc:.4f} | {auprcs}")

with open('outputs/quick_verify/results.json', 'w') as f:
    json.dump(results, f, indent=2)

log_print("\n结论：如果不同disc_weight产生明显不同的AUPRC，说明修复成功！")
log_file.close()
