"""
统一消融实验框架
支持 5 个核心模块的消融研究
"""
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
EPOCHS = 50
DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'

# 消融配置
ABLATION_CONFIGS = {
    'disc_weight': {
        'name': '判别器损失权重',
        'values': [0.0, 0.25, 0.5, 1.0],
        'default': 0.5,
    },
    'diffusion_steps': {
        'name': '扩散步数',
        'values': [50, 100, 200],
        'default': 100,
    },
    'heads': {
        'name': '注意力头数',
        'values': [1, 4, 8],
        'default': 4,
    },
    'traj_weight': {
        'name': '轨迹损失权重',
        'values': [0.0, 0.5, 1.0, 2.0],
        'default': 1.0,
    },
}


def build_alpha_target(batch, device):
    if 'alpha_target' in batch:
        return batch['alpha_target'].float().to(device)
    if 'landmark' in batch:
        return batch['landmark'].float().to(device)
    raise KeyError("Batch must contain 'alpha_target' or 'landmark'")


def evaluate_model(model, dataloader, device, disc_weight, traj_weight):
    model.eval()
    total_loss = 0.0
    all_y_true, all_y_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device)
            y_2year = batch['y_2year'].to(device)
            traj_target = batch['trajectory_target'].to(device)
            traj_mask = batch['trajectory_valid_mask'].to(device)

            outputs = model(x, alpha)
            loss = (outputs['diff_loss'] + 
                   disc_weight * outputs['disc_loss'] + 
                   traj_weight * model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask) +
                   F.binary_cross_entropy(outputs['risk_2year'], y_2year))

            total_loss += loss.item()
            all_y_true.append(y_2year.cpu().numpy())
            all_y_pred.append(outputs['risk_2year'].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    y_true = np.concatenate(all_y_true).flatten()
    y_pred = np.concatenate(all_y_pred).flatten()
    return avg_loss, y_true, y_pred


def train_single_config(seed, config, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = get_dataloader(DEFAULT_TABLE_PATH, 'train', batch_size=512, seed=seed)
    val_loader = get_dataloader(DEFAULT_TABLE_PATH, 'val', batch_size=512, seed=seed)

    sample = next(iter(train_loader))
    model = CausalTabDiffTrajectory(
        t_steps=sample['x'].shape[1],
        feature_dim=sample['x'].shape[2],
        diffusion_steps=config['diffusion_steps'],
        trajectory_len=sample['trajectory_target'].shape[1],
    ).to(device)
    
    # 修改注意力头数
    model.base_model.block1 = torch.nn.MultiheadAttention(
        embed_dim=config['feature_dim'], 
        num_heads=config['heads'], 
        batch_first=True
    ).to(device) if config['heads'] != 4 else model.base_model.block1

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_val_auprc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device)
            y_2year = batch['y_2year'].to(device)
            traj_target = batch['trajectory_target'].to(device)
            traj_mask = batch['trajectory_valid_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(x, alpha)

            loss = (outputs['diff_loss'] + 
                   config['disc_weight'] * outputs['disc_loss'] + 
                   config['traj_weight'] * model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask) +
                   F.binary_cross_entropy(outputs['risk_2year'], y_2year))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_y_true, val_y_pred = evaluate_model(model, val_loader, device, config['disc_weight'], config['traj_weight'])
        metrics = compute_ranking_metrics(val_y_true, val_y_pred)
        val_auprc = metrics['auprc'] if not np.isnan(metrics['auprc']) else 0.0

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | TrainLoss {train_loss:.4f} | ValAUPRC {val_auprc:.4f} | Best {best_val_auprc:.4f}", flush=True)

    return best_val_auprc


def run_ablation(ablation_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('logs/ablations', exist_ok=True)
    
    config_info = ABLATION_CONFIGS[ablation_type]
    log_file = open(f'logs/ablations/{ablation_type}.log', 'w', buffering=1)
    
    log_file.write(f"=== {config_info['name']} 消融实验 ===\n")
    log_file.write(f"Seeds: {SEEDS}\n")
    log_file.write(f"测试值: {config_info['values']}\n")
    log_file.write(f"Epochs: {EPOCHS}\n")
    log_file.write(f"Device: {device}\n\n")
    log_file.flush()

    results = {}
    for value in config_info['values']:
        results[value] = []
        
        for seed in SEEDS:
            config = {k: v['default'] for k, v in ABLATION_CONFIGS.items()}
            config[ablation_type] = value
            config['feature_dim'] = 15
            
            msg = f"训练 | {ablation_type}={value} | Seed={seed}"
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

            best_auprc = train_single_config(seed, config, device)
            results[value].append(best_auprc)

            msg = f"  结果 | {ablation_type}={value} | Seed={seed} | BestValAUPRC={best_auprc:.4f}"
            print(msg, flush=True)
            log_file.write(msg + "\n\n")
            log_file.flush()

    log_file.write("\n=== 汇总 ===\n")
    for value in config_info['values']:
        auprcs = results[value]
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)
        msg = f"{ablation_type}={value} | Mean AUPRC={mean_auprc:.4f} ± {std_auprc:.4f}"
        print(msg, flush=True)
        log_file.write(msg + "\n")

    log_file.close()
    
    with open(f'logs/ablations/{ablation_type}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n消融实验完成: {config_info['name']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_type', type=str, required=True, 
                       choices=list(ABLATION_CONFIGS.keys()),
                       help='消融类型')
    args = parser.parse_args()
    run_ablation(args.ablation_type)
