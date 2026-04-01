import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from src.data.data_module_landmark import load_and_split_data, LandmarkDataset, collate_fn
from src.data.data_module_landmark_cached import LandmarkDatasetCached
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory

DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'


def focal_loss_with_logits(logits, targets, gamma=2.0, pos_weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = ((1 - pt) ** gamma) * bce_loss
    if pos_weight is not None:
        weight = torch.where(targets > 0.5, pos_weight.reshape(-1, 1), torch.ones_like(targets))
        focal_loss = focal_loss * weight
    return focal_loss.mean()


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigma[i])
            total = total + precision * loss + self.log_sigma[i]
        return total


def build_alpha_target(batch, device):
    if 'alpha_target' in batch:
        return batch['alpha_target'].float().to(device)
    if 'landmark' in batch:
        return batch['landmark'].float().to(device)
    raise KeyError("Batch must contain either 'alpha_target' or 'landmark'.")


def benchmark_dataset(dataset_cls, batch_size, num_workers, use_amp, seed=42):
    device = torch.device('cuda')
    
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
        DEFAULT_TABLE_PATH, seed, debug_n_persons=None
    )
    
    cache_start = time.time()
    train_dataset = dataset_cls(train_df, landmark_to_idx)
    cache_time = time.time() - cache_start
    
    train_labels = train_df['y_2year'].values.astype(int)
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
    
    sample_weights = np.where(train_labels == 1, 5.0 * n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights), len(train_labels), replacement=True,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=num_workers, 
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    
    sample = next(iter(train_loader))
    t_steps = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    trajectory_len = sample['trajectory_target'].shape[1]
    
    model = CausalTabDiffTrajectory(t_steps, feature_dim, 100, trajectory_len).to(device)
    
    uw_loss = UncertaintyWeightedLoss(n_tasks=3).to(device)
    uw_loss.log_sigma.data[2] = -1.0
    
    optimizer = torch.optim.Adam([
        {'params': model.risk_head.parameters(), 'lr': 1e-3},
        {'params': model.trajectory_head.parameters(), 'lr': 5e-4},
        {'params': model.base_model.parameters(), 'lr': 1e-4},
        {'params': uw_loss.parameters(), 'lr': 1e-2},
    ], weight_decay=1e-4)
    
    pos_weight = torch.tensor([pos_weight_val], device=device)
    scaler = GradScaler() if use_amp else None
    
    model.train()
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        x = batch['x'].to(device, non_blocking=True)
        alpha = build_alpha_target(batch, device)
        y_2year = batch['y_2year'].to(device, non_blocking=True)
        traj_target = batch['trajectory_target'].to(device, non_blocking=True)
        traj_mask = batch['trajectory_valid_mask'].to(device, non_blocking=True)
        hl = batch['history_length'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(x, alpha, history_length=hl)
                loss_gen = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
                loss_traj = model.compute_trajectory_loss(outputs['trajectory_logits'], traj_target, traj_mask)
                loss_cls = focal_loss_with_logits(outputs['risk_2year_logit'], y_2year, gamma=2.0, pos_weight=pos_weight)
                loss = uw_loss([loss_gen, loss_traj, loss_cls])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x, alpha, history_length=hl)
            loss_gen = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
            loss_traj = model.compute_trajectory_loss(outputs['trajectory_logits'], traj_target, traj_mask)
            loss_cls = focal_loss_with_logits(outputs['risk_2year_logit'], y_2year, gamma=2.0, pos_weight=pos_weight)
            loss = uw_loss([loss_gen, loss_traj, loss_cls])
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize()
    
    model.train()
    train_start = time.time()
    train_batches = 0
    
    for batch in train_loader:
        if train_batches >= 200:
            break
        
        x = batch['x'].to(device, non_blocking=True)
        alpha = build_alpha_target(batch, device)
        y_2year = batch['y_2year'].to(device, non_blocking=True)
        traj_target = batch['trajectory_target'].to(device, non_blocking=True)
        traj_mask = batch['trajectory_valid_mask'].to(device, non_blocking=True)
        hl = batch['history_length'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(x, alpha, history_length=hl)
                loss_gen = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
                loss_traj = model.compute_trajectory_loss(outputs['trajectory_logits'], traj_target, traj_mask)
                loss_cls = focal_loss_with_logits(outputs['risk_2year_logit'], y_2year, gamma=2.0, pos_weight=pos_weight)
                loss = uw_loss([loss_gen, loss_traj, loss_cls])
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x, alpha, history_length=hl)
            loss_gen = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
            loss_traj = model.compute_trajectory_loss(outputs['trajectory_logits'], traj_target, traj_mask)
            loss_cls = focal_loss_with_logits(outputs['risk_2year_logit'], y_2year, gamma=2.0, pos_weight=pos_weight)
            loss = uw_loss([loss_gen, loss_traj, loss_cls])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        train_batches += 1
    
    torch.cuda.synchronize()
    train_time = time.time() - train_start
    
    samples_per_sec = (train_batches * batch_size) / train_time
    
    return {
        'cache_time': cache_time,
        'train_time': train_time,
        'train_batches': train_batches,
        'samples_per_sec': samples_per_sec,
    }


def main():
    configs = [
        {'name': 'Original BS128+AMP+W8', 'dataset': LandmarkDataset, 'bs': 128, 'workers': 8, 'amp': True},
        {'name': 'Cached BS128+AMP+W8', 'dataset': LandmarkDatasetCached, 'bs': 128, 'workers': 8, 'amp': True},
        {'name': 'Cached BS192+AMP+W8', 'dataset': LandmarkDatasetCached, 'bs': 192, 'workers': 8, 'amp': True},
        {'name': 'Cached BS256+AMP+W8', 'dataset': LandmarkDatasetCached, 'bs': 256, 'workers': 8, 'amp': True},
    ]
    
    print("=" * 90)
    print("Dataset Caching Benchmark")
    print("=" * 90)
    print(f"{'Config':<30} {'Cache(s)':<12} {'samples/sec':<15} {'Speedup':<10}")
    print("-" * 90)
    
    baseline_throughput = None
    for cfg in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        result = benchmark_dataset(
            cfg['dataset'], cfg['bs'], cfg['workers'], cfg['amp']
        )
        
        if baseline_throughput is None:
            baseline_throughput = result['samples_per_sec']
        
        speedup = result['samples_per_sec'] / baseline_throughput
        
        print(f"{cfg['name']:<30} {result['cache_time']:<12.1f} {result['samples_per_sec']:<15.1f} {speedup:<10.2f}x")
    
    print("=" * 90)


if __name__ == '__main__':
    main()
