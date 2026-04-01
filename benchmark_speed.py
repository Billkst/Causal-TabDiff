import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from src.data.data_module_landmark import load_and_split_data, LandmarkDataset, collate_fn
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory

DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'


def build_alpha_target(batch, device):
    if 'alpha_target' in batch:
        return batch['alpha_target'].float().to(device)
    if 'landmark' in batch:
        return batch['landmark'].float().to(device)
    raise KeyError("Batch must contain either 'alpha_target' or 'landmark'.")


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


def benchmark_config(batch_size, num_workers, use_amp, pin_memory, persistent_workers, seed=42):
    device = torch.device('cuda')
    
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
        DEFAULT_TABLE_PATH, seed, debug_n_persons=None
    )
    train_dataset = LandmarkDataset(train_df, landmark_to_idx)
    val_dataset = LandmarkDataset(val_df, landmark_to_idx)
    
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
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
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
    
    # Warmup
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
                loss_cls = focal_loss_with_logits(
                    outputs['risk_2year_logit'], y_2year, 
                    gamma=2.0, pos_weight=pos_weight
                )
                loss = uw_loss([loss_gen, loss_traj, loss_cls])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x, alpha, history_length=hl)
            loss_gen = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
            loss_traj = model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask)
            loss_cls = focal_loss_with_logits(
                outputs['risk_2year_logit'], y_2year, 
                gamma=2.0, pos_weight=pos_weight
            )
            loss = uw_loss([loss_gen, loss_traj, loss_cls])
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize()
    
    # Benchmark training
    model.train()
    train_start = time.time()
    train_batches = 0
    
    for batch in train_loader:
        if train_batches >= 100:
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
                loss_cls = focal_loss_with_logits(
                    outputs['risk_2year_logit'], y_2year, 
                    gamma=2.0, pos_weight=pos_weight
                )
                loss = uw_loss([loss_gen, loss_traj, loss_cls])
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x, alpha, history_length=hl)
            loss_gen = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
            loss_traj = model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask)
            loss_cls = focal_loss_with_logits(
                outputs['risk_2year_logit'], y_2year, 
                gamma=2.0, pos_weight=pos_weight
            )
            loss = uw_loss([loss_gen, loss_traj, loss_cls])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        train_batches += 1
    
    torch.cuda.synchronize()
    train_time = time.time() - train_start
    
    # Benchmark validation
    model.eval()
    val_start = time.time()
    val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if val_batches >= 50:
                break
            x = batch['x'].to(device, non_blocking=True)
            alpha = build_alpha_target(batch, device)
            hl = batch['history_length'].to(device, non_blocking=True)
            outputs = model(x, alpha, history_length=hl)
            val_batches += 1
    
    torch.cuda.synchronize()
    val_time = time.time() - val_start
    
    # GPU stats
    max_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    return {
        'train_time_per_batch': train_time / train_batches,
        'val_time_per_batch': val_time / val_batches,
        'train_batches': train_batches,
        'val_batches': val_batches,
        'max_memory_gb': max_memory,
    }


def main():
    configs = [
        # Baseline
        {'batch_size': 64, 'num_workers': 4, 'use_amp': False, 'pin_memory': True, 'persistent_workers': True, 'name': 'Baseline'},
        # Larger batch
        {'batch_size': 128, 'num_workers': 4, 'use_amp': False, 'pin_memory': True, 'persistent_workers': True, 'name': 'BS128'},
        {'batch_size': 256, 'num_workers': 4, 'use_amp': False, 'pin_memory': True, 'persistent_workers': True, 'name': 'BS256'},
        # AMP
        {'batch_size': 64, 'num_workers': 4, 'use_amp': True, 'pin_memory': True, 'persistent_workers': True, 'name': 'AMP'},
        {'batch_size': 128, 'num_workers': 4, 'use_amp': True, 'pin_memory': True, 'persistent_workers': True, 'name': 'BS128+AMP'},
        {'batch_size': 256, 'num_workers': 4, 'use_amp': True, 'pin_memory': True, 'persistent_workers': True, 'name': 'BS256+AMP'},
        # Workers
        {'batch_size': 128, 'num_workers': 8, 'use_amp': True, 'pin_memory': True, 'persistent_workers': True, 'name': 'BS128+AMP+W8'},
    ]
    
    print("=" * 80)
    print("Training Speed Benchmark")
    print("=" * 80)
    print(f"{'Config':<20} {'Train(ms/batch)':<18} {'Val(ms/batch)':<18} {'Memory(GB)':<12}")
    print("-" * 80)
    
    results = []
    for cfg in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        result = benchmark_config(
            cfg['batch_size'], cfg['num_workers'], cfg['use_amp'],
            cfg['pin_memory'], cfg['persistent_workers']
        )
        
        train_ms = result['train_time_per_batch'] * 1000
        val_ms = result['val_time_per_batch'] * 1000
        mem_gb = result['max_memory_gb']
        
        print(f"{cfg['name']:<20} {train_ms:<18.1f} {val_ms:<18.1f} {mem_gb:<12.2f}")
        
        results.append({
            'name': cfg['name'],
            'config': cfg,
            'result': result,
            'train_ms': train_ms,
            'val_ms': val_ms,
            'mem_gb': mem_gb,
        })
    
    print("=" * 80)
    
    # Find best
    best = min(results, key=lambda x: x['train_ms'])
    baseline = results[0]
    speedup = baseline['train_ms'] / best['train_ms']
    
    print(f"\nBest config: {best['name']}")
    print(f"Speedup: {speedup:.2f}x vs Baseline")
    print(f"Train: {best['train_ms']:.1f} ms/batch")
    print(f"Memory: {best['mem_gb']:.2f} GB")


if __name__ == '__main__':
    main()
