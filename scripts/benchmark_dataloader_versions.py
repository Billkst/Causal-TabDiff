#!/usr/bin/env python3
"""对照实验：原始 DataLoader vs 优化 DataLoader"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import subprocess
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.tslib_wrappers import iTransformerWrapper
from evaluation.metrics import find_optimal_threshold, compute_all_metrics

class CachedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        baseline_cols = ['baseline_age', 'baseline_gender', 'baseline_race', 'baseline_cigsmok']
        t2_cols = [c for c in df.columns if c.startswith(('screen_t2_', 'abn_t2_', 'change_t2_'))]
        t1_cols = [c for c in df.columns if c.startswith(('screen_t1_', 'abn_t1_', 'change_t1_'))]
        t0_cols = [c for c in df.columns if c.startswith(('screen_t0_', 'abn_t0_', 'change_t0_'))]
        
        t0_features = df[baseline_cols + t0_cols].fillna(0).values.astype(np.float32)
        t1_features = df[baseline_cols + t1_cols].fillna(0).values.astype(np.float32)
        t2_features = df[baseline_cols + t2_cols].fillna(0).values.astype(np.float32)
        
        X = np.stack([t0_features, t1_features, t2_features], axis=1)
        y = df['y_2year'].values.astype(np.float32)
        
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'x': self.X[idx], 'y_2year': self.y[idx]}

def create_optimized_dataloaders(train_df, val_df, test_df, batch_size=256, num_workers=4):
    train_dataset = CachedTensorDataset(train_df)
    val_dataset = CachedTensorDataset(val_df)
    test_dataset = CachedTensorDataset(test_df)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_loader

def get_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip().split('\n')[0])
    except:
        return -1

def train_itransformer(model, train_loader, val_loader, epochs, device, train_df, version_name):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_pos = int(train_df['y_2year'].sum())
    n_total = len(train_df)
    pos_weight = torch.tensor([(n_total - n_pos) / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_auprc = -float('inf')
    best_model_state = None
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y_2year'].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits[:, 1:2], y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y_2year'].cpu().numpy()
                logits = model(x)
                probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
                val_probs.append(probs)
                val_labels.append(y.flatten())
        
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        
        from sklearn.metrics import average_precision_score
        val_auprc = average_precision_score(val_labels, val_probs)
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if (epoch + 1) % 10 == 0:
            print(f"[{version_name}] Epoch {epoch+1}/{epochs} | ValAUPRC {val_auprc:.4f} | Time {epoch_time:.2f}s", flush=True)
    
    model.load_state_dict(best_model_state)
    return model, epoch_times

def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y_2year'].cpu().numpy()
            logits = model(x)
            probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
            preds.append(probs)
            labels.append(y.flatten())
    return np.concatenate(preds), np.concatenate(labels)

def main():
    SEED = 42
    EPOCHS = 10
    BATCH_SIZE = 256
    NUM_WORKERS = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Benchmark: Seed={SEED}, Epochs={EPOCHS}, BatchSize={BATCH_SIZE}", flush=True)
    print("="*80, flush=True)
    
    print("\nLoading data...", flush=True)
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
        'data/landmark_tables/unified_person_landmark_table.pkl', seed=SEED)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)
    
    mem_before = get_gpu_memory()
    
    print("\n" + "="*80, flush=True)
    print("ORIGINAL DATALOADER (num_workers=0)", flush=True)
    print("="*80, flush=True)
    
    train_loader_orig, val_loader_orig, test_loader_orig = create_dataloaders(
        train_df, val_df, test_df, landmark_to_idx, batch_size=BATCH_SIZE, num_workers=0)
    
    model_orig = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    
    start_time = time.time()
    model_orig, epoch_times_orig = train_itransformer(
        model_orig, train_loader_orig, val_loader_orig, EPOCHS, device, train_df, "Original")
    total_time_orig = time.time() - start_time
    
    val_y_pred_orig, val_y_true_orig = predict(model_orig, val_loader_orig, device)
    test_y_pred_orig, test_y_true_orig = predict(model_orig, test_loader_orig, device)
    
    optimal_threshold_orig, _ = find_optimal_threshold(val_y_true_orig, val_y_pred_orig, metric='f1')
    metrics_orig = compute_all_metrics(test_y_true_orig, test_y_pred_orig, threshold=optimal_threshold_orig)
    
    mem_after_orig = get_gpu_memory()
    
    del model_orig, train_loader_orig, val_loader_orig, test_loader_orig
    torch.cuda.empty_cache()
    
    print("\n" + "="*80, flush=True)
    print(f"OPTIMIZED DATALOADER (num_workers={NUM_WORKERS}, pin_memory=True)", flush=True)
    print("="*80, flush=True)
    
    train_loader_opt, val_loader_opt, test_loader_opt = create_optimized_dataloaders(
        train_df, val_df, test_df, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    model_opt = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    
    start_time = time.time()
    model_opt, epoch_times_opt = train_itransformer(
        model_opt, train_loader_opt, val_loader_opt, EPOCHS, device, train_df, "Optimized")
    total_time_opt = time.time() - start_time
    
    val_y_pred_opt, val_y_true_opt = predict(model_opt, val_loader_opt, device)
    test_y_pred_opt, test_y_true_opt = predict(model_opt, test_loader_opt, device)
    
    optimal_threshold_opt, _ = find_optimal_threshold(val_y_true_opt, val_y_pred_opt, metric='f1')
    metrics_opt = compute_all_metrics(test_y_true_opt, test_y_pred_opt, threshold=optimal_threshold_opt)
    
    mem_after_opt = get_gpu_memory()
    
    print("\n" + "="*80, flush=True)
    print("RESULTS", flush=True)
    print("="*80, flush=True)
    
    print(f"\nOriginal:  {total_time_orig:.2f}s ({total_time_orig/60:.2f}min)")
    print(f"Optimized: {total_time_opt:.2f}s ({total_time_opt/60:.2f}min)")
    print(f"Speedup:   {total_time_orig/total_time_opt:.2f}x")
    
    print(f"\nOriginal Avg Epoch:  {np.mean(epoch_times_orig):.2f}s")
    print(f"Optimized Avg Epoch: {np.mean(epoch_times_opt):.2f}s")
    
    print(f"\n{'Metric':<15} {'Original':<10} {'Optimized':<10} {'Diff':<10}")
    print("-" * 50)
    print(f"{'AUPRC':<15} {metrics_orig['auprc']:<10.6f} {metrics_opt['auprc']:<10.6f} {abs(metrics_orig['auprc']-metrics_opt['auprc']):<10.6f}")
    print(f"{'AUROC':<15} {metrics_orig['auroc']:<10.6f} {metrics_opt['auroc']:<10.6f} {abs(metrics_orig['auroc']-metrics_opt['auroc']):<10.6f}")
    print(f"{'F1':<15} {metrics_orig['f1']:<10.6f} {metrics_opt['f1']:<10.6f} {abs(metrics_orig['f1']-metrics_opt['f1']):<10.6f}")
    print(f"{'Threshold':<15} {optimal_threshold_orig:<10.6f} {optimal_threshold_opt:<10.6f} {abs(optimal_threshold_orig-optimal_threshold_opt):<10.6f}")
    
    if mem_before > 0:
        print(f"\nGPU Memory: Before={mem_before}MB, Original={mem_after_orig}MB, Optimized={mem_after_opt}MB")
    
    speedup = total_time_orig / total_time_opt
    max_diff = max(abs(metrics_orig['auprc']-metrics_opt['auprc']), 
                   abs(metrics_orig['auroc']-metrics_opt['auroc']),
                   abs(metrics_orig['f1']-metrics_opt['f1']))
    
    print(f"\nSpeedup: {speedup:.2f}x | Max Metric Diff: {max_diff:.6f}")
    
    if max_diff < 0.001:
        status = "✓ Equivalent (diff < 0.001)"
    elif max_diff < 0.01:
        status = "✓ Acceptable (diff < 0.01)"
    else:
        status = "✗ Significant difference"
    
    print(f"{status}")
    
    if max_diff < 0.01 and speedup >= 2:
        print(f"✓ Optimized version validated - ready for 5-seed")
    elif max_diff < 0.01:
        print(f"✓ Equivalent but speedup {speedup:.1f}x modest")
    else:
        print(f"✗ Not equivalent - investigate")

if __name__ == '__main__':
    main()
