#!/usr/bin/env python3
"""等价优化版本 - 保留原始语义，只优化数据加载"""
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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip().split('\n')[0])
    except:
        return -1

def train_itransformer(model, train_loader, val_loader, epochs, device, train_df, version_name, seed):
    set_seed(seed)
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
    import torch.nn as nn
    
    SEED = 42
    EPOCHS = 10
    BATCH_SIZE = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Seed={SEED}, Epochs={EPOCHS}, BatchSize={BATCH_SIZE}", flush=True)
    print("="*80, flush=True)
    
    print("\nLoading data...", flush=True)
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
        'data/landmark_tables/unified_person_landmark_table.pkl', seed=SEED)
    
    mem_before = get_gpu_memory()
    
    print("\n" + "="*80, flush=True)
    print("ORIGINAL (num_workers=0)", flush=True)
    print("="*80, flush=True)
    
    set_seed(SEED)
    train_loader_orig, val_loader_orig, test_loader_orig = create_dataloaders(
        train_df, val_df, test_df, landmark_to_idx, batch_size=BATCH_SIZE, num_workers=0)
    
    model_orig = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    
    start_time = time.time()
    model_orig, epoch_times_orig = train_itransformer(
        model_orig, train_loader_orig, val_loader_orig, EPOCHS, device, train_df, "Original", SEED)
    total_time_orig = time.time() - start_time
    
    val_y_pred_orig, val_y_true_orig = predict(model_orig, val_loader_orig, device)
    test_y_pred_orig, test_y_true_orig = predict(model_orig, test_loader_orig, device)
    
    optimal_threshold_orig, _ = find_optimal_threshold(val_y_true_orig, val_y_pred_orig, metric='f1')
    metrics_orig = compute_all_metrics(test_y_true_orig, test_y_pred_orig, threshold=optimal_threshold_orig)
    
    mem_after_orig = get_gpu_memory()
    
    del model_orig, train_loader_orig, val_loader_orig, test_loader_orig
    torch.cuda.empty_cache()
    
    print("\n" + "="*80, flush=True)
    print("OPTIMIZED (num_workers=4, pin_memory=True)", flush=True)
    print("="*80, flush=True)
    
    set_seed(SEED)
    train_loader_opt, val_loader_opt, test_loader_opt = create_dataloaders(
        train_df, val_df, test_df, landmark_to_idx, batch_size=BATCH_SIZE, num_workers=4)
    
    model_opt = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    
    start_time = time.time()
    model_opt, epoch_times_opt = train_itransformer(
        model_opt, train_loader_opt, val_loader_opt, EPOCHS, device, train_df, "Optimized", SEED)
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
    
    print(f"\n{'Metric':<15} {'Original':<10} {'Optimized':<10} {'Diff':<10}")
    print("-" * 50)
    print(f"{'AUPRC':<15} {metrics_orig['auprc']:<10.6f} {metrics_opt['auprc']:<10.6f} {abs(metrics_orig['auprc']-metrics_opt['auprc']):<10.6f}")
    print(f"{'AUROC':<15} {metrics_orig['auroc']:<10.6f} {metrics_opt['auroc']:<10.6f} {abs(metrics_orig['auroc']-metrics_opt['auroc']):<10.6f}")
    print(f"{'F1':<15} {metrics_orig['f1']:<10.6f} {metrics_opt['f1']:<10.6f} {abs(metrics_orig['f1']-metrics_opt['f1']):<10.6f}")
    
    if mem_before > 0:
        print(f"\nGPU: Before={mem_before}MB, Original={mem_after_orig}MB, Optimized={mem_after_opt}MB")
    
    max_diff = max(abs(metrics_orig['auprc']-metrics_opt['auprc']), 
                   abs(metrics_orig['auroc']-metrics_opt['auroc']),
                   abs(metrics_orig['f1']-metrics_opt['f1']))
    
    print(f"\nMax Diff: {max_diff:.6f}")
    
    if max_diff < 0.001:
        print("✓ Equivalent")
    elif max_diff < 0.01:
        print("✓ Acceptable")
    else:
        print("✗ Not equivalent")

if __name__ == '__main__':
    main()
