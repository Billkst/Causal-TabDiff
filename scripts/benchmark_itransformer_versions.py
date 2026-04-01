#!/usr/bin/env python3
"""iTransformer 原始版本 vs Fast 版本严格对照实验"""
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

def get_gpu_memory():
    """获取 GPU 显存占用"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip().split('\n')[0])
    except:
        return -1

def preload_data(df, device):
    """预加载数据为张量（Fast 版本）"""
    baseline_cols = ['baseline_age', 'baseline_gender', 'baseline_race', 'baseline_cigsmok']
    t2_cols = [c for c in df.columns if c.startswith(('screen_t2_', 'abn_t2_', 'change_t2_'))]
    t1_cols = [c for c in df.columns if c.startswith(('screen_t1_', 'abn_t1_', 'change_t1_'))]
    t0_cols = [c for c in df.columns if c.startswith(('screen_t0_', 'abn_t0_', 'change_t0_'))]
    
    t0_features = df[baseline_cols + t0_cols].fillna(0).values.astype(np.float32)
    t1_features = df[baseline_cols + t1_cols].fillna(0).values.astype(np.float32)
    t2_features = df[baseline_cols + t2_cols].fillna(0).values.astype(np.float32)
    
    X = np.stack([t0_features, t1_features, t2_features], axis=1)
    y = df['y_2year'].values.astype(np.float32)
    
    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).unsqueeze(1).to(device)
    
    return X_tensor, y_tensor

def train_original(model, train_loader, val_loader, epochs, device, train_df):
    """原始版本训练（使用 DataLoader）"""
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
            print(f"[Original] Epoch {epoch+1}/{epochs} | ValAUPRC {val_auprc:.4f} | Time {epoch_time:.2f}s", flush=True)
    
    model.load_state_dict(best_model_state)
    return model, epoch_times

def train_fast(model, X_train, y_train, X_val, y_val, epochs, device, batch_size=256):
    """Fast 版本训练（预加载张量）"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_pos = int(y_train.sum().item())
    n_total = len(y_train)
    pos_weight = torch.tensor([(n_total - n_pos) / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_auprc = -float('inf')
    best_model_state = None
    n_train = len(X_train)
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        indices = torch.randperm(n_train, device=device)
        
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits[:, 1:2], y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_probs = torch.sigmoid(val_logits[:, 1:2]).cpu().numpy().flatten()
        
        from sklearn.metrics import average_precision_score
        val_auprc = average_precision_score(y_val.cpu().numpy().flatten(), val_probs)
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if (epoch + 1) % 10 == 0:
            print(f"[Fast] Epoch {epoch+1}/{epochs} | ValAUPRC {val_auprc:.4f} | Time {epoch_time:.2f}s", flush=True)
    
    model.load_state_dict(best_model_state)
    return model, epoch_times

def predict_original(model, loader, device):
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

def predict_fast(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
    return probs

def main():
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Seed: {SEED}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}", flush=True)
    print("="*80, flush=True)
    
    print("\n[1/5] Loading data...", flush=True)
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
        'data/landmark_tables/unified_person_landmark_table.pkl', seed=SEED)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)
    
    mem_before = get_gpu_memory()
    print(f"GPU Memory before: {mem_before} MB", flush=True)
    
    print("\n" + "="*80, flush=True)
    print("ORIGINAL VERSION (DataLoader)", flush=True)
    print("="*80, flush=True)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, landmark_to_idx, batch_size=BATCH_SIZE, num_workers=0)
    
    model_original = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    
    print("\n[2/5] Training original version...", flush=True)
    start_time = time.time()
    model_original, epoch_times_original = train_original(
        model_original, train_loader, val_loader, EPOCHS, device, train_df)
    total_time_original = time.time() - start_time
    
    print("\n[3/5] Predicting with original version...", flush=True)
    val_y_pred_orig, val_y_true_orig = predict_original(model_original, val_loader, device)
    test_y_pred_orig, test_y_true_orig = predict_original(model_original, test_loader, device)
    
    optimal_threshold_orig, val_f1_orig = find_optimal_threshold(val_y_true_orig, val_y_pred_orig, metric='f1')
    metrics_orig = compute_all_metrics(test_y_true_orig, test_y_pred_orig, threshold=optimal_threshold_orig)
    
    mem_after_original = get_gpu_memory()
    
    del model_original, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    
    print("\n" + "="*80, flush=True)
    print("FAST VERSION (Preloaded Tensors)", flush=True)
    print("="*80, flush=True)
    
    print("\n[4/5] Preloading data to GPU...", flush=True)
    X_train, y_train = preload_data(train_df, device)
    X_val, y_val = preload_data(val_df, device)
    X_test, y_test = preload_data(test_df, device)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)
    
    mem_after_preload = get_gpu_memory()
    
    model_fast = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    
    print("\n[5/5] Training fast version...", flush=True)
    start_time = time.time()
    model_fast, epoch_times_fast = train_fast(
        model_fast, X_train, y_train, X_val, y_val, EPOCHS, device, BATCH_SIZE)
    total_time_fast = time.time() - start_time
    
    print("\nPredicting with fast version...", flush=True)
    val_y_pred_fast = predict_fast(model_fast, X_val, device)
    test_y_pred_fast = predict_fast(model_fast, X_test, device)
    
    val_y_true_fast = y_val.cpu().numpy().flatten()
    test_y_true_fast = y_test.cpu().numpy().flatten()
    
    optimal_threshold_fast, val_f1_fast = find_optimal_threshold(val_y_true_fast, val_y_pred_fast, metric='f1')
    metrics_fast = compute_all_metrics(test_y_true_fast, test_y_pred_fast, threshold=optimal_threshold_fast)
    
    mem_after_fast = get_gpu_memory()
    
    print("\n" + "="*80, flush=True)
    print("BENCHMARK RESULTS", flush=True)
    print("="*80, flush=True)
    
    print("\n--- TIMING ---")
    print(f"Original Total Time: {total_time_original:.2f}s ({total_time_original/60:.2f} min)")
    print(f"Fast Total Time:     {total_time_fast:.2f}s ({total_time_fast/60:.2f} min)")
    print(f"Speedup:             {total_time_original/total_time_fast:.2f}x")
    
    print(f"\nOriginal Avg Epoch Time: {np.mean(epoch_times_original):.2f}s")
    print(f"Fast Avg Epoch Time:     {np.mean(epoch_times_fast):.2f}s")
    print(f"Epoch Speedup:           {np.mean(epoch_times_original)/np.mean(epoch_times_fast):.2f}x")
    
    print("\n--- METRICS ---")
    print(f"{'Metric':<20} {'Original':<12} {'Fast':<12} {'Diff':<12}")
    print("-" * 60)
    print(f"{'Test AUPRC':<20} {metrics_orig['auprc']:<12.6f} {metrics_fast['auprc']:<12.6f} {abs(metrics_orig['auprc']-metrics_fast['auprc']):<12.6f}")
    print(f"{'Test AUROC':<20} {metrics_orig['auroc']:<12.6f} {metrics_fast['auroc']:<12.6f} {abs(metrics_orig['auroc']-metrics_fast['auroc']):<12.6f}")
    print(f"{'Test F1':<20} {metrics_orig['f1']:<12.6f} {metrics_fast['f1']:<12.6f} {abs(metrics_orig['f1']-metrics_fast['f1']):<12.6f}")
    print(f"{'Test Precision':<20} {metrics_orig['precision']:<12.6f} {metrics_fast['precision']:<12.6f} {abs(metrics_orig['precision']-metrics_fast['precision']):<12.6f}")
    print(f"{'Test Recall':<20} {metrics_orig['recall']:<12.6f} {metrics_fast['recall']:<12.6f} {abs(metrics_orig['recall']-metrics_fast['recall']):<12.6f}")
    print(f"{'Optimal Threshold':<20} {optimal_threshold_orig:<12.6f} {optimal_threshold_fast:<12.6f} {abs(optimal_threshold_orig-optimal_threshold_fast):<12.6f}")
    
    print("\n--- GPU MEMORY ---")
    if mem_before > 0:
        print(f"Before:              {mem_before} MB")
        print(f"After Original:      {mem_after_original} MB (delta: {mem_after_original - mem_before} MB)")
        print(f"After Preload:       {mem_after_preload} MB (delta: {mem_after_preload - mem_before} MB)")
        print(f"After Fast:          {mem_after_fast} MB (delta: {mem_after_fast - mem_before} MB)")
    else:
        print("GPU memory monitoring not available")
    
    print("\n--- CONCLUSION ---")
    speedup = total_time_original / total_time_fast
    max_metric_diff = max(
        abs(metrics_orig['auprc'] - metrics_fast['auprc']),
        abs(metrics_orig['auroc'] - metrics_fast['auroc']),
        abs(metrics_orig['f1'] - metrics_fast['f1'])
    )
    
    print(f"Real Speedup: {speedup:.2f}x")
    print(f"Max Metric Difference: {max_metric_diff:.6f}")
    
    if max_metric_diff < 0.001:
        print("✓ Metrics are equivalent (diff < 0.001)")
    elif max_metric_diff < 0.01:
        print("✓ Metrics are acceptable (diff < 0.01)")
    else:
        print("✗ Metrics differ significantly (diff >= 0.01)")
    
    if speedup >= 10:
        print(f"✓ Fast version is {speedup:.1f}x faster - READY for 5-seed expansion")
    elif speedup >= 5:
        print(f"✓ Fast version is {speedup:.1f}x faster - acceptable for 5-seed")
    else:
        print(f"✗ Speedup {speedup:.1f}x is insufficient - investigate further")
    
    print("\n" + "="*80, flush=True)

if __name__ == '__main__':
    main()
