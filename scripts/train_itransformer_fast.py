#!/usr/bin/env python3
"""iTransformer 优化版 - 预加载数据避免 DataLoader 迭代瓶颈"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data
from baselines.tslib_wrappers import iTransformerWrapper
from evaluation.metrics import find_optimal_threshold, compute_all_metrics

def preload_data(df, device):
    """预加载数据为张量，避免 DataLoader 迭代"""
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

def train_itransformer_fast(model, X_train, y_train, X_val, y_val, epochs, device, batch_size=256):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_pos = int(y_train.sum().item())
    n_total = len(y_train)
    pos_weight = torch.tensor([(n_total - n_pos) / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_auprc = -float('inf')
    best_model_state = None
    n_train = len(X_train)
    
    for epoch in range(epochs):
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | ValAUPRC {val_auprc:.4f} | Best {best_val_auprc:.4f}", flush=True)
    
    model.load_state_dict(best_model_state)
    return model

def predict_fast(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='outputs/submission_v3/itransformer')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    
    print("Loading data...", flush=True)
    train_df, val_df, test_df, _ = load_and_split_data('data/landmark_tables/unified_person_landmark_table.pkl', seed=args.seed)
    
    print("Preloading tensors...", flush=True)
    X_train, y_train = preload_data(train_df, device)
    X_val, y_val = preload_data(val_df, device)
    X_test, y_test = preload_data(test_df, device)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)
    
    model = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    
    print(f"Training iTransformer (seed={args.seed})...", flush=True)
    model = train_itransformer_fast(model, X_train, y_train, X_val, y_val, args.epochs, device, args.batch_size)
    
    print("Predicting...", flush=True)
    val_y_pred = predict_fast(model, X_val, device)
    test_y_pred = predict_fast(model, X_test, device)
    
    val_y_true = y_val.cpu().numpy().flatten()
    test_y_true = y_test.cpu().numpy().flatten()
    
    optimal_threshold, val_f1 = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
    print(f"Optimal threshold: {optimal_threshold:.6f} (Val F1: {val_f1:.6f})", flush=True)
    
    metrics = compute_all_metrics(test_y_true, test_y_pred, threshold=optimal_threshold)
    print(f"Test AUPRC: {metrics['auprc']:.6f} | AUROC: {metrics['auroc']:.6f} | F1: {metrics['f1']:.6f}", flush=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        f"{args.output_dir}/iTransformer_seed{args.seed}_official.npz",
        val_y_true=val_y_true,
        val_y_pred=val_y_pred,
        test_y_true=test_y_true,
        test_y_pred=test_y_pred,
        optimal_threshold=optimal_threshold,
        test_f1=metrics['f1'],
        test_precision=metrics['precision'],
        test_recall=metrics['recall'],
        test_specificity=metrics['specificity'],
        test_balanced_accuracy=metrics['balanced_accuracy'],
        test_mcc=metrics['mcc'],
        seed=args.seed
    )
    print(f"Saved: {args.output_dir}/iTransformer_seed{args.seed}_official.npz", flush=True)

if __name__ == '__main__':
    main()
