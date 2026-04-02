#!/usr/bin/env python3
"""iTransformer 等价快速版本 - 使用优化的 DataLoader"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data
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

def train_itransformer(model, train_loader, val_loader, epochs, device, train_df):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_pos = int(train_df['y_2year'].sum())
    n_total = len(train_df)
    pos_weight = torch.tensor([(n_total - n_pos) / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_auprc = -float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | ValAUPRC {val_auprc:.4f} | Best {best_val_auprc:.4f}", flush=True)
    
    model.load_state_dict(best_model_state)
    return model

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
    import argparse
    import time
    import subprocess
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs/submission_v3/itransformer')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Optimized DataLoader: num_workers={args.num_workers}, pin_memory=True", flush=True)
    
    print("Loading data...", flush=True)
    train_df, val_df, test_df, _ = load_and_split_data(
        'data/landmark_tables/unified_person_landmark_table.pkl', seed=args.seed)
    
    print("Creating optimized dataloaders...", flush=True)
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        train_df, val_df, test_df, batch_size=args.batch_size, num_workers=args.num_workers)
    
    model = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    
    print(f"Training iTransformer (seed={args.seed})...", flush=True)
    start_time = time.time()
    model = train_itransformer(model, train_loader, val_loader, args.epochs, device, train_df)
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}min)", flush=True)
    
    print("Predicting...", flush=True)
    val_y_pred, val_y_true = predict(model, val_loader, device)
    test_y_pred, test_y_true = predict(model, test_loader, device)
    
    optimal_threshold, val_f1 = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
    print(f"Optimal threshold: {optimal_threshold:.6f} (Val F1: {val_f1:.6f})", flush=True)
    
    metrics = compute_all_metrics(test_y_true, test_y_pred, threshold=optimal_threshold)
    print(f"Test AUPRC: {metrics['auprc']:.6f} | AUROC: {metrics['auroc']:.6f} | F1: {metrics['f1']:.6f}", flush=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        f"{args.output_dir}/iTransformer_seed{args.seed}_optimized.npz",
        val_y_true=val_y_true, val_y_pred=val_y_pred,
        test_y_true=test_y_true, test_y_pred=test_y_pred,
        optimal_threshold=optimal_threshold,
        test_f1=metrics['f1'], test_precision=metrics['precision'],
        test_recall=metrics['recall'], test_specificity=metrics['specificity'],
        test_balanced_accuracy=metrics['balanced_accuracy'], test_mcc=metrics['mcc'],
        seed=args.seed
    )
    print(f"Saved: {args.output_dir}/iTransformer_seed{args.seed}_optimized.npz", flush=True)

if __name__ == '__main__':
    main()
