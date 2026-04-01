#!/usr/bin/env python3
"""iTransformer 训练脚本 - 符合 submission_v3 协议"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.tslib_wrappers import iTransformerWrapper
from evaluation.metrics import find_optimal_threshold, compute_all_metrics

def train_itransformer(model, train_loader, val_loader, epochs, device, train_df):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_pos = int(train_df['y_2year'].sum())
    n_total = len(train_df)
    n_neg = n_total - n_pos
    pos_weight = torch.tensor([(n_neg / n_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_auprc = -float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y_2year'].to(device)
            
            if x.shape[1] < 3:
                pad_len = 3 - x.shape[1]
                x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits[:, 1:2], y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y_2year'].cpu().numpy()
                if x.shape[1] < 3:
                    pad_len = 3 - x.shape[1]
                    x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
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
        
        print(f"Epoch {epoch+1}/{epochs} | ValAUPRC {val_auprc:.4f} | BestValAUPRC {best_val_auprc:.4f}", flush=True)
    
    model.load_state_dict(best_model_state)
    return model

def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y_2year'].cpu().numpy()
            if x.shape[1] < 3:
                pad_len = 3 - x.shape[1]
                x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
            logits = model(x)
            probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
            preds.append(probs)
            labels.append(y.flatten())
    return np.concatenate(preds), np.concatenate(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='outputs/submission_v3/itransformer')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data('data/landmark_tables/unified_person_landmark_table.pkl', seed=args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=256, num_workers=0)
    
    model = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    
    print(f"Training iTransformer (seed={args.seed})...", flush=True)
    model = train_itransformer(model, train_loader, val_loader, args.epochs, device, train_df)
    
    print("Predicting...", flush=True)
    val_y_pred, val_y_true = predict(model, val_loader, device)
    test_y_pred, test_y_true = predict(model, test_loader, device)
    
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
