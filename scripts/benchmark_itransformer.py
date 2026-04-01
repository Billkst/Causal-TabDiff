#!/usr/bin/env python3
"""iTransformer 训练时间诊断脚本"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.tslib_wrappers import iTransformerWrapper

def benchmark_itransformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    
    t0 = time.time()
    print("\n[1] Loading data...", flush=True)
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data('data/landmark_tables/unified_person_landmark_table.pkl', seed=42)
    t1 = time.time()
    print(f"  Data loading: {t1-t0:.2f}s", flush=True)
    
    print("\n[2] Creating dataloaders...", flush=True)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=256, num_workers=0)
    t2 = time.time()
    print(f"  Dataloader creation: {t2-t1:.2f}s", flush=True)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)
    
    print("\n[3] Initializing model...", flush=True)
    model = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2).to(device)
    t3 = time.time()
    print(f"  Model init: {t3-t2:.2f}s", flush=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_pos = int(train_df['y_2year'].sum())
    n_total = len(train_df)
    pos_weight = torch.tensor([(n_total - n_pos) / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print("\n[4] Training 3 epochs (benchmark)...", flush=True)
    for epoch in range(3):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        batch_count = 0
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y_2year'].to(device)
            if x.shape[1] < 3:
                x = torch.cat([x, torch.zeros(x.shape[0], 3-x.shape[1], x.shape[2], device=device)], dim=1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits[:, 1:2], y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        
        val_start = time.time()
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y_2year'].cpu().numpy()
                if x.shape[1] < 3:
                    x = torch.cat([x, torch.zeros(x.shape[0], 3-x.shape[1], x.shape[2], device=device)], dim=1)
                logits = model(x)
                probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
                val_probs.append(probs)
                val_labels.append(y.flatten())
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        
        from sklearn.metrics import average_precision_score
        val_auprc = average_precision_score(val_labels, val_probs)
        
        epoch_end = time.time()
        train_time = val_start - epoch_start
        val_time = epoch_end - val_start
        total_time = epoch_end - epoch_start
        
        print(f"  Epoch {epoch+1}: Total={total_time:.2f}s (Train={train_time:.2f}s, Val={val_time:.2f}s) | ValAUPRC={val_auprc:.4f}", flush=True)
    
    t4 = time.time()
    print(f"\n[5] Total 3-epoch time: {t4-t3:.2f}s", flush=True)
    print(f"  Avg per epoch: {(t4-t3)/3:.2f}s", flush=True)
    print(f"  Estimated 50-epoch time: {(t4-t3)/3*50/60:.1f} minutes", flush=True)
    
    print(f"\n[TOTAL] End-to-end time: {t4-t0:.2f}s", flush=True)

if __name__ == '__main__':
    benchmark_itransformer()
