"""
训练 TSLib 模型 (iTransformer, TimeXer) - Layer 1 + Layer 2
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, LandmarkDataset, create_dataloaders
from baselines.tslib_wrappers import iTransformerWrapper, TimeXerWrapper
from torch.utils.data import DataLoader
from evaluation.efficiency import EfficiencyTracker
import argparse


def train_layer1(model, train_loader, val_loader, epochs, device, lr=1e-3):
    """训练 Layer 1: 2-year risk prediction"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 不平衡处理：2% 正例率 -> pos_weight = 49.0
    pos_weight = torch.tensor([49.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    tracker = EfficiencyTracker()
    tracker.set_model_size(model)
    best_val_loss = float('inf')
    epoch_times = []
    
    with tracker.track_training():
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            train_loss = 0
            for batch in train_loader:
                x = batch['x'].to(device)  # (batch, seq_len, features)
                y = batch['y_2year'].to(device)  # (batch, 1)
                
                # Pad to fixed length if needed
                if x.shape[1] < 3:
                    pad_len = 3 - x.shape[1]
                    x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
                
                optimizer.zero_grad()
                logits = model(x)  # (batch, 2)
                loss = criterion(logits[:, 1:2], y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['x'].to(device)
                    y = batch['y_2year'].to(device)
                    if x.shape[1] < 3:
                        pad_len = 3 - x.shape[1]
                        x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
                    logits = model(x)
                    loss = criterion(logits[:, 1:2], y)
                    val_loss += loss.item()
        
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
        
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | BestValLoss {best_val_loss:.4f}", flush=True)
    
    avg_epoch_time = np.mean(epoch_times)
    tracker.set_epoch_time(avg_epoch_time)
    
    return model, tracker


def predict_layer1(model, loader, device):
    """预测 Layer 1"""
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
            probs = torch.sigmoid(logits[:, 1]).cpu().numpy()
            preds.append(probs)
            labels.append(y.flatten())
    return np.concatenate(preds), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['itransformer', 'timexer'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='outputs/tslib_models')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n=== 训练 {args.model.upper()} (Seed {args.seed}) ===", flush=True)
    
    # Load data
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=args.seed)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, landmark_to_idx, batch_size=args.batch_size
    )
    
    # Get feature dim
    sample = next(iter(train_loader))
    feature_dim = sample['x'].shape[2]
    
    # Create model
    if args.model == 'itransformer':
        model = iTransformerWrapper(seq_len=3, enc_in=feature_dim, task='classification', num_class=2)
    else:
        model = TimeXerWrapper(seq_len=3, enc_in=feature_dim, exog_in=4, task='classification', num_class=2)
    
    # Train
    model, tracker = train_layer1(model, train_loader, val_loader, args.epochs, device, args.lr)
    
    # Predict
    val_pred, val_true = predict_layer1(model, val_loader, device)
    with tracker.track_inference(len(test_df)):
        test_pred, test_true = predict_layer1(model, test_loader, device)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        f'{args.output_dir}/{args.model}_seed{args.seed}_predictions.npz',
        val_y_true=val_true,
        val_y_pred=val_pred,
        test_y_true=test_true,
        test_y_pred=test_pred,
    )
    tracker.save_json(f'{args.output_dir}/{args.model}_efficiency_seed{args.seed}.json')
    
    print(f"\n=== {args.model.upper()} 训练完成 ===", flush=True)


if __name__ == '__main__':
    main()
