"""
TSLib Layer 2 训练 - iTransformer/TimeXer
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.tslib_wrappers import iTransformerWrapper, TimeXerWrapper
from evaluation.efficiency import EfficiencyTracker
import argparse


def train_layer2(model, train_loader, val_loader, epochs, device, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    tracker = EfficiencyTracker()
    tracker.set_model_size(model)
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        with tracker.track_training():
            for batch in train_loader:
                x = batch['x'].to(device)
                traj_target = batch['trajectory_target'].to(device)
                traj_mask = batch['trajectory_valid_mask'].to(device)
                
                if x.shape[1] < 3:
                    pad_len = 3 - x.shape[1]
                    x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
                
                optimizer.zero_grad()
                output = model(x)
                
                if len(output.shape) == 3:
                    output = output.mean(dim=-1)
                if output.shape[1] > traj_target.shape[1]:
                    output = output[:, :traj_target.shape[1]]
                
                loss = criterion(output * traj_mask, traj_target * traj_mask)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss {train_loss/len(train_loader):.4f}", flush=True)
    
    tracker.set_epoch_time(np.mean(epoch_times))
    return model, tracker


def predict_layer2(model, loader, device):
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            traj_target = batch['trajectory_target'].cpu().numpy()
            traj_mask = batch['trajectory_valid_mask'].cpu().numpy()
            
            if x.shape[1] < 3:
                pad_len = 3 - x.shape[1]
                x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
            
            output = model(x).cpu().numpy()
            preds.append(output)
            targets.append(traj_target)
            masks.append(traj_mask)
    
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['itransformer', 'timexer'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='outputs/tslib_layer2')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=64)
    
    sample = next(iter(train_loader))
    feature_dim = sample['x'].shape[2]
    
    if args.model == 'itransformer':
        model = iTransformerWrapper(seq_len=3, enc_in=feature_dim, task='long_term_forecast', pred_len=7)
    else:
        model = TimeXerWrapper(seq_len=3, enc_in=feature_dim, exog_in=4, task='long_term_forecast', pred_len=7)
    
    model, tracker = train_layer2(model, train_loader, val_loader, args.epochs, device)
    
    val_pred, val_true, val_mask = predict_layer2(model, val_loader, device)
    with tracker.track_inference(len(test_df)):
        test_pred, test_true, test_mask = predict_layer2(model, test_loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        f'{args.output_dir}/{args.model}_seed{args.seed}_layer2.npz',
        val_y_pred=val_pred,
        val_y_true=val_true,
        val_y_mask=val_mask,
        test_y_pred=test_pred,
        test_y_true=test_true,
        test_y_mask=test_mask,
    )

    tracker.save_json(f'{args.output_dir}/{args.model}_efficiency_seed{args.seed}.json')
    
    print(f"\n✓ {args.model} Layer 2 完成", flush=True)


if __name__ == '__main__':
    main()
