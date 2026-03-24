"""
训练 TSLib 模型 (iTransformer, TimeXer) - Layer 1 + Layer 2
"""
# pyright: reportMissingImports=false
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
from sklearn.metrics import average_precision_score
from tqdm import tqdm
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, LandmarkDataset, create_dataloaders
from baselines.tslib_wrappers import iTransformerWrapper, TimeXerWrapper
from torch.utils.data import DataLoader
from evaluation.efficiency import EfficiencyTracker
import argparse


def train_layer1(model, train_loader, val_loader, epochs, device, lr=1e-3, seed=42, train_df=None):
    """训练 Layer 1: 2-year risk prediction"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 动态计算类别不平衡权重: pos_weight = n_neg / n_pos
    # 直接从 DataFrame 计算，避免遍历 DataLoader（94K 样本遍历需 100+ 秒）
    if train_df is not None:
        n_pos = int(train_df['y_2year'].sum())
        n_total = len(train_df)
    else:
        n_pos = 0
        n_total = 0
        for batch in train_loader:
            y = batch['y_2year']
            n_pos += y.sum().item()
            n_total += y.numel()
        n_pos = int(n_pos)
    n_neg = n_total - n_pos
    pos_weight_value = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(f"Computed pos_weight from training data: {pos_weight_value:.4f} (n_pos={n_pos}, n_neg={n_neg})", flush=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    tracker = EfficiencyTracker()
    tracker.set_model_size(model)
    best_val_auprc = -float('inf')
    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience = 10
    patience_counter = 0
    epoch_times = []
    
    with tracker.track_training():
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=100, file=sys.stderr)
            for batch in pbar:
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
                pbar.set_postfix(loss=f"{train_loss/(pbar.n+1):.4f}")
        
            model.eval()
            val_loss = 0
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", ncols=100, file=sys.stderr):
                    x = batch['x'].to(device)
                    y = batch['y_2year'].to(device)
                    if x.shape[1] < 3:
                        pad_len = 3 - x.shape[1]
                        x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
                    logits = model(x)
                    loss = criterion(logits[:, 1:2], y)
                    val_loss += loss.item()
                    probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
                    labels = y.cpu().numpy().flatten()
                    val_probs.append(probs)
                    val_labels.append(labels)
        
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_probs = np.concatenate(val_probs)
            val_labels = np.concatenate(val_labels)
            val_auprc = average_precision_score(val_labels, val_probs)
        
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
        
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1}/{epochs} | Seed {seed} | LR {current_lr:.1e} | "
                f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValAUPRC {val_auprc:.4f} | "
                f"BestValAUPRC {best_val_auprc:.4f} | Time {epoch_time:.1f}s",
                flush=True
            )

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (patience={patience})", flush=True)
                break
    
    avg_epoch_time = np.mean(epoch_times)
    tracker.set_epoch_time(avg_epoch_time)

    # 恢复验证集 AUPRC 最优权重
    model.load_state_dict(best_model_state)
    
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
        train_df, val_df, test_df, landmark_to_idx, batch_size=args.batch_size, num_workers=4
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
    model, tracker = train_layer1(model, train_loader, val_loader, args.epochs, device, args.lr, seed=args.seed, train_df=train_df)
    
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
