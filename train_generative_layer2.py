"""
训练 generative baseline 的 layer2 trajectory 版本
支持: SSSD, SurvTraj
"""
import torch
import numpy as np
import sys
import os
import argparse
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from evaluation.efficiency import EfficiencyTracker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['sssd', 'survtraj'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='outputs/b2_baseline/layer2')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n=== 训练 {args.model.upper()} Layer2 (Seed {args.seed}) ===", flush=True)
    
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=64)
    
    sample = next(iter(train_loader))
    feature_dim = sample['x'].shape[2]
    
    tracker = EfficiencyTracker()
    
    if args.model == 'sssd':
        from baselines.sssd_landmark_strict import SSSDLandmarkWrapper
        model = SSSDLandmarkWrapper(seq_len=3, feature_dim=feature_dim, trajectory_len=7)
    elif args.model == 'survtraj':
        from baselines.survtraj_landmark_strict import SurvTrajLandmarkWrapper
        model = SurvTrajLandmarkWrapper(seq_len=3, feature_dim=feature_dim, trajectory_len=7)
    
    with tracker.track_training():
        model.fit(train_loader, args.epochs, device)
    
    y_pred_list, y_true_list, y_mask_list = [], [], []
    for batch in test_loader:
        y_true_list.append(batch['trajectory_target'].cpu().numpy())
        y_mask_list.append(batch['trajectory_valid_mask'].cpu().numpy())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_mask = np.concatenate(y_mask_list, axis=0)
    
    X_syn, Y_2year, Y_traj = model.sample(y_true.shape[0], device)
    y_pred = Y_traj.cpu().numpy()
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(f'{args.output_dir}/{args.model}_seed{args.seed}_layer2.npz',
             y_pred=y_pred, y_true=y_true, y_mask=y_mask)
    
    tracker.save_json(f'{args.output_dir}/{args.model}_efficiency_seed{args.seed}.json')
    
    print(f"\n=== {args.model.upper()} Layer2 完成 ===", flush=True)

if __name__ == '__main__':
    main()
