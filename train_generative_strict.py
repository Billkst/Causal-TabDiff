import torch
import numpy as np
import sys
import os
import argparse
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['tabsyn', 'tabdiff', 'survtraj', 'sssd', 'stasy', 'tsdiff'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='outputs/b2_baseline/tstr')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n=== 训练 {args.model.upper()}_strict (Seed {args.seed}) ===", flush=True)
    
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=64)
    
    sample = next(iter(train_loader))
    feature_dim = sample['x'].shape[2]
    
    if args.model == 'tabsyn':
        from baselines.tabsyn_landmark_strict import TabSynLandmarkStrictWrapper
        model = TabSynLandmarkStrictWrapper(seq_len=3, feature_dim=feature_dim)
    elif args.model == 'tabdiff':
        from baselines.tabdiff_landmark_strict import TabDiffLandmarkStrictWrapper
        model = TabDiffLandmarkStrictWrapper(seq_len=3, feature_dim=feature_dim)
    elif args.model == 'survtraj':
        from baselines.survtraj_landmark_strict import SurvTrajLandmarkWrapper
        model = SurvTrajLandmarkWrapper(seq_len=3, feature_dim=feature_dim)
    elif args.model == 'sssd':
        from baselines.sssd_landmark_strict import SSSDLandmarkWrapper
        model = SSSDLandmarkWrapper(seq_len=3, feature_dim=feature_dim)
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    model.fit(train_loader, args.epochs, device)
    
    X_syn, Y_syn = model.sample(1000, device)
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    X_flat = X_syn.cpu().numpy().reshape(X_syn.shape[0], -1)
    clf.fit(X_flat, Y_syn.cpu().numpy().flatten())
    
    X_test_list, y_test_list = [], []
    for batch in test_loader:
        X_test_list.append(batch['x'].cpu().numpy())
        y_test_list.append(batch['y_2year'].cpu().numpy())
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0).flatten()
    
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred_proba = clf.predict_proba(X_test_flat)
    y_pred = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(f'{args.output_dir}/{args.model}_seed{args.seed}_predictions.npz',
             y_pred=y_pred, y_true=y_test)
    
    print(f"\n=== {args.model.upper()}_strict 完成 ===", flush=True)

if __name__ == '__main__':
    main()
