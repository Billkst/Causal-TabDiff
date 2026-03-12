"""
完整 TSTR Pipeline 实现 - STaSy/TabSyn/TabDiff/TSDiff
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.tstr_pipeline import TSTRPipeline, extract_features_and_labels
from baselines.wrappers import STaSyWrapper, TabSynWrapper, TabDiffWrapper, TSDiffWrapper
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['stasy', 'tabsyn', 'tabdiff', 'tsdiff'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen_epochs', type=int, default=50, help='生成模型训练轮数')
    parser.add_argument('--n_synthetic', type=int, default=2000, help='合成样本数量')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n=== TSTR Pipeline: {args.model.upper()} (Seed {args.seed}) ===", flush=True)
    
    # Load data
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, landmark_to_idx, batch_size=args.batch_size
    )
    
    # Get dimensions
    sample = next(iter(train_loader))
    seq_len = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    
    # Create generative model
    print(f"[TSTR] 创建生成模型: {args.model}", flush=True)
    if args.model == 'stasy':
        gen_model = STaSyWrapper(t_steps=seq_len, feature_dim=feature_dim)
    elif args.model == 'tabsyn':
        gen_model = TabSynWrapper(t_steps=seq_len, feature_dim=feature_dim)
    elif args.model == 'tabdiff':
        gen_model = TabDiffWrapper(t_steps=seq_len, feature_dim=feature_dim)
    else:  # tsdiff
        gen_model = TSDiffWrapper(t_steps=seq_len, feature_dim=feature_dim)
    
    # Create TSTR pipeline
    pipeline = TSTRPipeline(gen_model, downstream_classifier='xgboost')
    
    # Step 1: Train generative model
    print(f"\n[TSTR] Step 1: 训练生成模型 ({args.gen_epochs} epochs)", flush=True)
    pipeline.train_generative_model(train_loader, args.gen_epochs, device)
    
    # Step 2: Generate synthetic data
    print(f"\n[TSTR] Step 2: 生成合成数据 (N={args.n_synthetic})", flush=True)
    X_synthetic = pipeline.generate_synthetic_data(args.n_synthetic, device)
    
    # Extract real training labels
    X_train, y_train = extract_features_and_labels(train_loader, device)
    
    # Use synthetic features with real label distribution
    y_synthetic = np.random.choice(y_train, size=args.n_synthetic, replace=True)
    
    # Step 3: Train downstream classifier
    print(f"\n[TSTR] Step 3: 训练下游分类器 (XGBoost)", flush=True)
    pipeline.train_downstream_classifier(X_synthetic, y_synthetic)
    
    # Step 4: Evaluate on real test data
    print(f"\n[TSTR] Step 4: 在真实测试集上评估", flush=True)
    X_test, y_test = extract_features_and_labels(test_loader, device)
    y_pred = pipeline.predict(X_test)
    
    # Save results
    os.makedirs('outputs/tstr_baselines', exist_ok=True)
    np.savez(f'outputs/tstr_baselines/{args.model}_seed{args.seed}_predictions.npz',
             y_pred=y_pred, y_true=y_test)
    
    pipeline.save(f'outputs/tstr_baselines/{args.model}_seed{args.seed}_pipeline.pkl')
    
    print(f"\n=== TSTR Pipeline 完成: {args.model.upper()} ===", flush=True)


if __name__ == '__main__':
    main()
