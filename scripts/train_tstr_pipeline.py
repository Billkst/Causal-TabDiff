"""
完整 TSTR Pipeline 实现 - STaSy/TabSyn/TabDiff/TSDiff
"""
import torch
import numpy as np
import sys
import os
import time
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.tstr_pipeline import TSTRPipeline, extract_features_and_labels
from baselines.wrappers import STaSyWrapper, TabSynWrapper, TabDiffWrapper, TSDiffWrapper
import argparse


class DualLogger:
    """同时输出到终端和日志文件的日志器"""
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'w', buffering=1)  # 行缓冲
        self.terminal = sys.stdout
    
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['stasy', 'tabsyn', 'tabdiff', 'tsdiff'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen_epochs', type=int, default=50, help='生成模型训练轮数')
    parser.add_argument('--n_synthetic', type=int, default=2000, help='合成样本数量')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    # 设置日志
    log_path = f'logs/train_tstr_{args.model}_seed{args.seed}.log'
    logger = DualLogger(log_path)
    sys.stdout = logger
    
    start_time = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n=== TSTR Pipeline: {args.model.upper()} (Seed {args.seed}) ===", flush=True)
    print(f"Device: {device} | Batch Size: {args.batch_size} | Gen Epochs: {args.gen_epochs}", flush=True)
    
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
    
    # Step 2: Generate synthetic data (联合生成 X 和 Y)
    print(f"\n[TSTR] Step 2: 生成合成数据 (N={args.n_synthetic})", flush=True)
    X_synthetic, Y_synthetic = pipeline.generate_synthetic_data(args.n_synthetic, device)
    
    # Step 3: Train downstream classifier
    print(f"\n[TSTR] Step 3: 训练下游分类器 (XGBoost)", flush=True)
    pipeline.train_downstream_classifier(X_synthetic, Y_synthetic)
    
    # Step 4: Evaluate on real test data
    print(f"\n[TSTR] Step 4: 在真实测试集上评估", flush=True)
    X_test, y_test = extract_features_and_labels(test_loader, device)
    y_pred = pipeline.predict(X_test)
    
    # Save results
    os.makedirs('outputs/tstr_baselines', exist_ok=True)
    np.savez(f'outputs/tstr_baselines/{args.model}_seed{args.seed}_predictions.npz',
             y_pred=y_pred, y_true=y_test)
    
    pipeline.save(f'outputs/tstr_baselines/{args.model}_seed{args.seed}_pipeline.pkl')
    
    total_time = time.time() - start_time
    print(f"\n=== TSTR Pipeline 完成: {args.model.upper()} | Total Time: {total_time:.1f}s ===", flush=True)
    
    # 关闭日志
    sys.stdout = logger.terminal
    logger.close()


if __name__ == '__main__':
    main()
