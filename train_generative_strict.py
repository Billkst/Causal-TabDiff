import torch
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from evaluation.efficiency import EfficiencyTracker
from xgboost import XGBClassifier


def extract_xy(loader):
    x_list, y_list = [], []
    for batch in loader:
        x_list.append(batch['x'].cpu().numpy())
        y_list.append(batch['y_2year'].cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0).flatten()
    return x.reshape(x.shape[0], -1), y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['tabsyn', 'tabdiff', 'survtraj', 'sssd', 'stasy', 'tsdiff'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n_synthetic', type=int, default=-1, help='合成样本数量 (-1=与训练集等规模)')
    parser.add_argument('--output_dir', type=str, default='outputs/b2_baseline/tstr')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tracker = EfficiencyTracker()
    
    print(f"\n=== 训练 {args.model.upper()}_strict (Seed {args.seed}) ===", flush=True)
    
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=64, num_workers=4)

    # Auto-compute synthetic sample size to match training set
    if args.n_synthetic <= 0:
        args.n_synthetic = len(train_df)
        print(f"[TSTR] Auto synthetic sample size = {args.n_synthetic} (matching train set)", flush=True)
    
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
    elif args.model == 'stasy':
        from baselines.stasy_landmark_v2 import STaSyLandmarkWrapper
        model = STaSyLandmarkWrapper(seq_len=3, feature_dim=feature_dim)
    elif args.model == 'tsdiff':
        from baselines.tsdiff_landmark_wrapper import TSDiffLandmarkWrapper
        model = TSDiffLandmarkWrapper(seq_len=3, feature_dim=feature_dim)
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    print(f"[TSTR] 训练生成模型 ({args.epochs} epochs)...", flush=True)
    with tracker.track_training():
        model.fit(train_loader, args.epochs, device)

    if hasattr(model, 'model') and getattr(model, 'model') is not None:
        tracker.set_model_size(model.model)
    elif hasattr(model, 'vae_model') and getattr(model, 'vae_model') is not None:
        tracker.set_model_size(model.vae_model)
    elif hasattr(model, 'vae') and getattr(model, 'vae') is not None:
        tracker.set_model_size(model.vae)
    elif hasattr(model, 'diffusion_model') and getattr(model, 'diffusion_model') is not None:
        tracker.set_model_size(model.diffusion_model)
    
    SAMPLE_BATCH = 2048
    n_total = args.n_synthetic
    X_parts, Y_parts = [], []
    n_done = 0
    print(f"[TSTR] 生成 {n_total} 合成样本 (batch={SAMPLE_BATCH})...", flush=True)
    while n_done < n_total:
        chunk = min(SAMPLE_BATCH, n_total - n_done)
        torch.cuda.empty_cache()
        sample_out = model.sample(chunk, device)
        if isinstance(sample_out, tuple) and len(sample_out) >= 2:
            X_parts.append(sample_out[0].cpu())
            Y_parts.append(sample_out[1].cpu())
        else:
            raise ValueError('sample() must return at least (X_syn, Y_syn)')
        n_done += chunk
        print(f"  生成进度: {n_done}/{n_total}", flush=True)

    X_syn = torch.cat(X_parts, dim=0)
    Y_syn = torch.cat(Y_parts, dim=0)

    X_flat = X_syn.numpy().reshape(X_syn.shape[0], -1)
    y_syn = Y_syn.numpy().flatten()
    print(
        f"Synthetic label stats | unique={np.unique(y_syn)} | pos_rate={float(np.mean(y_syn)):.4f}",
        flush=True,
    )
    if np.unique(y_syn).size < 2:
        os.makedirs(args.output_dir, exist_ok=True)
        failure_path = os.path.join(args.output_dir, f'{args.model}_seed{args.seed}_FAILED.txt')
        with open(failure_path, 'w') as f:
            f.write('synthetic labels collapsed to a single class\n')
        print(f"[FAIL] {args.model} seed {args.seed}: synthetic labels collapsed to one class", flush=True)
        return
    pos_count = float(np.sum(y_syn == 1))
    neg_count = float(np.sum(y_syn == 0))
    scale_pos_weight = neg_count / max(1.0, pos_count)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
    )
    print(f"[TSTR] 训练 XGBoost 下游分类器...", flush=True)
    clf.fit(X_flat, y_syn)

    print(f"[TSTR] 在真实验证/测试集上评估...", flush=True)
    X_val_flat, y_val = extract_xy(val_loader)
    X_test_flat, y_test = extract_xy(test_loader)
    val_pred = clf.predict_proba(X_val_flat)
    with tracker.track_inference(len(y_test)):
        test_pred = clf.predict_proba(X_test_flat)
    val_pred = val_pred[:, 1] if val_pred.shape[1] > 1 else val_pred[:, 0]
    test_pred = test_pred[:, 1] if test_pred.shape[1] > 1 else test_pred[:, 0]
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(f'{args.output_dir}/{args.model}_seed{args.seed}_predictions.npz',
             val_y_true=y_val, val_y_pred=val_pred,
             test_y_true=y_test, test_y_pred=test_pred,
             synthetic_sample_size=args.n_synthetic)
    
    tracker.save_json(f'{args.output_dir}/{args.model}_efficiency_seed{args.seed}.json')
    
    print(f"\n=== {args.model.upper()}_strict 完成 ===", flush=True)

if __name__ == '__main__':
    main()
