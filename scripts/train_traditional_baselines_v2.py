#!/usr/bin/env python3
"""
传统 ML Baseline 训练脚本 - 符合 submission_v3 协议
支持: Logistic Regression, Random Forest, XGBoost
"""
import argparse
import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.insert(0, 'src')
from data.data_module_landmark import load_and_split_data
from evaluation.metrics import find_optimal_threshold, compute_all_metrics

def extract_features_from_df(df):
    """直接从 DataFrame 提取特征，避免 DataLoader 迭代问题"""
    X = np.stack([np.stack(seq)[-1] for seq in df['feature_seq'].values])
    y = df['y_2year'].values
    return X, y

def train_and_evaluate(model_name, model, train_df, val_df, test_df, seed, output_dir):
    print(f"\n{'='*60}", flush=True)
    print(f"Training {model_name} (seed={seed})", flush=True)
    print(f"{'='*60}", flush=True)
    
    print("Extracting features...", flush=True)
    X_train, y_train = extract_features_from_df(train_df)
    X_val, y_val = extract_features_from_df(val_df)
    X_test, y_test = extract_features_from_df(test_df)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)
    print(f"Positive rate - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}, Test: {y_test.mean():.4f}", flush=True)
    
    print(f"Fitting {model_name}...", flush=True)
    model.fit(X_train, y_train)
    
    print("Predicting...", flush=True)
    val_y_pred = model.predict_proba(X_val)[:, 1]
    test_y_pred = model.predict_proba(X_test)[:, 1]
    
    print("Computing optimal threshold...", flush=True)
    optimal_threshold, val_f1 = find_optimal_threshold(y_val, val_y_pred, metric='f1')
    print(f"Optimal threshold: {optimal_threshold:.6f} (Val F1: {val_f1:.6f})", flush=True)
    
    print("Computing test metrics...", flush=True)
    metrics = compute_all_metrics(y_test, test_y_pred, threshold=optimal_threshold)
    print(f"Test AUPRC: {metrics['auprc']:.6f} | AUROC: {metrics['auroc']:.6f} | F1: {metrics['f1']:.6f}", flush=True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_seed{seed}_official.npz")
    np.savez(
        output_file,
        val_y_true=y_val,
        val_y_pred=val_y_pred,
        test_y_true=y_test,
        test_y_pred=test_y_pred,
        optimal_threshold=optimal_threshold,
        test_f1=metrics['f1'],
        test_precision=metrics['precision'],
        test_recall=metrics['recall'],
        test_specificity=metrics['specificity'],
        test_balanced_accuracy=metrics['balanced_accuracy'],
        test_mcc=metrics['mcc'],
        seed=seed
    )
    print(f"Saved: {output_file}", flush=True)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_path', type=str, default='data/landmark_tables/unified_person_landmark_table.pkl')
    parser.add_argument('--output_dir', type=str, default='outputs/submission_v3/traditional_baselines')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', type=str, default='lr,rf,xgb', help='Comma-separated: lr,rf,xgb')
    args = parser.parse_args()
    
    print(f"Loading data from {args.table_path}...", flush=True)
    train_df, val_df, test_df, _ = load_and_split_data(args.table_path, seed=args.seed)
    
    models_to_run = args.models.lower().split(',')
    
    if 'lr' in models_to_run:
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=args.seed)
        train_and_evaluate('LogisticRegression', lr, train_df, val_df, test_df, args.seed, args.output_dir)
    
    if 'rf' in models_to_run:
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=args.seed, n_jobs=-1)
        train_and_evaluate('RandomForest', rf, train_df, val_df, test_df, args.seed, args.output_dir)
    
    if 'xgb' in models_to_run:
        _, y_train = extract_features_from_df(train_df)
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(scale_pos_weight=pos_weight, max_depth=6, n_estimators=100, random_state=args.seed, n_jobs=-1)
        train_and_evaluate('XGBoost', xgb, train_df, val_df, test_df, args.seed, args.output_dir)
    
    print(f"\n{'='*60}", flush=True)
    print("All models completed!", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == '__main__':
    main()
