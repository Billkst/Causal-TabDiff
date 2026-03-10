import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from src.data.data_module_landmark import get_landmark_dataloader
import logging
import os

os.makedirs('logs/evaluation', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('logs/evaluation/baselines_landmark.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def flatten_features(loader):
    X, y = [], []
    for batch in loader:
        x_flat = batch['x'][:, -1, :].numpy()
        X.append(x_flat)
        y.append(batch['y_2year'].numpy())
    return np.vstack(X), np.vstack(y).ravel()

def evaluate_model(y_true, y_pred_proba):
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {'AUROC': auroc, 'AUPRC': auprc, 'F1': f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_loader = get_landmark_dataloader(args.data_dir, 'train', 32, args.seed, args.debug_mode)
    test_loader = get_landmark_dataloader(args.data_dir, 'test', 32, args.seed, args.debug_mode)

    X_train, y_train = flatten_features(train_loader)
    X_test, y_test = flatten_features(test_loader)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Positive rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

    results = {}

    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=args.seed)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict_proba(X_test)[:, 1]
    results['LogisticRegression'] = evaluate_model(y_test, y_pred_lr)

    logger.info("Training XGBoost...")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(scale_pos_weight=pos_weight, max_depth=6, n_estimators=100, random_state=args.seed)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict_proba(X_test)[:, 1]
    results['XGBoost'] = evaluate_model(y_test, y_pred_xgb)

    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=args.seed)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    results['RandomForest'] = evaluate_model(y_test, y_pred_rf)

    logger.info("\n=== Results ===")
    for model, metrics in results.items():
        logger.info(f"{model}: AUROC={metrics['AUROC']:.3f}, AUPRC={metrics['AUPRC']:.3f}, F1={metrics['F1']:.3f}")

if __name__ == '__main__':
    main()
