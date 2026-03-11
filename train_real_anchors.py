"""
B2-1: Real-data anchors training script.
Trains Logistic Regression, XGBoost, and Balanced Random Forest on full dataset.
"""
import sys
import os
import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, 'src')
from data.data_module_landmark import load_and_split_data


def prepare_flat_features(df):
    """将 landmark 数据展平为固定维度特征向量"""
    baseline_cols = ['baseline_age', 'baseline_gender', 'baseline_race', 'baseline_cigsmok']
    
    temporal_cols = {
        't0': ['screen_t0_ctdxqual', 'screen_t0_kvp', 'screen_t0_ma', 'screen_t0_fov',
               'abn_t0_count', 'abn_t0_max_long_dia', 'abn_t0_max_perp_dia', 'abn_t0_has_spiculated',
               'change_t0_has_growth', 'change_t0_has_attn_change', 'change_t0_change_count'],
        't1': ['screen_t1_ctdxqual', 'screen_t1_kvp', 'screen_t1_ma', 'screen_t1_fov',
               'abn_t1_count', 'abn_t1_max_long_dia', 'abn_t1_max_perp_dia', 'abn_t1_has_spiculated',
               'change_t1_has_growth', 'change_t1_has_attn_change', 'change_t1_change_count'],
        't2': ['screen_t2_ctdxqual', 'screen_t2_kvp', 'screen_t2_ma', 'screen_t2_fov',
               'abn_t2_count', 'abn_t2_max_long_dia', 'abn_t2_max_perp_dia', 'abn_t2_has_spiculated',
               'change_t2_has_growth', 'change_t2_has_attn_change', 'change_t2_change_count']
    }
    
    all_feature_cols = baseline_cols + temporal_cols['t0'] + temporal_cols['t1'] + temporal_cols['t2']
    
    X = df[all_feature_cols].fillna(0).values.astype(np.float32)
    y = df['y_2year'].values.astype(np.int32)
    
    return X, y


def train_logistic_regression(X_train, y_train, seed):
    """训练 Logistic Regression with class weights"""
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        class_weight=class_weight_dict,
        max_iter=1000,
        random_state=seed,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def train_xgboost(X_train, y_train, seed):
    """训练 XGBoost with scale_pos_weight"""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=seed,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    
    return model


def train_balanced_rf(X_train, y_train, seed):
    """训练 Balanced Random Forest"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['lr', 'xgb', 'brf'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/real_anchors')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} with seed {args.seed}")
    print(f"{'='*60}\n")
    
    # Load data
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, _ = load_and_split_data(table_path, seed=args.seed, debug_n_persons=None)
    
    print(f"Train: {len(train_df)} samples, {train_df['y_2year'].sum()} positive")
    print(f"Val:   {len(val_df)} samples, {val_df['y_2year'].sum()} positive")
    print(f"Test:  {len(test_df)} samples, {test_df['y_2year'].sum()} positive\n")
    
    # Prepare features
    X_train, y_train = prepare_flat_features(train_df)
    X_val, y_val = prepare_flat_features(val_df)
    X_test, y_test = prepare_flat_features(test_df)
    
    print(f"Feature dimension: {X_train.shape[1]}\n")
    
    # Train model
    if args.model == 'lr':
        model, scaler = train_logistic_regression(X_train, y_train, args.seed)
        X_val_proc = scaler.transform(X_val)
        X_test_proc = scaler.transform(X_test)
    elif args.model == 'xgb':
        model = train_xgboost(X_train, y_train, args.seed)
        X_val_proc = X_val
        X_test_proc = X_test
        scaler = None
    else:  # brf
        model = train_balanced_rf(X_train, y_train, args.seed)
        X_val_proc = X_val
        X_test_proc = X_test
        scaler = None
    
    print("Training completed.\n")
    
    # Predict
    if args.model == 'lr':
        val_pred_proba = model.predict_proba(X_val_proc)[:, 1]
        test_pred_proba = model.predict_proba(X_test_proc)[:, 1]
    else:
        val_pred_proba = model.predict_proba(X_val_proc)[:, 1]
        test_pred_proba = model.predict_proba(X_test_proc)[:, 1]
    
    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    pred_file = os.path.join(args.output_dir, f'{args.model}_seed{args.seed}_predictions.npz')
    
    np.savez(
        pred_file,
        val_y_true=y_val,
        val_y_pred=val_pred_proba,
        test_y_true=y_test,
        test_y_pred=test_pred_proba
    )
    
    print(f"Predictions saved to: {pred_file}")
    
    # Save model
    model_file = os.path.join(args.output_dir, f'{args.model}_seed{args.seed}_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print(f"Model saved to: {model_file}\n")


if __name__ == '__main__':
    main()
