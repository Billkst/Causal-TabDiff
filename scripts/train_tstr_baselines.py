import sys
import os
import numpy as np
import pickle
import argparse
import torch
from xgboost import XGBClassifier

sys.path.insert(0, 'src')
from data.data_module_landmark import load_and_split_data
from baselines import STaSyWrapper, TabSynWrapper, TabDiffWrapper, TSDiffWrapper


def prepare_flat_features(df):
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


def train_tstr_baseline(baseline_name, train_df, val_df, test_df, seed, output_dir):
    print(f"\n{'='*60}")
    print(f"TSTR Protocol: {baseline_name.upper()}")
    print(f"{'='*60}\n")
    
    print(f"⚠️  {baseline_name} TSTR implementation not yet complete")
    print(f"   Requires:")
    print(f"   1. Train generative model on train split")
    print(f"   2. Generate synthetic training set")
    print(f"   3. Train XGBoost on synthetic data")
    print(f"   4. Evaluate on real val/test")
    print(f"\n   Skipping {baseline_name} for now\n")
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, 
                       choices=['stasy', 'tabsyn', 'tabdiff', 'tsdiff'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/retained_baselines_b2')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Training {args.baseline.upper()} (TSTR) with seed {args.seed}")
    print(f"{'='*60}\n")
    
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, _ = load_and_split_data(table_path, seed=args.seed, debug_n_persons=None)
    
    print(f"Train: {len(train_df)} samples, {train_df['y_2year'].sum()} positive")
    print(f"Val:   {len(val_df)} samples, {val_df['y_2year'].sum()} positive")
    print(f"Test:  {len(test_df)} samples, {test_df['y_2year'].sum()} positive\n")
    
    result = train_tstr_baseline(args.baseline, train_df, val_df, test_df, args.seed, args.output_dir)
    
    if result is None:
        print(f"❌ {args.baseline} TSTR not implemented\n")
        with open(os.path.join(args.output_dir, f'{args.baseline}_seed{args.seed}_FAILED.txt'), 'w') as f:
            f.write(f"TSTR implementation not complete for {args.baseline}\n")


if __name__ == '__main__':
    main()
