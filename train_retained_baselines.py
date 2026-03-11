import sys
import os
import numpy as np
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.path.insert(0, 'src')
from data.data_module_landmark import load_and_split_data


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


def train_causal_forest(X_train, y_train, seed):
    from econml.dml import CausalForestDML
    from sklearn.linear_model import Ridge
    
    T_train = np.random.rand(len(y_train))
    W_train = np.ones((len(y_train), 1))
    
    model = CausalForestDML(
        model_y=Ridge(),
        model_t=Ridge(),
        discrete_treatment=False,
        n_estimators=100,
        random_state=seed
    )
    
    XY_train = np.column_stack([X_train, y_train])
    model.fit(Y=XY_train, T=T_train, X=W_train, cache_values=True)
    
    return model


def train_generative_baseline(baseline_name, X_train, y_train, seed):
    print(f"⚠️  {baseline_name} is a generative model - not yet adapted for discriminative task")
    print(f"   Skipping {baseline_name} in this phase")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, 
                       choices=['causal_forest', 'stasy', 'tabsyn', 'tabdiff', 'tsdiff'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/retained_baselines_b2')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Training {args.baseline.upper()} with seed {args.seed}")
    print(f"{'='*60}\n")
    
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, _ = load_and_split_data(table_path, seed=args.seed, debug_n_persons=None)
    
    print(f"Train: {len(train_df)} samples, {train_df['y_2year'].sum()} positive")
    print(f"Val:   {len(val_df)} samples, {val_df['y_2year'].sum()} positive")
    print(f"Test:  {len(test_df)} samples, {test_df['y_2year'].sum()} positive\n")
    
    X_train, y_train = prepare_flat_features(train_df)
    X_val, y_val = prepare_flat_features(val_df)
    X_test, y_test = prepare_flat_features(test_df)
    
    print(f"Feature dimension: {X_train.shape[1]}\n")
    
    if args.baseline == 'causal_forest':
        model = train_causal_forest(X_train, y_train, args.seed)
        
        if model is None:
            print(f"❌ {args.baseline} training failed\n")
            return
        
        print("⚠️  CausalForest prediction not yet implemented for discriminative task")
        print("   Skipping prediction export\n")
        return
    
    else:
        model = train_generative_baseline(args.baseline, X_train, y_train, args.seed)
        
        if model is None:
            return


if __name__ == '__main__':
    main()
