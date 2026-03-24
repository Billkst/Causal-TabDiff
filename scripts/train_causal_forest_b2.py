import sys
import os
import numpy as np
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, 'src')
from data.data_module_landmark import load_and_split_data
from evaluation.efficiency import EfficiencyTracker


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
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=class_weight_dict,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    return model


def main():
     parser = argparse.ArgumentParser()
     parser.add_argument('--seed', type=int, required=True)
     parser.add_argument('--output_dir', type=str, default='outputs/retained_baselines_b2')
     args = parser.parse_args()
     
     tracker = EfficiencyTracker()
     
     print(f"\n{'='*60}")
     print(f"Training CausalForest with seed {args.seed}")
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
     
     with tracker.track_training():
          model = train_causal_forest(X_train, y_train, args.seed)
    
     print("Training completed.\n")
     
     tracker.set_model_size(model)
     
     val_pred_proba = model.predict_proba(X_val)[:, 1]
     with tracker.track_inference(len(y_test)):
          test_pred_proba = model.predict_proba(X_test)[:, 1]
     
     os.makedirs(args.output_dir, exist_ok=True)
     pred_file = os.path.join(args.output_dir, f'causal_forest_seed{args.seed}_predictions.npz')
     
     np.savez(
         pred_file,
         val_y_true=y_val,
         val_y_pred=val_pred_proba,
         test_y_true=y_test,
         test_y_pred=test_pred_proba
     )
     
     print(f"Predictions saved to: {pred_file}")
     
     model_file = os.path.join(args.output_dir, f'causal_forest_seed{args.seed}_model.pkl')
     with open(model_file, 'wb') as f:
         pickle.dump(model, f)
     
     print(f"Model saved to: {model_file}\n")
     
     efficiency_file = os.path.join(args.output_dir, f'causal_forest_efficiency_seed{args.seed}.json')
     tracker.save_json(efficiency_file)


if __name__ == '__main__':
    main()
