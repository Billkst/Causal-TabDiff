import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pickle
import os


class LandmarkDatasetCached(Dataset):
    def __init__(self, df, landmark_to_idx):
        self.landmark_to_idx = landmark_to_idx
        
        baseline_cols = ['baseline_age', 'baseline_gender', 'baseline_race', 'baseline_cigsmok']
        treatment_col = os.environ.get('ALPHA_TREATMENT_COLUMN') or None
        baseline_cols = [c for c in baseline_cols if c != treatment_col]
        temporal_cols = {
            't0': [
                'screen_t0_ctdxqual', 'screen_t0_kvp', 'screen_t0_ma', 'screen_t0_fov',
                'abn_t0_count', 'abn_t0_max_long_dia', 'abn_t0_max_perp_dia', 'abn_t0_has_spiculated',
                'change_t0_has_growth', 'change_t0_has_attn_change', 'change_t0_change_count'
            ],
            't1': [
                'screen_t1_ctdxqual', 'screen_t1_kvp', 'screen_t1_ma', 'screen_t1_fov',
                'abn_t1_count', 'abn_t1_max_long_dia', 'abn_t1_max_perp_dia', 'abn_t1_has_spiculated',
                'change_t1_has_growth', 'change_t1_has_attn_change', 'change_t1_change_count'
            ],
            't2': [
                'screen_t2_ctdxqual', 'screen_t2_kvp', 'screen_t2_ma', 'screen_t2_fov',
                'abn_t2_count', 'abn_t2_max_long_dia', 'abn_t2_max_perp_dia', 'abn_t2_has_spiculated',
                'change_t2_has_growth', 'change_t2_has_attn_change', 'change_t2_change_count'
            ]
        }
        
        print(f"Caching {len(df)} samples...")
        self.cached_data = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            landmark = int(row['landmark'])
            
            baseline = row[baseline_cols].values.astype(np.float32)
            
            history = []
            for t in range(landmark + 1):
                t_key = f't{t}'
                temporal_feats = row[temporal_cols[t_key]].values
                temporal_feats = np.array([float(x) if isinstance(x, (bool, np.bool_)) else x 
                                          for x in temporal_feats], dtype=np.float32)
                temporal_feats = np.nan_to_num(temporal_feats, nan=0.0)
                time_features = np.concatenate([baseline, temporal_feats])
                history.append(time_features)
            
            x = np.stack(history, axis=0)
            y_2year = float(row['y_2year'])
            trajectory_target = np.array(row['trajectory_target'], dtype=np.float32)
            trajectory_valid_mask = np.array(row['trajectory_valid_mask'], dtype=np.float32)
            
            self.cached_data.append({
                'x': torch.tensor(x, dtype=torch.float32),
                'y_2year': torch.tensor([y_2year], dtype=torch.float32),
                'trajectory_target': torch.tensor(trajectory_target, dtype=torch.float32),
                'trajectory_valid_mask': torch.tensor(trajectory_valid_mask, dtype=torch.float32),
                'landmark': torch.tensor([landmark], dtype=torch.long),
                'history_length': torch.tensor([landmark + 1], dtype=torch.long),
                'pid': int(row['pid']),
                **({'alpha_target': torch.tensor([float(row[treatment_col])], dtype=torch.float32)} if treatment_col else {})
            })
        
        print(f"Caching complete.")
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]


def load_and_split_data(table_path, seed=42, debug_n_persons=None):
    with open(table_path, 'rb') as f:
        df = pickle.load(f)
    
    if debug_n_persons:
        unique_pids = df['pid'].unique()
        sampled_pids = np.random.RandomState(seed).choice(unique_pids, size=debug_n_persons, replace=False)
        df = df[df['pid'].isin(sampled_pids)].copy()
    
    unique_pids = df['pid'].unique()
    pid_has_cancer = df.groupby('pid')['y_2year'].max().to_dict()
    pid_labels = np.array([pid_has_cancer[pid] for pid in unique_pids])
    
    train_pids, temp_pids = train_test_split(
        unique_pids, test_size=0.4, random_state=seed, stratify=pid_labels
    )
    temp_labels = np.array([pid_has_cancer[pid] for pid in temp_pids])
    val_pids, test_pids = train_test_split(
        temp_pids, test_size=0.5, random_state=seed, stratify=temp_labels
    )
    
    train_df = df[df['pid'].isin(train_pids)].copy()
    val_df = df[df['pid'].isin(val_pids)].copy()
    test_df = df[df['pid'].isin(test_pids)].copy()
    
    landmark_to_idx = {0: 0, 1: 1, 2: 2}
    
    print("\n=== Pid-Level Split Statistics ===")
    print(f"Train: {len(train_pids)} persons, {len(train_df)} samples, pos_rate={train_df['y_2year'].mean():.4f}")
    print(f"Val:   {len(val_pids)} persons, {len(val_df)} samples, pos_rate={val_df['y_2year'].mean():.4f}")
    print(f"Test:  {len(test_pids)} persons, {len(test_df)} samples, pos_rate={test_df['y_2year'].mean():.4f}")
    
    return train_df, val_df, test_df, landmark_to_idx


def collate_fn(batch):
    max_time = max(item['x'].shape[0] for item in batch)
    
    x_padded = []
    for item in batch:
        x = item['x']
        if x.shape[0] < max_time:
            pad = torch.zeros(max_time - x.shape[0], x.shape[1], dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        x_padded.append(x)
    
    return {
        'x': torch.stack(x_padded),
        'y_2year': torch.stack([item['y_2year'] for item in batch]),
        'trajectory_target': torch.stack([item['trajectory_target'] for item in batch]),
        'trajectory_valid_mask': torch.stack([item['trajectory_valid_mask'] for item in batch]),
        'landmark': torch.stack([item['landmark'] for item in batch]),
        'history_length': torch.stack([item['history_length'] for item in batch]),
        'alpha_target': (torch.stack([item['alpha_target'] for item in batch]) if 'alpha_target' in batch[0] else torch.stack([item['landmark'] for item in batch]).float()),
    }
