"""
Landmark-based data module using unified_person_landmark_table from B1-1.
Implements pid-level split, missing value handling, and real short history.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import os


class LandmarkDataset(Dataset):
    """
    Dataset using unified_person_landmark_table.pkl from B1-1.
    Each sample = (person, landmark) with genuine short history.
    """
    def __init__(self, df, landmark_to_idx):
        self.df = df.reset_index(drop=True)
        self.landmark_to_idx = landmark_to_idx
        
        # Define feature columns (exclude bookkeeping and labels)
        self.baseline_cols = ['baseline_age', 'baseline_gender', 'baseline_race', 'baseline_cigsmok']
        
        # Temporal features by time point
        self.temporal_cols = {
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
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        landmark = int(row['landmark'])
        
        # Extract baseline features (always available)
        baseline = row[self.baseline_cols].values.astype(np.float32)
        
        # Extract temporal features based on landmark (real short history)
        history = []
        for t in range(landmark + 1):  # T0, T1, T2 -> 0, 1, 2
            t_key = f't{t}'
            temporal_feats = row[self.temporal_cols[t_key]].values
            
            # Convert boolean to float
            temporal_feats = np.array([float(x) if isinstance(x, (bool, np.bool_)) else x 
                                      for x in temporal_feats], dtype=np.float32)
            
            # Fill NaN with 0 (missing value strategy)
            temporal_feats = np.nan_to_num(temporal_feats, nan=0.0)
            
            # Concatenate baseline + temporal for this time point
            time_features = np.concatenate([baseline, temporal_feats])
            history.append(time_features)
        
        # Stack into (time, features) array
        x = np.stack(history, axis=0)  # Shape: (landmark+1, feature_dim)
        
        # Extract labels
        y_2year = float(row['y_2year'])
        trajectory_target = np.array(row['trajectory_target'], dtype=np.float32)
        trajectory_valid_mask = np.array(row['trajectory_valid_mask'], dtype=np.float32)
        
        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'y_2year': torch.tensor([y_2year], dtype=torch.float32),
            'trajectory_target': torch.tensor(trajectory_target, dtype=torch.float32),
            'trajectory_valid_mask': torch.tensor(trajectory_valid_mask, dtype=torch.float32),
            'landmark': torch.tensor([landmark], dtype=torch.long),
            'history_length': torch.tensor([landmark + 1], dtype=torch.long),
            'pid': int(row['pid'])
        }


def load_and_split_data(table_path, seed=42, debug_n_persons=None):
    """
    Load unified_person_landmark_table and perform pid-level split.
    
    Args:
        table_path: Path to unified_person_landmark_table.pkl
        seed: Random seed
        debug_n_persons: If set, sample N persons for debugging
    
    Returns:
        train_df, val_df, test_df, landmark_to_idx
    """
    df = pd.read_pickle(table_path)
    
    # Debug mode: sample persons
    if debug_n_persons is not None:
        unique_pids = df['pid'].unique()
        np.random.seed(seed)
        sampled_pids = np.random.choice(unique_pids, size=min(debug_n_persons, len(unique_pids)), replace=False)
        df = df[df['pid'].isin(sampled_pids)].copy()
    
    # Pid-level split
    unique_pids = df['pid'].unique()
    
    # Stratify by cancer status (any landmark has y_2year=1)
    pid_has_cancer = df.groupby('pid')['y_2year'].max()
    pid_has_cancer = pid_has_cancer.reindex(unique_pids).fillna(0).astype(int).values
    
    # Train/test split (80/20)
    train_pids, test_pids = train_test_split(
        unique_pids, 
        test_size=0.2, 
        random_state=seed,
        stratify=pid_has_cancer
    )
    
    # Train/val split (60/20 of total)
    train_pid_set = set(train_pids)
    train_cancer = np.array([pid_has_cancer[list(unique_pids).index(pid)] for pid in train_pids])
    train_pids, val_pids = train_test_split(
        train_pids,
        test_size=0.25,  # 0.25 * 0.8 = 0.2
        random_state=seed,
        stratify=train_cancer
    )
    
    # Split dataframes
    train_df = df[df['pid'].isin(train_pids)].copy()
    val_df = df[df['pid'].isin(val_pids)].copy()
    test_df = df[df['pid'].isin(test_pids)].copy()
    
    # Create landmark mapping
    landmark_to_idx = {0: 0, 1: 1, 2: 2}
    
    # Print split statistics
    print(f"\n=== Pid-Level Split Statistics ===")
    print(f"Train: {len(train_pids)} persons, {len(train_df)} samples, pos_rate={train_df['y_2year'].mean():.4f}")
    print(f"Val:   {len(val_pids)} persons, {len(val_df)} samples, pos_rate={val_df['y_2year'].mean():.4f}")
    print(f"Test:  {len(test_pids)} persons, {len(test_df)} samples, pos_rate={test_df['y_2year'].mean():.4f}")
    
    return train_df, val_df, test_df, landmark_to_idx


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads x to max_time_steps=3.
    """
    max_time = 3
    batch_size = len(batch)
    feature_dim = batch[0]['x'].shape[1]
    
    x_padded = torch.zeros(batch_size, max_time, feature_dim, dtype=torch.float32)
    y_2year = []
    trajectory_target = []
    trajectory_valid_mask = []
    landmark = []
    history_length = []
    pids = []
    
    for i, sample in enumerate(batch):
        seq_len = sample['x'].shape[0]
        x_padded[i, :seq_len, :] = sample['x']
        y_2year.append(sample['y_2year'])
        trajectory_target.append(sample['trajectory_target'])
        trajectory_valid_mask.append(sample['trajectory_valid_mask'])
        landmark.append(sample['landmark'])
        history_length.append(sample['history_length'])
        pids.append(sample['pid'])
    
    return {
        'x': x_padded,
        'y_2year': torch.stack(y_2year),
        'trajectory_target': torch.stack(trajectory_target),
        'trajectory_valid_mask': torch.stack(trajectory_valid_mask),
        'landmark': torch.stack(landmark),
        'history_length': torch.stack(history_length),
        'pid': pids
    }


def get_dataloader(table_path, split='train', batch_size=32, seed=42, debug_n_persons=None, num_workers=0):
    """
    Get dataloader for specified split.
    
    Args:
        table_path: Path to unified_person_landmark_table.pkl
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        seed: Random seed
        debug_n_persons: If set, use only N persons for debugging
        num_workers: Number of dataloader workers
    """
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed, debug_n_persons)
    
    split_map = {'train': train_df, 'val': val_df, 'test': test_df}
    df = split_map[split]
    
    dataset = LandmarkDataset(df, landmark_to_idx)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataloader
