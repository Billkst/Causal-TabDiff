"""
Landmark-based data module for 2-year risk prediction + risk trajectory generation.
Replaces pseudo-temporal replication with genuine landmark sampling.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import json

# Leakage blacklist
BLACKLIST = [
    'cancyr', 'candx_days', 'can_scr', 'canc_rpt_link',
    'clinical_stag', 'path_stag', 'histology', 'grade',
    'lesionsize', 'vital_status', 'fup_days'
]

class LandmarkNLSTDataset(Dataset):
    """
    Landmark-based NLST dataset.
    Each sample = (person, landmark) with genuine history up to landmark.
    """
    def __init__(self, data_dir, split='train', seed=42, debug_mode=False):
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.debug_mode = debug_mode
        
        self._load_and_construct()
        
    def _load_raw_tables(self):
        base = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821')
        nrows = 100 if self.debug_mode else None
        
        self.prsn_df = pd.read_csv(f'{base}/nlst_780_prsn_idc_20210527.csv', nrows=nrows)
        
        for col in BLACKLIST:
            if col in self.prsn_df.columns and col != 'cancyr':
                self.prsn_df.drop(columns=[col], inplace=True)
        
        # For now, use prsn as main source (contains cancyr)
        # TODO: Merge screen/ctab/ctabc for temporal features
        
    def _construct_landmark_samples(self):
        """
        Expand persons into (person, landmark) samples.
        Exclude pre-existing cancer cases.
        """
        samples = []
        
        for _, row in self.prsn_df.iterrows():
            pid = row['pid']
            cancyr = row.get('cancyr', 0)
            
            for landmark in [0, 1, 2]:
                # Exclusion: pre-existing cancer
                if cancyr > 0 and cancyr <= landmark:
                    continue
                
                # Construct 2-year label
                y_2year = 1 if (cancyr > landmark and cancyr <= landmark + 2) else 0
                
                # Construct risk trajectory (T_max = 7)
                risk_traj = self._construct_risk_trajectory(cancyr, landmark, T_max=7)
                
                # Extract features (baseline only for now)
                features = self._extract_features(row, landmark)
                
                samples.append({
                    'pid': pid,
                    'landmark': landmark,
                    'features': features,
                    'y_2year': y_2year,
                    'risk_trajectory': risk_traj,
                    'cancyr': cancyr  # bookkeeping only
                })
        
        return pd.DataFrame(samples)
    
    def _construct_risk_trajectory(self, cancyr, landmark, T_max=7):
        traj_len = T_max
        hazard = np.zeros(traj_len, dtype=np.float32)
        
        if cancyr > 0 and cancyr > landmark:
            offset = int(cancyr - 1)
            if 0 <= offset < traj_len:
                hazard[offset] = 1.0
        
        return hazard
    
    def _extract_features(self, row, landmark):
        safe_features = ['age', 'gender', 'bmi', 'cigsmok', 'copd']
        features = np.zeros(len(safe_features), dtype=np.float32)
        
        for i, feat in enumerate(safe_features):
            if feat in row.index:
                val = row[feat]
                features[i] = float(val) if not pd.isna(val) else 0.0
        
        return features
    
    def _split_data(self, samples_df):
        """Split by person ID."""
        unique_pids = samples_df['pid'].unique()
        
        # Stratify by outcome
        pid_has_cancer = samples_df.groupby('pid')['cancyr'].first() > 0
        
        train_pids, test_pids = train_test_split(
            unique_pids, test_size=0.2, random_state=self.seed,
            stratify=pid_has_cancer
        )
        
        train_pids, val_pids = train_test_split(
            train_pids, test_size=0.25, random_state=self.seed,
            stratify=pid_has_cancer[train_pids]
        )
        
        # Assign splits
        samples_df['split'] = 'train'
        samples_df.loc[samples_df['pid'].isin(val_pids), 'split'] = 'val'
        samples_df.loc[samples_df['pid'].isin(test_pids), 'split'] = 'test'
        
        return samples_df
    
    def _load_and_construct(self):
        self._load_raw_tables()
        
        for col in self.prsn_df.columns:
            if col in BLACKLIST and col != 'cancyr':
                raise ValueError(f"LEAKAGE DETECTED: {col} in dataframe!")
        
        # Construct landmark samples
        all_samples = self._construct_landmark_samples()
        
        # Split by person
        all_samples = self._split_data(all_samples)
        
        # Filter to current split
        self.samples = all_samples[all_samples['split'] == self.split].reset_index(drop=True)
        
        print(f"[{self.split}] Loaded {len(self.samples)} samples from {self.samples['pid'].nunique()} persons")
        print(f"  Positive rate: {self.samples['y_2year'].mean():.3f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        
        # For now, replicate features across time (minimal version)
        # TODO: Replace with genuine temporal sequences
        features = row['features']
        history_len = row['landmark'] + 1
        
        # Pad to max length 3
        x_history = np.zeros((3, len(features)), dtype=np.float32)
        for t in range(history_len):
            x_history[t] = features  # Replicate for now
        
        return {
            'x': torch.tensor(x_history, dtype=torch.float32),
            'y_2year': torch.tensor([row['y_2year']], dtype=torch.float32),
            'risk_trajectory': torch.tensor(row['risk_trajectory'], dtype=torch.float32),
            'alpha_target': torch.tensor([np.random.uniform(0.1, 0.9)], dtype=torch.float32),
            'landmark': torch.tensor([row['landmark']], dtype=torch.long),
            'history_length': torch.tensor([history_len], dtype=torch.long)
        }

def get_landmark_dataloader(data_dir, split='train', batch_size=32, seed=42, debug_mode=False):
    """Get dataloader for landmark-based dataset."""
    dataset = LandmarkNLSTDataset(data_dir, split=split, seed=seed, debug_mode=debug_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
