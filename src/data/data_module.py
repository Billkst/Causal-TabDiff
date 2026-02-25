import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm
import json
import os

class NLSTDataset(Dataset):
    def __init__(self, data_dir, debug_mode=False):
        """
        Loads the NLST dataset and applies transformations anchored by dataset_metadata.json.
        """
        self.data_dir = data_dir
        self.debug_mode = debug_mode
        self._load_metadata()
        self._load_data()
        
    def _load_metadata(self):
        meta_path = os.path.join(os.path.dirname(__file__), 'dataset_metadata.json')
        if not os.path.exists(meta_path):
            import subprocess, sys
            print(f"Metadata not found at {meta_path}. Auto-generating it.")
            gen_script = os.path.join(os.path.dirname(__file__), 'generate_metadata.py')
            subprocess.run([sys.executable, gen_script], check=True)
            
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.continuous_cols = [c['name'] for c in self.metadata['continuous']]
        self.categorical_cols = [c['name'] for c in self.metadata['categorical']]
        self.y_col = self.metadata['y_col']['name']
        self.cat_meta_map = {c['name']: c for c in self.metadata['categorical']}
        
    def _load_data(self):
        prsn_path = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_prsn_idc_20210527.csv')
        screen_path = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_screen_idc_20210527.csv')
        ctab_path = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_ctab_idc_20210527.csv')
        canc_path = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_canc_idc_20210527.csv')
        
        nrows = 100 if self.debug_mode else None
        
        self.prsn_df = pd.read_csv(prsn_path, nrows=nrows)
        
        # CORE FIX: 'cancyr' (Y) and all core features (X) are natively inside 'prsn_df'.
        # We do NOT merge with canc_df or screen_df to avoid KeyErrors or data corruption.
        self.merged_df = self.prsn_df.copy()
        
        # Ensure the outcome variable is strictly integer and handles any missing values
        if self.y_col in self.merged_df.columns:
            self.merged_df[self.y_col] = self.merged_df[self.y_col].fillna(0).astype(int)
        else:
            raise KeyError(f"CRITICAL ERROR: The outcome variable '{self.y_col}' was not found in the primary prsn_df dataset!")
        
        # 3. 严格校验所有在 dataset_metadata.json 中定义的特征列 (Strict Schema Validation)
        # 完全禁止一切形式的特征造假！(Zero Tolerance for Mock Data)
        expected_cols = self.continuous_cols + self.categorical_cols + [self.metadata['alpha_col']]
        missing_cols = [col for col in expected_cols if col not in self.merged_df.columns]
        
        if missing_cols:
            raise KeyError(f"CRITICAL ERROR: The following required columns from dataset_metadata.json are MISSING in prsn_df: {missing_cols}. Please verify your CSV headers!")
            
        self._preprocess()
            
    def _gaussian_quantile_transform(self, series):
        """Eq 25: \tilde{x}_{num} = \Phi^{-1}(F_{emp}(x_{num}))"""
        s_noise = series + np.random.normal(0, 1e-6, len(series))
        ranks = s_noise.rank(method='average')
        uniform = (ranks - 0.5) / len(ranks) 
        return norm.ppf(uniform)
        
    def _analog_bits_encode(self, series, col_name):
        """Eq 26: x_{analog} = 2b - 1 \in \{-1, 1\}^m mapped via static metadata."""
        meta = self.cat_meta_map[col_name]
        m_bits = meta['analog_bits']
        val_to_idx = meta['val_to_idx']
        
        encoded = []
        for val in series:
            # Map raw value to 0..K-1 index based on immutable JSON schema
            idx_str = str(int(val)) if not np.isnan(val) else "0"
            idx_int = int(val_to_idx.get(idx_str, 0))
            
            b_str = format(idx_int, f'0{m_bits}b')
            b_vec = np.array([int(c) for c in b_str])
            analog_vec = 2.0 * b_vec - 1.0
            encoded.append(analog_vec)
        return np.vstack(encoded)

    def _raw_cat_encode(self, series, col_name):
        """Extracts exact categorical [0, K-1] indices directly from JSON metadata mapping."""
        meta = self.cat_meta_map[col_name]
        val_to_idx = meta['val_to_idx']
        
        encoded = []
        for val in series:
            idx_str = str(int(val)) if not np.isnan(val) else "0"
            idx_int = int(val_to_idx.get(idx_str, 0))
            encoded.append([idx_int])
        return np.vstack(encoded)

    def _preprocess(self):
        self.num_samples = len(self.merged_df)
        self.T = 3 # T0, T1, T2
        
        # Determine total dimensions based on original column sequence
        D_orig = sum([c['dim'] for c in self.metadata['columns']])
        num_cats = len([c for c in self.metadata['columns'] if c['type'] == 'categorical'])
        
        self.X_input = np.zeros((self.num_samples, self.T, D_orig))
        self.X_cat_raw = np.zeros((self.num_samples, self.T, num_cats))
        
        feature_offset = 0
        cat_idx = 0
        
        for t in range(self.T):
            feature_offset = 0
            cat_idx = 0
            for col_meta in self.metadata['columns']:
                col = col_meta['name']
                col_type = col_meta['type']
                col_dim = col_meta['dim']
                
                if col_type == 'continuous':
                    transformed = self._gaussian_quantile_transform(self.merged_df[col])
                    self.X_input[:, t, feature_offset : feature_offset + col_dim] = transformed.reshape(-1, 1)
                else:
                    analog_feat = self._analog_bits_encode(self.merged_df[col], col)
                    raw_feat = self._raw_cat_encode(self.merged_df[col], col)
                    
                    self.X_input[:, t, feature_offset : feature_offset + col_dim] = analog_feat
                    self.X_cat_raw[:, t, cat_idx : cat_idx + 1] = raw_feat
                    cat_idx += 1
                
                feature_offset += col_dim
             
        # Generate dummy causal conditions (alpha target representing environmental exposure)
        # We will use this as our Treatment T
        self.alpha_target = np.random.uniform(0.1, 0.9, size=(self.num_samples, 1))
        
        # Ground truth Y (Outcome)
        self.y = self.merged_df[self.y_col].values.reshape(-1, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X_input[idx], dtype=torch.float32),
            'x_cat_raw': torch.tensor(self.X_cat_raw[idx], dtype=torch.long),
            'alpha_target': torch.tensor(self.alpha_target[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.float32)
        }

def get_dataloader(data_dir, batch_size=32, debug_mode=False):
    dataset = NLSTDataset(data_dir, debug_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
