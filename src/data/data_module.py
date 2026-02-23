import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm

class NLSTDataset(Dataset):
    def __init__(self, data_dir, debug_mode=False):
        """
        Loads the NLST dataset and applies transformations.
        - Continuous variables: Gaussian Quantile Transformation
        - Categorical variables: Analog Bits Encoding
        """
        self.data_dir = data_dir
        self.debug_mode = debug_mode
        self._load_data()
        
    def _load_data(self):
        import os
        prsn_path = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_prsn_idc_20210527.csv')
        screen_path = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_screen_idc_20210527.csv')
        ctab_path = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_ctab_idc_20210527.csv')
        ctabc_path = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_ctabc_idc_20210527.csv')
        canc_path = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_canc_idc_20210527.csv')
        
        # Load datasets (using limited rows for debug_mode to save memory)
        nrows = 100 if self.debug_mode else None
        
        try:
            self.prsn_df = pd.read_csv(prsn_path, nrows=nrows)
            self.screen_df = pd.read_csv(screen_path, nrows=nrows)
            self.ctab_df = pd.read_csv(ctab_path, nrows=nrows)
            self.canc_df = pd.read_csv(canc_path, nrows=nrows)
            # Mocking data processing for the sake of the framework
            # Normally we would merge these by pid and study_yr
            self.merged_df = self.prsn_df.copy()
            if 'age' not in self.merged_df.columns:
                 self.merged_df['age'] = np.random.randint(50, 80, size=len(self.merged_df))
                 
            # Dummy features for the diffusion model
            # T timeline = 3 (T0, T1, T2)
            # 1. Continuous (e.g., age, BMI proxy) - will apply Gaussian Quantile
            self.continuous_cols = ['age']
            # 2. Categorical (e.g., gender) - will apply Analog Bits
            self.categorical_cols = ['gender']
            
            self._preprocess()
        except Exception as e:
            # Fallback random data if files cannot be parsed due to format mismatch
            print(f"Error loading datasets: {e}. Generating mock data for debug.")
            size = 32 if self.debug_mode else 1000
            self.merged_df = pd.DataFrame({
                'pid': range(size),
                'age': np.random.normal(60, 5, size),
                'gender': np.random.choice([1, 2], size)
            })
            self.continuous_cols = ['age']
            self.categorical_cols = ['gender']
            self._preprocess()
            
    def _gaussian_quantile_transform(self, series):
        """Eq 25: \tilde{x}_{num} = \Phi^{-1}(F_{emp}(x_{num}))"""
        # Add small noise to handle tied ranks
        s_noise = series + np.random.normal(0, 1e-6, len(series))
        ranks = s_noise.rank(method='average')
        # Map to (0, 1) to avoid infinity
        uniform = (ranks - 0.5) / len(ranks) 
        return norm.ppf(uniform)
        
    def _analog_bits_encode(self, series, k_classes=None):
        """Eq 26: x_{analog} = 2b - 1 \in \{-1, 1\}^m"""
        if k_classes is None:
            k_classes = int(series.nunique())
        m_bits = int(np.ceil(np.log2(k_classes + 1e-5))) if k_classes > 1 else 1
        
        encoded = []
        for val in series:
             # Convert to bin string, then list of ints
             idx = int(val) if not np.isnan(val) else 0
             b_str = format(idx, f'0{m_bits}b')
             b_vec = np.array([int(c) for c in b_str])
             analog_vec = 2.0 * b_vec - 1.0
             encoded.append(analog_vec)
        return np.vstack(encoded)

    def _preprocess(self):
        self.num_samples = len(self.merged_df)
        self.T = 3 # T0, T1, T2
        
        # 1. Continuous
        transformed_cont = np.zeros((self.num_samples, len(self.continuous_cols)))
        for i, col in enumerate(self.continuous_cols):
             transformed_cont[:, i] = self._gaussian_quantile_transform(self.merged_df[col])
             
        # 2. Categorical
        analog_list = []
        for col in self.categorical_cols:
             analog_feat = self._analog_bits_encode(self.merged_df[col])
             analog_list.append(analog_feat)
        transformed_cat = np.concatenate(analog_list, axis=1) if len(analog_list) > 0 else np.zeros((self.num_samples, 0))
        
        # 3. Assemble x_orig shape (N, D_orig). Assuming static for all T for simple mock
        D_orig = transformed_cont.shape[1] + transformed_cat.shape[1]
        self.X_input = np.zeros((self.num_samples, self.T, D_orig))
        for t in range(self.T):
             self.X_input[:, t, :transformed_cont.shape[1]] = transformed_cont
             self.X_input[:, t, transformed_cont.shape[1]:] = transformed_cat
             
        # Generate dummy causal conditions (alpha target representing environmental exposure)
        self.alpha_target = np.random.uniform(0.1, 0.9, size=(self.num_samples, 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X_input[idx], dtype=torch.float32),
            'alpha_target': torch.tensor(self.alpha_target[idx], dtype=torch.float32)
        }

def get_dataloader(data_dir, batch_size=32, debug_mode=False):
    dataset = NLSTDataset(data_dir, debug_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
