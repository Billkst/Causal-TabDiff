"""
TSDiff Landmark Wrapper - 最小化重写版本
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm


class TSDiffLandmarkWrapper:
    def __init__(self, seq_len, feature_dim):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.fitted = False
        self.model = None
    
    def fit(self, train_loader, epochs, device):
        from baselines.tsdiff_core.model import TSDiffDDPM
        
        total_dim = self.seq_len * self.feature_dim + 1
        self.model = TSDiffDDPM(input_dim=total_dim, timesteps=100).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            batch_count = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100, file=sys.stderr):
                x = batch['x'].to(device)
                y = batch['y_2year'].to(device)
                
                x_flat = x.reshape(x.shape[0], -1)
                xy = torch.cat([x_flat, y], dim=1).unsqueeze(1)
                
                optimizer.zero_grad()
                loss = self.model.train_step(xy)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().item())
                batch_count += 1
            avg_loss = epoch_loss / max(batch_count, 1)
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{epochs} | Loss {avg_loss:.4f} | Time {elapsed:.1f}s", flush=True)
        
        self.fitted = True
    
    def sample(self, n_samples, device):
        if not self.fitted:
            return torch.randn(n_samples, self.seq_len, self.feature_dim, device=device), torch.randn(n_samples, 1, device=device)
        
        self.model.eval()
        samples = self.model.sample(n_samples, seq_len=1, features=self.seq_len * self.feature_dim + 1, device=device)
        samples = samples.squeeze(1)
        
        X_syn = samples[:, :-1].reshape(n_samples, self.seq_len, self.feature_dim)
        Y_syn = samples[:, -1:]
        
        return X_syn, Y_syn
    
    def predict(self, df):
        """Generate predictions for layer1 2-year risk"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        n_samples = len(df)
        device = next(self.model.parameters()).device
        
        # Generate samples
        X_syn, Y_syn = self.sample(n_samples, device)
        
        # Return predictions as probabilities
        return torch.sigmoid(Y_syn).cpu().numpy().flatten()
    
    def predict(self, df):
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        n_samples = len(df)
        device = next(self.model.parameters()).device
        X_syn, Y_syn = self.sample(n_samples, device)
        return torch.sigmoid(Y_syn).cpu().numpy().flatten()
