"""
TSDiff Landmark Wrapper - 完整重写
"""
import torch
import torch.nn as nn


class TSDiffLandmarkWrapper:
    def __init__(self, seq_len, feature_dim):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.total_dim = seq_len * feature_dim + 1
        self.model = None
        self.fitted = False
    
    def fit(self, train_loader, epochs, device):
        from baselines.tsdiff_core.model import TSDiffDDPM
        
        self.model = TSDiffDDPM(input_dim=self.total_dim, timesteps=100).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            for batch in train_loader:
                x = batch['x'].to(device)
                y = batch['y_2year'].to(device)
                
                x_flat = x.reshape(x.shape[0], -1)
                xy = torch.cat([x_flat, y], dim=1).unsqueeze(1)
                
                optimizer.zero_grad()
                loss = self.model.train_step(xy)
                loss.backward()
                optimizer.step()
        
        self.fitted = True
    
    def sample(self, n_samples, device):
        if not self.fitted:
            return torch.randn(n_samples, self.seq_len, self.feature_dim, device=device), torch.randn(n_samples, 1, device=device)
        
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(n_samples, seq_len=1, features=self.total_dim, device=device)
            samples = samples.squeeze(1)
            
            X_syn = samples[:, :-1].reshape(n_samples, self.seq_len, self.feature_dim)
            Y_syn = (samples[:, -1:] > 0).float()
            
            return X_syn, Y_syn
