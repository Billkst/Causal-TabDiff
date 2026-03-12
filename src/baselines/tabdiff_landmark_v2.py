"""
TabDiff Landmark Wrapper - 最小化重写
"""
import torch
import torch.nn as nn


class TabDiffLandmarkWrapper:
    def __init__(self, seq_len, feature_dim):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.total_dim = seq_len * feature_dim + 1
        self.model = None
        self.fitted = False
    
    def fit(self, train_loader, epochs, device):
        # TabDiff 使用连续时间扩散
        # 简化为标准 DDPM
        
        class SimpleDiffusion(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim + 1, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim)
                )
            
            def forward(self, x, t):
                t_embed = t.unsqueeze(-1)
                xt = torch.cat([x, t_embed], dim=-1)
                return self.net(xt)
        
        self.model = SimpleDiffusion(self.total_dim).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        timesteps = 100
        
        for epoch in range(epochs):
            for batch in train_loader:
                x = batch['x'].to(device)
                y = batch['y_2year'].to(device)
                
                x_flat = x.reshape(x.shape[0], -1)
                xy = torch.cat([x_flat, y], dim=1)
                
                t = torch.randint(0, timesteps, (xy.shape[0],), device=device).float() / timesteps
                noise = torch.randn_like(xy)
                
                alpha = 1 - t.unsqueeze(-1)
                x_t = alpha * xy + (1 - alpha) * noise
                
                optimizer.zero_grad()
                pred_noise = self.model(x_t, t)
                loss = nn.functional.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()
        
        self.fitted = True
    
    def sample(self, n_samples, device):
        if not self.fitted:
            return torch.randn(n_samples, self.seq_len, self.feature_dim, device=device), torch.randn(n_samples, 1, device=device)
        
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, self.total_dim, device=device)
            
            timesteps = 100
            for i in reversed(range(timesteps)):
                t = torch.full((n_samples,), i / timesteps, device=device)
                pred_noise = self.model(x, t)
                alpha = 1 - t.unsqueeze(-1)
                x = (x - (1 - alpha) * pred_noise) / alpha.sqrt()
            
            X_syn = x[:, :-1].reshape(n_samples, self.seq_len, self.feature_dim)
            Y_syn = (x[:, -1:] > 0).float()
            
            return X_syn, Y_syn
