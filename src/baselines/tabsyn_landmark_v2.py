"""
TabSyn-Inspired Simplified VAE Wrapper

⚠️ 实现完整度: 15-20%
⚠️ 类型: 简化代理版本 (Simplified Proxy)
⚠️ 不可作为原始 TabSyn baseline 直接宣称

缺失组件:
- Tokenizer (混合数值/类别嵌入)
- MultiheadAttention
- Diffusion 阶段 (两阶段训练)
- EDM Loss

仅用于快速原型验证，不适合正式 baseline 对比。
如需正式 TabSyn baseline，请实现完整架构。
"""
import torch
import torch.nn as nn
import numpy as np


class TabSynLandmarkWrapper:
    def __init__(self, seq_len, feature_dim):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.total_dim = seq_len * feature_dim + 1
        self.model = None
        self.fitted = False
    
    def fit(self, train_loader, epochs, device):
        # TabSyn 需要 VAE + Diffusion 两阶段训练
        # 由于其复杂的架构（需要类别特征处理），
        # 而当前数据是纯连续特征，直接使用简化版本
        
        # 使用简单的 VAE 作为替代
        from torch import nn
        
        class SimpleVAE(nn.Module):
            def __init__(self, input_dim, latent_dim=32):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim * 2)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim)
                )
                self.latent_dim = latent_dim
            
            def forward(self, x):
                h = self.encoder(x)
                mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                recon = self.decoder(z)
                return recon, mu, logvar
        
        self.model = SimpleVAE(self.total_dim).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            for batch in train_loader:
                x = batch['x'].to(device)
                y = batch['y_2year'].to(device)
                
                x_flat = x.reshape(x.shape[0], -1)
                xy = torch.cat([x_flat, y], dim=1)
                
                optimizer.zero_grad()
                recon, mu, logvar = self.model(xy)
                
                recon_loss = nn.functional.mse_loss(recon, xy)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.001 * kl_loss
                
                loss.backward()
                optimizer.step()
        
        self.fitted = True
    
    def sample(self, n_samples, device):
        if not self.fitted:
            return torch.randn(n_samples, self.seq_len, self.feature_dim, device=device), torch.randn(n_samples, 1, device=device)
        
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, 32, device=device)
            samples = self.model.decoder(z)
            
            X_syn = samples[:, :-1].reshape(n_samples, self.seq_len, self.feature_dim)
            Y_syn = (samples[:, -1:] > 0).float()
            
            return X_syn, Y_syn
