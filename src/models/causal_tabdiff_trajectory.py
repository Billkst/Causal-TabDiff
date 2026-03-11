import torch
import torch.nn as nn
from .causal_tabdiff import CausalTabDiff

class CausalTabDiffTrajectory(nn.Module):
    def __init__(self, t_steps, feature_dim, diffusion_steps=100, trajectory_len=7, cond_dim=1):
        super().__init__()
        self.base_model = CausalTabDiff(t_steps, feature_dim, cond_dim, diffusion_steps)
        self.trajectory_len = trajectory_len
        
        self.trajectory_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, trajectory_len)
        )
        
        self.risk_2year_head = nn.Linear(trajectory_len, 1)
    
    def forward(self, x, alpha_target):
        diff_loss, disc_loss = self.base_model(x, alpha_target)
        
        c_emb = self.base_model.cond_mlp(alpha_target)
        h = self.base_model.block1(x, c_emb)
        h_pooled = h.mean(dim=1)
        
        trajectory_logits = self.trajectory_head(h_pooled)
        trajectory_probs = torch.sigmoid(trajectory_logits)
        
        risk_2year_logits = self.risk_2year_head(trajectory_probs)
        risk_2year = torch.sigmoid(risk_2year_logits)
        
        return {
            'diff_loss': diff_loss,
            'disc_loss': disc_loss,
            'trajectory': trajectory_probs,
            'risk_2year': risk_2year
        }
    
    def sample_with_guidance(self, batch_size, alpha_target, guidance_scale=2.0):
        return self.base_model.sample_with_guidance(batch_size, alpha_target, guidance_scale)
