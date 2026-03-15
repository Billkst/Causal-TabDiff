import torch
import torch.nn as nn
import torch.nn.functional as F
from .causal_tabdiff import CausalTabDiff

class CausalTabDiffTrajectory(nn.Module):
    def __init__(self, t_steps, feature_dim, diffusion_steps=100, trajectory_len=7, cond_dim=1):
        super().__init__()
        self.base_model = CausalTabDiff(t_steps, feature_dim, cond_dim, diffusion_steps)
        self.trajectory_len = trajectory_len
        self.feature_dim = feature_dim
        
        self.trajectory_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, trajectory_len)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
    
    def _masked_mean_pool(self, h, history_length):
        """Average pooling over valid time steps only, ignoring padding."""
        B, T, D = h.shape
        mask = torch.arange(T, device=h.device).unsqueeze(0) < history_length.view(B, 1)
        mask_f = mask.unsqueeze(2).float()
        h_sum = (h * mask_f).sum(dim=1)
        h_pooled = h_sum / mask_f.sum(dim=1).clamp(min=1)
        return h_pooled
    
    def forward(self, x, alpha_target, history_length=None):
        diff_loss, disc_loss = self.base_model(x, alpha_target)
        
        c_emb = self.base_model.cond_mlp(alpha_target)
        h = self.base_model.block1(x, c_emb)
        
        if history_length is not None:
            h_pooled = self._masked_mean_pool(h, history_length)
        else:
            h_pooled = h.mean(dim=1)
        
        trajectory_logits = self.trajectory_head(h_pooled)
        trajectory_probs = torch.sigmoid(trajectory_logits)
        
        risk_2year = self.compute_2year_risk(trajectory_probs)
        risk_2year_direct = self.risk_head(h_pooled)
        
        return {
            'diff_loss': diff_loss,
            'disc_loss': disc_loss,
            'trajectory': trajectory_probs,
            'risk_2year': risk_2year,
            'risk_2year_logit': risk_2year_direct,
        }
    
    def compute_2year_risk(self, trajectory_probs):
        hazards_2year = trajectory_probs[:, :2]
        survival_2year = torch.cumprod(1.0 - hazards_2year.clamp(min=1e-6, max=1.0 - 1e-6), dim=1)
        risk_2year = 1.0 - survival_2year[:, -1:]
        return risk_2year
    
    def compute_trajectory_loss(self, pred_trajectory, target_trajectory, valid_mask):
        valid_mask = valid_mask.float()
        loss_per_element = F.binary_cross_entropy(pred_trajectory, target_trajectory, reduction='none')
        masked_loss = loss_per_element * valid_mask
        total_valid = valid_mask.sum()
        if total_valid > 0:
            return masked_loss.sum() / total_valid
        else:
            return masked_loss.sum()
    
    def sample_with_guidance(self, batch_size, alpha_target, guidance_scale=2.0):
        return self.base_model.sample_with_guidance(batch_size, alpha_target, guidance_scale)
