import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_logit(p, eps=1e-6):
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

class PreAdaGN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, c):
        """
        Eq 28: \widetilde{\boldsymbol{H}}_{1} = \gamma_1(\boldsymbol{c}) \cdot \frac{\boldsymbol{H}_{in}^{(l)} - \mu}{\sigma} + \beta_1(\boldsymbol{c})
        """
        x = self.norm(x)
        # B x C_cond -> B x 1 x 1 x C_proj if x is BxCxTxD
        # Or if x is BxT(or D)xDim, we need to map correctly
        c_mapped = self.cond_proj(c).unsqueeze(1)
        gamma, beta = torch.chunk(c_mapped, 2, dim=-1)
        return x * (gamma + 1.0) + beta

class OrthogonalDualAttentionBlock(nn.Module):
    def __init__(self, dim, t_steps, cond_dim, heads=1,
                 use_time_attn=True, use_feat_attn=True):
        super().__init__()
        self.use_time_attn = use_time_attn
        self.use_feat_attn = use_feat_attn

        if use_time_attn:
            self.ada_gn_time = PreAdaGN(dim, cond_dim)
            # Using 1 head to avoid divisibility assertion errors with arbitrary D_prime/t_steps dimensions
            self.time_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)
        
        if use_feat_attn:
            # When we permute, the sequence is dim, and the feature (embed_dim) becomes t_steps
            self.ada_gn_feat = PreAdaGN(t_steps, cond_dim)
            self.feat_attn = nn.MultiheadAttention(embed_dim=t_steps, num_heads=1, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, c):
        """
        x shape: [Batch, T, D_prime] -> Here D_prime corresponds to features treated as the embed_dim for time attn
        Wait, standard attention treats the last dim as embed_dim.
        If we want time attention, sequence is T, embed is D_prime.
        """
        # --- Time Attention (Sequence = T, Embed = D_prime) ---
        if self.use_time_attn:
            res = x
            x_time = self.ada_gn_time(x, c)
            x_attn_time, _ = self.time_attn(x_time, x_time, x_time) # Eq 29
            x = res + x_attn_time
        
        # --- Feature Attention (Sequence = D_prime, Embed = T) ---
        if self.use_feat_attn:
            # Permute: [B, T, D_prime] -> [B, D_prime, T]
            x_perm = x.transpose(1, 2)
            res_feat = x_perm
            
            # NOTE: PreAdaGN expects the last dimension to match `dim`.
            # For permutation to work flawlessly, we assume time and feat embed dims are handled symmetrically
            # or we flatten. Here we simplify by treating the transposed tensor directly if they are square, 
            # or we project. To strictly adhere to the proposal: 
            # "Permutation is key to spatial-temporal decoupling."
            
            # Simple transposition for feature attention is conceptual. In practice we might need another Linear projection.
            # Let's mock the feat attention for structural completeness
            x_feat = self.ada_gn_feat(x_perm, c)
            x_attn_feat, _ = self.feat_attn(x_feat, x_feat, x_feat) # Eq 32
            x_perm = res_feat + x_attn_feat
            
            # Permute back
            x = x_perm.transpose(1, 2)
        
        # FFN
        x = x + self.ffn(x)
        return x

class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_t):
        """
        Calculates predicted alpha (environmental exposure).
        x_t shape: [Batch, T, D]
        """
        _, (hn, _) = self.lstm(x_t)
        # hn shape: [1, Batch, hidden_dim]
        alpha_pred = self.regressor(hn.squeeze(0))
        return alpha_pred

class CausalTabDiff(nn.Module):
    """
    【受控原创降级协议】 (Controlled Originality Fallback Protocol) Triggered.
    
    This class implements the Time-series Causal Diffusion Model with Analog Bits 
    and Orthogonal Dual-Attention as existing libraries lack this integration.
    
    ### Step A: Theoretical Derivation 
    1. **Energy Function Validation**:
       The LSTM discriminator evaluates if the generated trajectory $x_t$ matches the target causal exposure $\alpha_{target}$.
       $$U(\mathbf{x}_t) = \|f_\phi(\mathbf{x}_t) - \alpha_{target}\|^2$$
       
    2. **Gradient Guidance Sampling (Reverse Diffusion)**:
       Moves the mean of the reverse diffusion step towards the low-energy (causally sound) regions.
       $$\mathbf{x}_{t-1} = \pmb{\mu}_{\theta}(\mathbf{x}_t) - s \cdot \sigma_t \cdot \nabla_{\mathbf{x}_t} U(\mathbf{x}_t) + \sigma_t\mathbf{z}$$
    """
    def __init__(self, t_steps=3, feature_dim=12, cond_dim=1, diffusion_steps=1000, heads=4, use_noise_head=True,
                 use_time_attn=True, use_feat_attn=True):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.feature_dim = feature_dim
        self.t_steps = t_steps
        self.use_noise_head = bool(use_noise_head)
        
        # Cond embedding
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, 64), nn.SiLU(), nn.Linear(64, feature_dim))
        
        # Backbone
        self.block1 = OrthogonalDualAttentionBlock(dim=feature_dim, t_steps=t_steps, cond_dim=feature_dim, heads=heads,
                                                   use_time_attn=use_time_attn, use_feat_attn=use_feat_attn)
        
        # Time embedding for t
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, feature_dim)
        )
        self.noise_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Discriminator for gradient guidance
        self.discriminator = LSTMDiscriminator(input_dim=feature_dim)

        # Minimal joint outcome head:
        # keeps Y inside the model contract instead of leaving it purely to wrapper glue code.
        self.outcome_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.risk_time_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.risk_hazard_head = nn.Linear(32, 1)
        
        # Noise schedule
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, diffusion_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def predict_outcome_logits(self, x, alpha_target):
        """Minimal joint Y head using trajectory mean summary + treatment embedding."""
        traj_summary = x.mean(dim=1)
        alpha_emb = self.cond_mlp(alpha_target)
        return self.outcome_head(torch.cat([traj_summary, alpha_emb], dim=-1))

    def predict_outcome_proba(self, x, alpha_target):
        return torch.sigmoid(self.predict_outcome_logits(x, alpha_target))

    def predict_risk_hazard_logits(self, x, alpha_target):
        alpha_emb = self.cond_mlp(alpha_target).unsqueeze(1).expand(-1, x.shape[1], -1)
        traj_context = x.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        risk_features = torch.cat([x, alpha_emb, traj_context], dim=-1)
        hidden = self.risk_time_mlp(risk_features)
        return self.risk_hazard_head(hidden)

    def predict_risk_hazards(self, x, alpha_target):
        return torch.sigmoid(self.predict_risk_hazard_logits(x, alpha_target))

    def predict_cumulative_risk(self, x, alpha_target):
        hazards = self.predict_risk_hazards(x, alpha_target)
        survival = torch.cumprod(1.0 - hazards.clamp(min=1e-6, max=1.0 - 1e-6), dim=1)
        risk = 1.0 - survival[:, -1, :]
        return risk

    def predict_counterfactual_risk_gap(self, x, alpha_target=None):
        batch_size = x.shape[0]
        device = x.device
        alpha_low = torch.zeros((batch_size, 1), device=device, dtype=x.dtype)
        alpha_high = torch.ones((batch_size, 1), device=device, dtype=x.dtype)
        risk_low = self.predict_cumulative_risk(x, alpha_low)
        risk_high = self.predict_cumulative_risk(x, alpha_high)
        return risk_high - risk_low

    def _predict_noise(self, x_t, alpha_target, t):
        c_emb = self.cond_mlp(alpha_target) + self.time_mlp(t.unsqueeze(-1).float())
        h = self.block1(x_t, c_emb)
        if self.use_noise_head:
            return self.noise_head(h)
        return h

    def _predict_x0_from_noise(self, x_t, pred_noise, t):
        sqrt_alphas_cumprod_t = self._extract(torch.sqrt(self.alphas_cumprod), t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(torch.sqrt(1.0 - self.alphas_cumprod), t, x_t.shape)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t.clamp(min=1e-6)

    def _guidance_weight(self, step_idx, base_scale, schedule='constant', power=2.0):
        if base_scale <= 0.0:
            return 0.0

        schedule = str(schedule).strip().lower()
        if schedule in ('', 'constant', 'flat'):
            return float(base_scale)

        progress = 1.0 - float(step_idx) / max(1.0, float(self.diffusion_steps - 1))
        progress = min(1.0, max(0.0, progress))
        shaped = progress ** max(1.0, float(power))

        if schedule in ('late', 'late_ramp', 'progressive'):
            return float(base_scale) * shaped

        return float(base_scale)

    def _pairwise_outcome_ranking_loss(self, outcome_logits, y_binary, max_samples_per_class=32):
        logits_flat = outcome_logits.reshape(-1)
        y_flat = y_binary.reshape(-1) > 0.5

        pos_logits = logits_flat[y_flat]
        neg_logits = logits_flat[~y_flat]

        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return outcome_logits.new_tensor(0.0)

        if pos_logits.numel() > max_samples_per_class:
            pos_idx = torch.randperm(pos_logits.numel(), device=pos_logits.device)[:max_samples_per_class]
            pos_logits = pos_logits[pos_idx]
        if neg_logits.numel() > max_samples_per_class:
            neg_idx = torch.randperm(neg_logits.numel(), device=neg_logits.device)[:max_samples_per_class]
            neg_logits = neg_logits[neg_idx]

        pairwise_margin = pos_logits[:, None] - neg_logits[None, :]
        return F.softplus(-pairwise_margin).mean()

    def forward(
        self,
        x_0,
        alpha_target,
        y_target=None,
        pos_weight=None,
        rank_loss_weight=0.0,
        use_trajectory_risk_head=False,
        risk_smoothness_weight=0.0,
        cf_consistency_weight=0.0,
        denoise_recon_weight=0.0,
        batch_moment_weight=0.0,
    ):
        """Training forward pass (computes diffusion loss, discriminator loss, optional outcome loss)."""
        batch_size = x_0.shape[0]
        # Sample random t
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        
        # Add noise
        sqrt_alphas_cumprod_t = self._extract(torch.sqrt(self.alphas_cumprod), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        pred_noise = self._predict_noise(x_t, alpha_target, t)
        
        # 1. Diffusion Loss
        diff_loss = F.mse_loss(pred_noise, noise)
        x0_hat = self._predict_x0_from_noise(x_t, pred_noise, t)

        if denoise_recon_weight > 0.0:
            recon_loss = F.smooth_l1_loss(x0_hat, x_0)
            diff_loss = diff_loss + float(denoise_recon_weight) * recon_loss

        if batch_moment_weight > 0.0:
            reduce_dims = (0, 1)
            x0_mean = x_0.mean(dim=reduce_dims)
            hat_mean = x0_hat.mean(dim=reduce_dims)
            x0_std = x_0.std(dim=reduce_dims, unbiased=False)
            hat_std = x0_hat.std(dim=reduce_dims, unbiased=False)
            moment_loss = F.mse_loss(hat_mean, x0_mean) + F.mse_loss(hat_std, x0_std)
            diff_loss = diff_loss + float(batch_moment_weight) * moment_loss
        
        # 2. Discriminator Loss (Energy function training)
        alpha_pred = self.discriminator(x_t.detach())
        disc_loss = F.mse_loss(alpha_pred, alpha_target)

        if y_target is None:
            return diff_loss, disc_loss

        y_binary = (y_target > 0.5).float()
        if pos_weight is None:
            pos_weight = torch.ones((1,), device=x_0.device, dtype=x_0.dtype)
            
        def _focal_loss_with_logits(logits, targets, gamma=2.0, pos_weight=None):
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = ((1 - pt) ** gamma) * bce_loss
            if pos_weight is not None:
                weight = torch.where(targets > 0.5, pos_weight.reshape(1), torch.ones_like(targets))
                focal_loss = focal_loss * weight
            return focal_loss.mean()

        def _focal_loss_prob(probs, targets, gamma=2.0, pos_weight=None):
            bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = ((1 - pt) ** gamma) * bce_loss
            if pos_weight is not None:
                weight = torch.where(targets > 0.5, pos_weight.reshape(1), torch.ones_like(targets))
                focal_loss = focal_loss * weight
            return focal_loss.mean()

        if use_trajectory_risk_head:
            cumulative_risk = self.predict_cumulative_risk(x_0, alpha_target)
            outcome_loss = _focal_loss_prob(
                cumulative_risk.clamp(min=1e-6, max=1.0 - 1e-6),
                y_binary,
                pos_weight=pos_weight
            )

            if risk_smoothness_weight > 0.0:
                hazards = self.predict_risk_hazards(x_0, alpha_target)
                smooth_loss = ((hazards[:, 1:, :] - hazards[:, :-1, :]) ** 2).mean()
                outcome_loss = outcome_loss + float(risk_smoothness_weight) * smooth_loss

            if cf_consistency_weight > 0.0:
                cf_gap = self.predict_counterfactual_risk_gap(x_0, alpha_target)
                cf_loss = F.relu(0.0 - cf_gap.mean())
                outcome_loss = outcome_loss + float(cf_consistency_weight) * cf_loss
        else:
            outcome_logits = self.predict_outcome_logits(x_0, alpha_target)
            outcome_loss = _focal_loss_with_logits(
                outcome_logits,
                y_binary,
                pos_weight=pos_weight
            )
        if rank_loss_weight > 0.0:
            rank_loss = self._pairwise_outcome_ranking_loss(outcome_logits, y_binary)
            outcome_loss = outcome_loss + float(rank_loss_weight) * rank_loss
        
        return diff_loss, disc_loss, outcome_loss

    def sample_with_guidance(
        self,
        batch_size,
        alpha_target,
        guidance_scale=1.0,
        guidance_schedule='constant',
        guidance_power=2.0,
    ):
        """
        Step D: Causal Constraint Gradient Guidance
        """
        device = alpha_target.device
        x = torch.randn((batch_size, self.t_steps, self.feature_dim), device=device)
        discriminator_training = self.discriminator.training
        
        try:
            if guidance_scale > 0.0:
                self.discriminator.train()

            for i in reversed(range(0, self.diffusion_steps)):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                
                pred_noise = self._predict_noise(x, alpha_target, t)
                
                # Eq 36: Calculate \mu_\theta
                alpha = self.alphas[t][:, None, None]
                alpha_cumprod = self.alphas_cumprod[t][:, None, None]
                beta = self.betas[t][:, None, None]
                
                mu_theta = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * pred_noise)
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                sigma_t = torch.sqrt(beta)

                scheduled_guidance = self._guidance_weight(
                    step_idx=i,
                    base_scale=guidance_scale,
                    schedule=guidance_schedule,
                    power=guidance_power,
                )

                if scheduled_guidance > 0.0:
                    # Eq 37: Gradient Guided Step
                    x_in = x.detach().requires_grad_()
                    with torch.enable_grad():
                        alpha_pred = self.discriminator(x_in)
                        energy = F.mse_loss(alpha_pred, alpha_target, reduction='sum')
                    grad_U = torch.autograd.grad(energy, x_in)[0]

                    x = mu_theta - scheduled_guidance * sigma_t * grad_U + sigma_t * noise
                else:
                    # Standard Denoising Step
                    x = mu_theta + sigma_t * noise
        finally:
            self.discriminator.train(discriminator_training)

        return x
