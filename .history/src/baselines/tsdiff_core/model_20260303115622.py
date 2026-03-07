import torch
import torch.nn as nn
import torch.nn.functional as F


class TSDiffDDPM(nn.Module):
    def __init__(self, input_dim: int, timesteps: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.timesteps = max(50, min(int(timesteps), 1000))

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, input_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, input_dim),
        )

        betas = torch.linspace(1e-4, 2e-2, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape):
        out = arr.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x0 = x.reshape(b, -1)

        t = torch.randint(0, self.timesteps, (b,), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x0)

        ac = self._extract(self.alphas_cumprod, t, x0.shape)
        xt = torch.sqrt(ac) * x0 + torch.sqrt(1.0 - ac) * noise

        t_embed = self.time_mlp(t.float().unsqueeze(-1))
        pred_noise = self.net(xt + t_embed)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int, features: int, device):
        x = torch.randn(batch_size, features, device=device)

        sample_steps = min(50, self.timesteps)
        step_ids = torch.linspace(self.timesteps - 1, 0, steps=sample_steps, device=device).long()

        for t_val in step_ids:
            t = torch.full((batch_size,), int(t_val.item()), device=device, dtype=torch.long)
            t_embed = self.time_mlp(t.float().unsqueeze(-1))
            pred_noise = self.net(x + t_embed)

            alpha = self._extract(self.alphas, t, x.shape)
            alpha_cumprod = self._extract(self.alphas_cumprod, t, x.shape)
            beta = self._extract(self.betas, t, x.shape)

            x = (x - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha)
            if int(t_val.item()) > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

        return x.view(batch_size, seq_len, features)
