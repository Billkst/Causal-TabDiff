"""Three local switches on the archived unified model. Prediction uses the prefix only."""
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.with_name("unified_followup_20261122")
sys.path.insert(0, str(ARCHIVE))
from model import Unified


@dataclass(frozen=True)
class Variant:
    prefix: bool = False
    weighting: str = "double"
    sparse: bool = False
    diffusion: bool = True


def observed(x, length):
    return torch.arange(x.size(1), device=x.device)[None, :] < length[:, None]


def prefix_block(block, x, condition, valid):
    keep = valid.unsqueeze(-1)
    x = x.masked_fill(~keep, 0.)
    t = block.ada_gn_time(x, condition)
    z, _ = block.time_attn(t, t, t, key_padding_mask=~valid)
    x = (x + z).masked_fill(~keep, 0.)
    feature = x.transpose(1, 2)
    axis = valid.unsqueeze(1)
    # Feature-axis attention embeds an entire time series per feature. Compute
    # its LayerNorm statistics on the observed times before affine modulation.
    mod = block.ada_gn_feat
    count = axis.sum(-1, keepdim=True)
    mean = feature.sum(-1, keepdim=True) / count
    centered = (feature - mean).masked_fill(~axis, 0.)
    var = centered.square().sum(-1, keepdim=True) / count
    normed = centered * (var + mod.norm.eps).rsqrt()
    normed = normed * mod.norm.weight + mod.norm.bias
    gamma, beta = mod.cond_proj(condition).unsqueeze(1).chunk(2, -1)
    f = (normed * (1 + gamma) + beta).masked_fill(~axis, 0.)
    z, _ = block.feat_attn(f, f, f)
    x = (feature + z).transpose(1, 2).masked_fill(~keep, 0.)
    return (x + block.ffn(x)).masked_fill(~keep, 0.)


class Repaired(Unified):
    def __init__(self, v):
        super().__init__("C" if v.diffusion else "D")
        self.variant = v
        if v.sparse:
            # Same hidden representation and initial parameters in the inherited head.
            self.presence = nn.Linear(64, 7)

    def encode(self, x, a, length):
        if not self.variant.prefix:
            return super().encode(x, a, length)
        base = self.core.base_model
        h = prefix_block(base.block1, x, base.cond_mlp(a), observed(x, length))
        return self.core._masked_mean_pool(h, length)

    def history_terms(self, x, a, length, *, step=None, noise=None):
        base = self.core.base_model
        if not self.variant.prefix:
            return base(x, a)
        valid = observed(x, length)
        keep = valid.unsqueeze(-1)
        if step is None:
            step = torch.randint(0, base.diffusion_steps, (len(x),), device=x.device)
        if noise is None:
            noise = torch.randn_like(x)
        noise = noise.masked_fill(~keep, 0.)
        x = x.masked_fill(~keep, 0.)
        coeff = base.alphas_cumprod[step, None, None]
        noisy = coeff.sqrt()*x + (1.-coeff).sqrt()*noise
        condition = base.cond_mlp(a) + base.time_mlp(step.float().unsqueeze(-1))
        pred = base.noise_head(prefix_block(base.block1, noisy, condition, valid))
        # All landmarks have equal weight, regardless of the available prefix length.
        sq = (pred-noise).square().masked_fill(~keep, 0.)
        denoise = (sq.sum((1, 2)) / (length * x.shape[-1])).mean()
        clean = base._predict_x0_from_noise(noisy, pred, step).masked_fill(~keep, 0.)
        disc = base.discriminator
        sequence, _ = disc.lstm(clean)
        last = sequence[torch.arange(len(x), device=x.device), length-1]
        return denoise, F.mse_loss(disc.regressor(last), a)

    def sparse_report(self, h, teacher=None, target=None):
        hidden = self.auxiliary[:3](h)
        output = self.auxiliary[3](hidden)
        logits, raw = output[:, :7], output[:, 7:]
        gate_logits = self.presence(hidden)
        cls = logits.argmax(1) if teacher is None else teacher
        gates = gate_logits >= 0 if target is None else target[:, 1:] > 0
        dominant = F.one_hot((cls-1).clamp_min(0), 6).bool()
        top = F.softplus(raw[:, :1]) * (cls > 0).unsqueeze(1)
        other = top * raw[:, 1:].sigmoid() * gates
        sites = torch.where(dominant, top, other[:, :6])
        return logits, torch.cat((top, sites, other[:, 6:]), dim=1), gate_logits, dominant

    def report(self, h, teacher=None):
        if self.variant.sparse:
            return self.sparse_report(h, teacher)[:2]
        return super().report(h, teacher)

    def auxiliary_loss(self, h, target, dom, mask):
        if not self.variant.sparse:
            return super().auxiliary_loss(h, target, dom, mask)
        if not mask.any():
            return h.sum()*0.
        y, cls = target[mask], dom[mask]
        logits, values, gate_logits, dominant = self.sparse_report(h[mask], cls, y)
        loss = F.cross_entropy(logits, cls)
        pos = cls > 0
        if pos.any():
            include = torch.cat((~dominant, torch.ones_like(dominant[:, :1])), 1)
            eligible = include & pos.unsqueeze(1)
            presence = F.binary_cross_entropy_with_logits(
                gate_logits[eligible], (y[:, 1:][eligible] > 0).to(h.dtype))
            loss = loss + F.smooth_l1_loss(values[pos], y[pos]) + presence
        return loss
