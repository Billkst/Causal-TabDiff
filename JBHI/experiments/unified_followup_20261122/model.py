"""One history encoder, a deployment-only risk route, and train-time report supervision."""
import sys

import torch
from torch import nn
from torch.nn import functional as F

from registry_data import ORIGINAL
sys.path.insert(0, str(ORIGINAL))
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from run_phase2_twostage import focal_loss_with_logits, UncertaintyWeightedLoss


class Unified(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm=arm
        self.core=CausalTabDiffTrajectory(3,14,100,7)
        if arm!="A":
            self.auxiliary=nn.Sequential(nn.LayerNorm(14),nn.Linear(14,64),nn.GELU(),nn.Linear(64,15))

    def encode(self,x,a,length):
        base=self.core.base_model
        h=base.block1(x,base.cond_mlp(a))
        return self.core._masked_mean_pool(h,length)

    def risk(self,x,a,length):
        return self.core.risk_head(self.encode(x,a,length)).flatten()

    def report(self,h,teacher=None):
        output=self.auxiliary(h)
        logits,raw=output[:,:7],output[:,7:]
        cls=logits.argmax(1) if teacher is None else teacher
        present=(cls>0)[:,None]
        top=F.softplus(raw[:,:1])
        if self.arm=="B":
            decoded=F.softplus(raw)
        else:
            direction=F.one_hot((cls-1).clamp_min(0),6).bool()
            regions=torch.where(direction,1.,raw[:,1:7].sigmoid())*top
            decoded=torch.cat([top,regions,top*raw[:,7:].sigmoid()],dim=1)
        return logits,decoded*present

    def auxiliary_loss(self,h,target,dom,mask):
        if not mask.any():
            return h.sum()*0
        cls=dom[mask]
        logits,values=self.report(h[mask],cls)
        loss=F.cross_entropy(logits,cls)
        has=cls>0
        if has.any():
            loss=loss+F.smooth_l1_loss(values[has],target[mask][has])
        return loss
