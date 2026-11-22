"""Audit paths, frozen model and the identical reporting rule with an adjustable gate."""
import os
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIVATE = HERE/'private'
STATS = HERE/'statistics'
LOGS = HERE/'logs'
EXP = HERE.parent
FIX = EXP/'unified_followup_repair_20261122'
OLD = EXP/'unified_followup_20261122'
sys.path.insert(0, str(FIX))
import numpy as np
import torch
import engine
from repair import Repaired, Variant
from train import inputs, predict, CFG, cutoff
from registry_data import save_json
sys.path.insert(0,str(EXP/'clinical_paths'))
from private_benchmark_math import endpoints, WeightedAP, boot_people
from torch.nn import functional as F

os.umask(0o077)
for folder in (LOGS, PRIVATE, STATS):
    folder.mkdir(exist_ok=True)
NOTICE='冻结后稳健性审计，不改变已锁定主模型 R'
SEEDS=(42,52,62)
TAUS=(.05,.10,.15,.20,.30,.50)
ARMS={'R':(1.,2.), 'aux025':(.25,2.), 'aux05':(.5,2.), 'aux2':(2.,2.),
      'gamma0':(1.,0.), 'gamma1':(1.,1.), 'gamma3':(1.,3.)}
DEEP=('iTransformer','TimeXer','SSSD','TSDiff','SurvTraj')


def obj(p):
    return json.loads(p.read_text())


def spec(name='R'):
    v=obj(FIX/'freeze_record.json')['variant']
    assert v==dict(prefix=True,weighting='double',sparse=True,diffusion=True)
    return Variant(**dict(v,diffusion=name!='N'))


def group(f,s):
    return Path(f'dev{f}')/str(s)


def source(f,s,arm='R'):
    return FIX/'private/validation'/group(f,s)/arm


class GateScan(Repaired):
    tau=.50

    def report(self,h,teacher=None):
        hidden=self.auxiliary[:3](h)
        out=self.auxiliary[3](hidden)
        logits,raw=out[:,:7],out[:,7:]
        cls=logits.argmax(1) if teacher is None else teacher
        gates=self.presence(hidden).sigmoid()>=self.tau
        dom=F.one_hot((cls-1).clamp_min(0),6).bool()
        top=F.softplus(raw[:,:1])*(cls>0).unsqueeze(1)
        other=top*raw[:,1:].sigmoid()*gates
        return logits,torch.cat((top,torch.where(dom,top,other[:,:6]),other[:,6:]),dim=1)


def load_net(f,s,name='R',scan=False):
    path=source(f,s,name) if name in ('R','N') else PRIVATE/'fits'/group(f,s)/name
    net=(GateScan if scan else Repaired)(spec(name))
    net.load_state_dict(engine.state(path/'best.pt'))
    return net.eval()
