import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from registry_data import ROOT, PRIVATE, save_json
from model import Unified, UncertaintyWeightedLoss, focal_loss_with_logits

CFG=json.loads((ROOT/"protocol.json").read_text())
torch.set_num_threads(1)


def load():
    with np.load(PRIVATE/"registry.npz") as data:
        return {k:torch.from_numpy(data[k].copy()) for k in data}


def indices(d,fold):
    return tuple(torch.where(d["folds"][fold]==k)[0] for k in range(3))


def inputs(d,b):
    return d["x"][b],d["a"][b,None],d["lm"][b]+1


def cutoff(y,p):
    prec,rec,thr=precision_recall_curve(y,p)
    f=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-15)
    return float(thr[np.flatnonzero(f==f.max())[-1]])


@torch.no_grad()
def predict(net,d,idx,aux=False):
    net.eval()
    ps=[];values=[];probs=[]
    for b in idx.split(512):
        h=net.encode(*inputs(d,b))
        ps.append(net.core.risk_head(h).sigmoid().flatten().numpy())
        if aux:
            cl,v=net.report(h)
            values.append(v.numpy())
            probs.append(cl.softmax(1).numpy())
    if aux:
        return np.concatenate(ps),np.concatenate(values),np.concatenate(probs)
    return np.concatenate(ps)


def random_phase(seed,epoch,batch):
    return (1000003*seed+1009*epoch+batch)%2147483647


def history_loss(net,d,rows):
    base=net.core.base_model
    den,cond=base(d["x"][rows],d["a"][rows,None])
    return den+.5*cond


def pretrain(d,f,s,folder):
    path=folder/"pretrained.pt"
    if path.exists():
        return path
    torch.manual_seed(s)
    net=Unified("A")
    fit,_,_=indices(d,f)
    opt=torch.optim.Adam(net.core.base_model.parameters(),lr=.001,weight_decay=.0001)
    gen=torch.Generator().manual_seed(s)
    start=time.monotonic()
    with (folder/"pretrain.log").open("w",buffering=1) as log:
        for ep in range(1,CFG["pretrain_epochs"]+1):
            order=fit[torch.randperm(len(fit),generator=gen)]
            total=0.
            for j,b in enumerate(order.split(CFG["batch"])):
                torch.manual_seed(random_phase(s,ep,j))
                loss=history_loss(net,d,b)
                opt.zero_grad();loss.backward();opt.step()
                total+=loss.item()*len(b)
            if not np.isfinite(total):raise FloatingPointError("pretrain")
            print(ep,total/len(fit),time.monotonic()-start,file=log)
    tmp=folder/"pretraining.tmp"
    torch.save(net.core.state_dict(),tmp)
    tmp.replace(path)
    return path


def fit_one(d,f,s,arm):
    common=ROOT/"runs"/f"dev{f}"/str(s)
    common.mkdir(parents=True,exist_ok=True)
    folder=common/arm
    folder.mkdir(exist_ok=True)
    if (folder/"complete.json").exists():return
    fit,cal,ev=indices(d,f)
    torch.manual_seed(s)
    net=Unified(arm)
    if arm!="D":
        net.core.load_state_dict(torch.load(pretrain(d,f,s,common),weights_only=True))
    core=net.core
    uw=UncertaintyWeightedLoss(3)
    with torch.no_grad():uw.log_sigma[2]=-1
    group=[dict(params=core.risk_head.parameters(),lr=.001),dict(params=core.trajectory_head.parameters(),lr=.0005),
           dict(params=core.base_model.parameters(),lr=.0001),dict(params=uw.parameters(),lr=.01)]
    if arm!="A":group.append(dict(params=net.auxiliary.parameters(),lr=.001))
    opt=torch.optim.Adam(group,weight_decay=.0001)
    labels=d["y"][fit].float()
    ratio=(len(fit)-labels.sum())/labels.sum()
    weights=torch.where(labels>0,5*ratio,1.).double()
    gen=torch.Generator().manual_seed(s+1)
    best=-1.; start=time.monotonic()
    with (folder/"training.log").open("w",buffering=1) as log:
        print("epoch,loss,cal_ap,best_ap,seconds",file=log)
        for ep in range(1,CFG["finetune_epochs"]+1):
            net.train()
            order=fit[torch.multinomial(weights,len(fit),replacement=True,generator=gen)]
            total=0.
            for j,b in enumerate(order.split(CFG["batch"])):
                stream=random_phase(s,ep,j)
                if arm!="D":
                    torch.manual_seed(stream+330001)
                    hist=history_loss(net,d,b)
                torch.manual_seed(stream)
                h=net.encode(*inputs(d,b))
                annual=core.compute_trajectory_loss(core.trajectory_head(h),d["annual"][b],d["active"][b])
                cls=focal_loss_with_logits(core.risk_head(h),d["y"][b,None].float(),2.,ratio.reshape(1))
                if arm!="D":
                    loss=uw([hist,annual,cls])
                else:
                    losses=torch.stack([annual,cls])
                    loss=(torch.exp(-uw.log_sigma[1:])*losses+uw.log_sigma[1:]).sum()
                if arm!="A":
                    loss=loss+CFG["aux_weight"]*net.auxiliary_loss(h,d["aux"][b],d["dom"][b],d["mask"][b])
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                total+=loss.item()*len(b)
            pc=predict(net,d,cal)
            ap=float(average_precision_score(d["y"][cal],pc))
            if ap>best:
                best,chosen=ap,ep
                torch.save(net.state_dict(),folder/"best.pt")
            if ep==10:torch.save(net.state_dict(),folder/"epoch10.pt")
            print(f"{ep},{total/len(fit):.9g},{ap:.10g},{best:.10g},{time.monotonic()-start:.3f}",file=log)
            if ep-chosen>=CFG["patience"]:break
    net.load_state_dict(torch.load(folder/"epoch10.pt",weights_only=True))
    fixed=predict(net,d,ev)
    net.load_state_dict(torch.load(folder/"best.pt",weights_only=True))
    pc=predict(net,d,cal)
    extra={}
    if arm=="A":p=predict(net,d,ev)
    else:
        p,v,q=predict(net,d,ev,True)
        valid=d["mask"][ev].numpy()
        extra=dict(aux_rows=ev.numpy()[valid],sizes=v[valid],classes=q[valid])
    assert np.isfinite(p).all() and np.isfinite(pc).all()
    np.savez_compressed(folder/"predictions.npz",cal_rows=cal,rows=ev,cal=pc,p=p,fixed10=fixed,**extra)
    save_json(folder/"complete.json",dict(arm=arm,fold=f,seed=s,epochs=ep,selected=chosen,cal_ap=best,
             threshold=cutoff(d["y"][cal],pc),seconds_fine=time.monotonic()-start,device="cpu",torch=torch.__version__,
             platform=platform.platform(),parameters=sum(u.numel() for u in net.parameters()),
             scoring_parameters=sum(u.numel() for module in (core.base_model.cond_mlp,core.base_model.block1,core.risk_head) for u in module.parameters())))
    print("completed",arm,f,s,flush=True)
