"""Nine locked baseline families on the identical development inputs and outcomes."""
import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
from registry_data import ROOT, EXP, ORIGINAL, PRIVATE, save_json
from train import cutoff

sys.path.insert(0,str(ORIGINAL/"external/TSLib"))
sys.path.insert(0,str(EXP/"conditional_trajectories"))
from backbones import build,predict
sys.path.insert(0,str(EXP/"marked_progression"))
from trees import models
UP=EXP/"marked_progression/assets/upstream"
DEEP=["SSSD","TSDiff","SurvTraj","iTransformer","TimeXer"]
TREES=["LR","RF","HGB","XGBoost"]
EPOCHS=(1,3,6,10,16,25,36,45)


def views(data,fit):
    x=data["x"];t=data["lm"]
    valid=np.arange(3)<=t[:,None]
    sc=StandardScaler().fit(x[fit][valid[fit]])
    z=((x-sc.mean_)/sc.scale_).astype("f")*valid[:,:,None]
    neural=np.concatenate([z,valid[:,:,None],(data["a"][:,None]*valid)[:,:,None]],axis=2).astype("f")
    current=x[np.arange(len(t)),t]
    prior=x[np.arange(len(t)),np.maximum(t-1,0)]
    diff=(current-prior)*(t>0)[:,None]
    native=neural.reshape(len(x),-1)
    size=current[:,8:9]
    summary=np.column_stack([current,diff,np.divide(x.sum(1),1+t[:,None]),np.eye(3)[t],data["a"],
              np.maximum(size-np.array([4,6,8,12,20]),0),data["a"]*current[:,0],data["a"]*current[:,8]])
    return sc,neural,{"current_history":summary,"full_history":np.column_stack([native,summary])}


def neural(name,z,tr,ca,ev,bank,seed,folder,log):
    event=np.where(np.isin(bank["lead"][tr],[1,2]),bank["lead"][tr],0)
    y=bank["y"]; rng=np.random.default_rng(seed)
    np.random.seed(seed);torch.manual_seed(seed)
    dec=name in ("TimeXer","iTransformer")
    net=build(name,z[tr],event,UP,"cpu")
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    xx=torch.from_numpy(z[tr]);ee=torch.from_numpy(event)
    cc=torch.from_numpy(z[ca])
    pos,neg=np.flatnonzero(event),np.flatnonzero(event==0)
    negatives=rng.permutation(neg);ptr=0
    best=-1.;grid={}
    for ep in range(1,46):
        net.train(); nneg=7*len(pos)
        if ptr+nneg>len(neg):
            left=negatives[ptr:];negatives=rng.permutation(neg);ptr=nneg-len(left)
            controls=np.r_[left,negatives[:ptr]]
        else:
            controls=negatives[ptr:ptr+nneg];ptr+=nneg
        ix=rng.permutation(np.r_[pos,controls]);total=0.
        for b in np.array_split(ix,int(np.ceil(len(ix)/128))):
            xb,eb=xx[b],ee[b]
            loss=F.binary_cross_entropy_with_logits(net(xb),(eb>0).float()) if dec else net.loss(xb,eb)
            if not torch.isfinite(loss):raise FloatingPointError((name,ep))
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.,error_if_nonfinite=True)
            opt.step();total+=loss.item()*len(b)
        if ep in EPOCHS:
            pc=predict(net,dec,cc)
            ap=float(average_precision_score(y[ca],pc));grid[str(ep)]=ap
            if ap>best:
                best,chosen=ap,ep
                torch.save(net.state_dict(),folder/"best.pt")
        print(ep,total/len(ix),grid.get(str(ep)),file=log,flush=True)
    net.load_state_dict(torch.load(folder/"best.pt",weights_only=True),assign=name=="SSSD")
    return predict(net,dec,cc),predict(net,dec,torch.from_numpy(z[ev])),dict(selected=chosen,grid=grid)


def classical(name,mats,tr,ca,ev,y,seed,folder,log):
    grid={};best=-1.
    for vn,x in mats.items():
        ss=StandardScaler().fit(x[tr]) if name=="LR" else None
        xt=ss.transform(x[tr]) if ss else x[tr]
        xc=ss.transform(x[ca]) if ss else x[ca]
        for key,model in models(name,seed,y[tr].mean()):
            if name in ("RF","XGBoost"):model.set_params(n_jobs=1)
            model.fit(xt,y[tr])
            pc=model.predict_proba(xc)[:,1];ap=float(average_precision_score(y[ca],pc))
            grid[vn+"_"+key]=ap
            print(vn,key,ap,file=log,flush=True)
            if ap>best:
                best=ap;winner=(vn,key,ss,model,pc)
    vn,key,ss,est,pc=winner
    joblib.dump((vn,ss,est),folder/"best.joblib",compress=3)
    xv=mats[vn][ev]
    return pc,est.predict_proba(ss.transform(xv) if ss else xv)[:,1],dict(selected=vn+"_"+key,grid=grid)


def run(job):
    f,s,name=job
    folder=ROOT/"baselines"/f"dev{f}"/str(s)/name
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():return
    started=time.monotonic()
    tr,ca,ev=(np.flatnonzero(bank["folds"][f]==k) for k in range(3))
    sc,z,mat=views(bank,tr)
    with (folder/"training.log").open("w",buffering=1) as log,threadpool_limits(limits=1):
        if name in DEEP:
            pc,p,info=neural(name,z,tr,ca,ev,bank,s,folder,log)
        else:
            pc,p,info=classical(name,mat,tr,ca,ev,bank["y"],s,folder,log)
    assert np.isfinite(pc).all() and np.isfinite(p).all()
    np.savez_compressed(folder/"predictions.npz",cal_rows=ca,rows=ev,cal=pc,p=p)
    np.savez(folder/"scaling.npz",mean=sc.mean_,scale=sc.scale_)
    save_json(folder/"complete.json",dict(model=name,fold=f,seed=s,threshold=cutoff(bank["y"][ca],pc),seconds=time.monotonic()-started,**info))
    print("completed",name,f,s,flush=True)


def smoke():
    x=bank["x"][:200]
    e=np.tile([0,1,0,2],50)
    for name in DEEP:
        torch.manual_seed(42)
        net=build(name,x,e,UP,"cpu")
        xb=torch.from_numpy(x[:12]);ev=torch.from_numpy(e[:12]);disc=name in DEEP[-2:]
        loss=F.binary_cross_entropy_with_logits(net(xb),(ev>0).float()) if disc else net.loss(xb,ev)
        assert torch.isfinite(loss)
        loss.backward()
        assert np.isfinite(predict(net,disc,xb)).all()
        print("preflight",name,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=5);p.add_argument("--smoke",action="store_true")
    p.add_argument("--names",nargs="+",default=DEEP+TREES,choices=DEEP+TREES)
    args=p.parse_args();torch.set_num_threads(1)
    with np.load(PRIVATE/"registry.npz") as z:bank={k:z[k] for k in ("x","y","lm","a","lead","folds")}
    if args.smoke:smoke()
    else:
        jobs=[(f,s,name) for s in (42,52,62) for f in range(3) for name in args.names]
        with mp.get_context("fork").Pool(args.workers) as pool:list(pool.imap_unordered(run,jobs))
