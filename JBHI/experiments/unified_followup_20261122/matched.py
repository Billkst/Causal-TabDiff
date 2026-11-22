"""Equal total landmark presentations and update count, a secondary D checkpoint."""
import multiprocessing as mp
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
import time
import numpy as np
import torch
from train import CFG, ROOT, Unified, UncertaintyWeightedLoss, load, indices, inputs, random_phase, predict, focal_loss_with_logits, save_json


def run(job):
    f,s=job
    folder=ROOT/"matched"/f"dev{f}"/str(s)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():return
    fit,cal,ev=indices(d,f)
    torch.manual_seed(s);net=Unified("D");core=net.core;uw=UncertaintyWeightedLoss(3)
    with torch.no_grad():uw.log_sigma[2]=-1
    groups=[dict(params=core.risk_head.parameters(),lr=.001),dict(params=core.trajectory_head.parameters(),lr=.0005),
            dict(params=core.base_model.parameters(),lr=.0001),dict(params=uw.parameters(),lr=.01),
            dict(params=net.auxiliary.parameters(),lr=.001)]
    opt=torch.optim.Adam(groups,weight_decay=.0001)
    y=d["y"][fit].float();ratio=(len(y)-y.sum())/y.sum()
    w=torch.where(y>0,5*ratio,1.).double()
    gen=torch.Generator().manual_seed(s+1);started=time.monotonic()
    with (folder/"training.log").open("w",buffering=1) as log:
        for ep in range(1,61):
            net.train();order=fit[torch.multinomial(w,len(fit),replacement=True,generator=gen)]
            total=0.
            for j,b in enumerate(order.split(CFG["batch"])):
                torch.manual_seed(random_phase(s,ep,j))
                h=net.encode(*inputs(d,b))
                annual=core.compute_trajectory_loss(core.trajectory_head(h),d["annual"][b],d["active"][b])
                cls=focal_loss_with_logits(core.risk_head(h),d["y"][b,None].float(),2.,ratio.reshape(1))
                loss=(torch.exp(-uw.log_sigma[1:])*torch.stack([annual,cls])+uw.log_sigma[1:]).sum()
                loss=loss+CFG["aux_weight"]*net.auxiliary_loss(h,d["aux"][b],d["dom"][b],d["mask"][b])
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
                total+=loss.item()*len(b)
            print(ep,total/len(fit),time.monotonic()-started,file=log)
            if ep==10:fixed10=predict(net,d,ev)
    torch.save(net.state_dict(),folder/"epoch60.pt")
    np.savez_compressed(folder/"predictions.npz",rows=ev,p=predict(net,d,ev),fixed10=fixed10)
    save_json(folder/"complete.json",dict(steps=60*len(order.split(CFG["batch"])),presentations=60*len(fit),seconds=time.monotonic()-started))
    print("completed equal-presentation D",f,s,flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=3);args=parser.parse_args()
    d=load()
    with mp.get_context("fork").Pool(args.workers) as pool:
        list(pool.imap_unordered(run,[(f,s) for f in CFG["folds"] for s in CFG["seeds"]]))
