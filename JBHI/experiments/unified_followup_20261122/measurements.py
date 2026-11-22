"""Measure positive burden even when the most likely categorical output is no nodule."""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from train import load, CFG, indices, inputs
from model import Unified
from registry_data import ROOT


@torch.no_grad()
def run():
    d=load()
    output=[];at_risk=[]
    for f in CFG["folds"]:
        tr,_,ev=indices(d,f)
        tr=tr[d["mask"][tr]]
        ev=ev[d["mask"][ev]]
        truth=d["aux"][ev].expm1().numpy();dom=d["dom"][ev].numpy();pos=dom>0
        prevalence=float((d["dom"][tr]>0).float().mean())
        med=float(d["aux"][tr[d["dom"][tr]>0],0].expm1().median())
        for s in CFG["seeds"]:
            for arm in "BCD":
                home=ROOT/"runs"/f"dev{f}"/str(s)/arm
                net=Unified(arm)
                net.load_state_dict(torch.load(home/"best.pt",weights_only=True))
                net.eval()
                values=[];prob=[]
                for b in ev.split(512):
                    h=net.encode(*inputs(d,b))
                    logits,_=net.report(h)
                    positive_label=logits[:,1:].argmax(1)+1
                    _,v=net.report(h,positive_label)
                    values.append(v.expm1().numpy());prob.append(logits.softmax(1)[:,0].numpy())
                size=np.concatenate(values);p=1-np.concatenate(prob)
                pred=size[:,0]*p
                error=np.abs(size[:,1:7].max(1)-size[:,0])
                output.append(dict(fold=f,seed=s,model=arm,presence_ap=average_precision_score(pos,p),
                    prevalence=pos.mean(),train_prevalence=prevalence,presence_brier=np.mean((p-pos)**2),
                    prevalence_brier=np.mean((prevalence-pos)**2),conditional_positive_mae=np.abs(size[pos,0]-truth[pos,0]).mean(),
                    positive_median_mae=np.abs(med-truth[pos,0]).mean(),expected_diameter_mae=np.abs(pred-truth[:,0]).mean(),
                    always_zero_mae=truth[:,0].mean(),positive_route_disagreement=(error>1e-5).mean(),
                    positive_route_mean_gap_mm=error.mean(),positive_route_growth_exceeds=(size[:,7]>size[:,0]+1e-5).mean()))
                np.savez_compressed(home/"report_diagnostics.npz",rows=ev,conditional_sizes_mm=size,prob_positive=p)
    out=ROOT/"statistics";out.mkdir(exist_ok=True)
    pd.DataFrame(output).to_csv(out/"measurement_diagnostics.csv",index=False)


if __name__=="__main__":run()
