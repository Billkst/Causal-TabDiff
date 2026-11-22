"""Patient-paired intervals for secondary risk metrics at frozen thresholds."""
import json
import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from analyze import folder,grouping,boot_people,SEEDS,NINE
from registry_data import ROOT,PRIVATE


@njit
def recalculated(y,p,who,threshold,weights):
    result=np.empty((8,weights.shape[0]))
    for j,w in enumerate(weights):
        n=0.;events=0.
        for i in range(len(y)):
            a=w[who[i]]
            n+=a;events+=a*y[i]
        nneg=n-events
        prev_pos=0.;seen=0.;rank=0.;tp=0.;fp=0.;cap1=0.;cap5=0.;i=0
        while i<len(y):
            h=i;positive=0.;mass=0.
            while h<len(y) and p[h]==p[i]:
                a=w[who[h]]
                positive+=a*y[h];mass+=a;h+=1
            negative=mass-positive
            rank+=negative*(prev_pos+positive*.5)
            if p[i]>=threshold:
                tp+=positive;fp+=negative
            if mass>0:
                cap1+=positive*max(0.,min(1.,(.01*n-seen)/mass))
                cap5+=positive*max(0.,min(1.,(.05*n-seen)/mass))
            prev_pos+=positive;seen+=mass;i=h
        fn=events-tp;tn=nneg-fp
        denom=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        result[:,j]=np.array([rank/(events*nneg),2*tp/(2*tp+fp+fn),(tp*tn-fp*fn)/denom if denom>0 else 0.,
            cap1/events,cap1/(.01*events),cap5/events,cap5/(.05*events),cap5/(.05*n)])
    return result


def run():
    out=ROOT/"statistics"
    means=pd.read_csv(out/"means.csv").set_index("model")
    champion=means.loc[NINE].ap.idxmax()
    models=list("ABCD")+[champion]
    edges=[("B","A"),("C","B"),("C","D"),("C","A"),("C",champion)]
    rng=np.random.default_rng(20261122);data=np.load(PRIVATE/"registry.npz")
    group={m:np.zeros((8,2000)) for m in models}
    for f in range(3):
        ev=np.flatnonzero(data["folds"][f]==2)
        y=data["y"][ev].astype(np.int64);who,case=grouping(data["who"][ev],y)
        w=np.ascontiguousarray(boot_people(case,2000,rng).T)
        for s in SEEDS:
            for model in models:
                home=folder(f,s,model)
                with np.load(home/"predictions.npz") as file:p=file["p"].astype("d")
                cut=json.loads((home/"complete.json").read_text())["threshold"]
                order=np.argsort(-p,kind="stable")
                q=recalculated(y[order],p[order],who[order],cut,w)
                ww=w[0][who];cl=p>=cut
                np.testing.assert_allclose(q[:3,0],[roc_auc_score(y,p,sample_weight=ww),f1_score(y,cl,sample_weight=ww),
                                      matthews_corrcoef(y,cl,sample_weight=ww)],atol=1e-12)
                unit=recalculated(y[order],p[order],who[order],cut,np.ones((1,len(case))))[:,0]
                from private_benchmark_math import endpoints
                metrics=("auc","f1","mcc","coverage1","enrichment1","coverage5","enrichment5","ppv5")
                exact=endpoints(y,p,cut)
                np.testing.assert_allclose(unit,[exact[k] for k in metrics],atol=1e-12)
                group[model]+=q/9
    rows=[]
    for a,b in edges:
        draws=group[a]-group[b]
        for j,k in enumerate(metrics):
            lo,hi=np.quantile(draws[j],[.025,.975])
            rows.append(dict(pair=a+"-"+b,metric=k,delta=means.loc[a,k]-means.loc[b,k],lo=lo,hi=hi))
    pd.DataFrame(rows).to_csv(out/"secondary_paired.csv",index=False)
    print("Verified paired secondary metrics",len(rows),flush=True)


if __name__=="__main__":run()
