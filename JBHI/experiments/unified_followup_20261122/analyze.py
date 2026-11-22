"""Paired development estimates from locked, patient-indexed predictions."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from registry_data import EXP, PRIVATE, ROOT, save_json

sys.path.insert(0,str(EXP/"clinical_paths"))
from private_benchmark_math import WeightedAP, boot_people, endpoints

ARMS=list("ABCD")
NINE=["LR","RF","HGB","XGBoost","iTransformer","TimeXer","SSSD","TSDiff","SurvTraj"]
EDGES=[("B","A"),("C","A"),("C","B"),("C","D")]+[("C",name) for name in NINE]
FIXED=[("C10","D10"),("C10","D60")]
SEEDS=[42,52,62]


def folder(f,s,name):
    tree="runs" if name in ARMS else "baselines"
    return ROOT/tree/f"dev{f}"/str(s)/name


def grouping(who,y):
    unique,idx=np.unique(who,return_inverse=True)
    case=np.zeros(len(unique),dtype=np.int8)
    np.maximum.at(case,idx,y)
    return idx,case


def add_aux(acc,m,who,w,true,dom,vals,pr,meta):
    actual=np.expm1(true)
    estimates=np.expm1(vals)
    positive=dom>0
    pred=pr.argmax(1)
    v={"overall_mae":abs(estimates[:,0]-actual[:,0]),"site_mae":abs(estimates[:,1:7]-actual[:,1:7]).mean(1),
       "growth_mae":abs(estimates[:,7]-actual[:,7]),"dominant_accuracy":(pred==dom),
       "positive_measured_mae":abs(estimates[:,0]-actual[:,0]),"dominant_accuracy_positive":(pred==dom),
       "brier_any":((1-pr[:,0])-positive)**2,"pred_nonzero":(estimates[:,0]>0),
       "max_disagree":abs(vals[:,1:7].max(1)-vals[:,0])>1e-6,"growth_gt_max":vals[:,7]>vals[:,0]+1e-6,
       "inconsistent_given_nonzero":abs(vals[:,1:7].max(1)-vals[:,0])>1e-6}
    row=dict(meta,rows=len(dom),positive=int(sum(positive)))
    for metric,value in v.items():
        keep=(positive if metric in ("positive_measured_mae","dominant_accuracy_positive") else
              estimates[:,0]>0 if metric=="inconsistent_given_nonzero" else np.ones(len(dom),dtype=bool))
        num=np.bincount(who[keep],weights=value[keep],minlength=len(w))
        den=np.bincount(who[keep],minlength=len(w))
        row[metric]=float(num.sum()/den.sum()) if den.sum() else None
        if metric in ("overall_mae","site_mae","growth_mae","brier_any","max_disagree","positive_measured_mae"):
            acc[(m,metric)]=np.divide(num@w,den@w)
    return row


def summarize():
    dat=np.load(PRIVATE/"registry.npz")
    rows=[]; aux=[];occ=[];modality=[]; fixedrows=[]; points={}; draw_results=[]; aux_draws=[]
    rng=np.random.default_rng(20261122)
    for f in range(3):
        tr,ca,ev=(np.flatnonzero(dat["folds"][f]==k) for k in range(3))
        y=dat["y"][ev].astype(int)
        inv,case=grouping(dat["who"][ev],y)
        weights=boot_people(case,2000,rng)
        seed_draws=[];fold_aux=[]
        for s in SEEDS:
            maps={};aa={}
            for name in ARMS+NINE:
                home=folder(f,s,name)
                meta=json.loads((home/"complete.json").read_text())
                with np.load(home/"predictions.npz") as result:
                    np.testing.assert_array_equal(result["rows"],ev)
                    np.testing.assert_array_equal(result["cal_rows"],ca)
                    p,pc=result["p"],result["cal"]
                    assert (np.isfinite(p).all() and len(pc)==len(ca) and len(p)==len(ev))
                    from sklearn.metrics import precision_recall_curve
                    pr,re,th=precision_recall_curve(dat["y"][ca],pc)
                    f1=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-15)
                    assert float(th[np.where(f1==f1.max())[0][-1]])==meta["threshold"]
                    met=endpoints(y,p,meta["threshold"])
                    rows.append(dict(fold=f,seed=s,model=name,**met))
                    points[f,s,name]=met["ap"]
                    maps[name]=p
                    if name in ARMS:
                        maps[name+"10"]=result["fixed10"]
                        fixedrows.append(dict(fold=f,seed=s,model=name+"10",ap=average_precision_score(y,maps[name+"10"])))
                    if name in ("B","C","D"):
                        ix=result["aux_rows"];active=dat["mask"][ev]
                        np.testing.assert_array_equal(ev[active],ix)
                        aux.append(add_aux(aa,name,inv[active],weights,dat["aux"][ix],dat["dom"][ix],
                                           result["sizes"],result["classes"],dict(fold=f,seed=s,model=name)))
                    if name in ARMS:
                        for ct in (False,True):
                            group=dat["ct"][ev]==ct
                            modality.append(dict(fold=f,seed=s,model=name,current_ct=ct,n=int(group.sum()),
                                cases=int(y[group].sum()),ap=average_precision_score(y[group],p[group])))
                        for eligible in (False,True):
                            sub=dat["mask"][ev]==eligible
                            occ.append(dict(fold=f,seed=s,model=name,has_next_measurement=eligible,
                                 n=int(sub.sum()),cases=int(y[sub].sum()),ap=average_precision_score(y[sub],p[sub])))
            with np.load(ROOT/"matched"/f"dev{f}"/str(s)/"predictions.npz") as b:
                np.testing.assert_array_equal(b["rows"],ev)
                np.testing.assert_array_equal(b["fixed10"],maps["D10"])
                maps["D60"]=b["p"]
                fixedrows.append(dict(fold=f,seed=s,model="D60",ap=average_precision_score(y,b["p"])))
            boot={}
            for name in set(sum(([a,b] for a,b in EDGES+FIXED),[])):
                p=maps[name]
                ap=WeightedAP(dat["who"][ev],y,p)
                v=np.concatenate([ap(weights[:,j:j+128]) for j in range(0,2000,128)])
                for j in [0,15,31]:
                    np.testing.assert_allclose(v[j],average_precision_score(y,p,sample_weight=weights[inv,j]),atol=1e-12,rtol=0)
                boot[name]=v
                points[f,s,name]=float(average_precision_score(y,p))
            seed_draws.append(boot);fold_aux.append(aa)
        draw_results.append({k:np.mean([v[k] for v in seed_draws],0) for k in seed_draws[0]})
        aux_draws.append({k:np.mean([v[k] for v in fold_aux],0) for k in fold_aux[0]})
    out=ROOT/"statistics";out.mkdir(exist_ok=True)
    for fname,data in [("metrics",rows),("auxiliary",aux),("selected_observations",occ),("fixed_budget",fixedrows),("current_ct",modality)]:
        pd.DataFrame(data).to_csv(out/(fname+".csv"),index=False)
    means=pd.DataFrame(rows).groupby("model").mean(numeric_only=True).drop(columns=["fold","seed"])
    means.to_csv(out/"means.csv")
    boot={k:np.mean([d[k] for d in draw_results],0) for k in draw_results[0]}
    np.savez_compressed(out/"patient_bootstrap_ap.npz",**boot)
    save_json(out/"ap_intervals.json",{k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()})
    def contrast(pair):
        a,b=pair;vals=np.array([[points[f,s,a]-points[f,s,b] for s in SEEDS] for f in range(3)])
        return vals,boot[a]-boot[b]
    center=np.array([contrast(e)[0].mean() for e in EDGES])
    draws=np.stack([contrast(e)[1] for e in EDGES]);sd=draws.std(1,ddof=1)
    critical=float(np.quantile(np.max((draws-center[:,None])/np.maximum(sd[:,None],1e-15),axis=0),.95,method="higher"))
    pairs=[]
    for i,e in enumerate(EDGES+FIXED):
        v,b=contrast(e);lo,hi=np.quantile(b,[.025,.975])
        pairs.append(dict(pair="-".join(e),delta=v.mean(),lo=lo,hi=hi,
                    simultaneous_lower=v.mean()-critical*sd[i] if i<len(EDGES) else None,
                    fold_means=v.mean(1).tolist(),seed_means=v.mean(0).tolist(),positive_cells=int((v>0).sum())))
    save_json(out/"paired.json",dict(draws=2000,critical=critical,edges=pairs))
    auxmeans=pd.DataFrame(aux).groupby("model").mean(numeric_only=True).drop(columns=["fold","seed"])
    auxmeans.to_csv(out/"aux_means.csv")
    am=[]
    for name,metric in aux_draws[0]:
        if name!="C":continue
        delta=np.mean([d["C",metric]-d["B",metric] for d in aux_draws],0)
        am.append(dict(metric=metric,delta=auxmeans.loc["C",metric]-auxmeans.loc["B",metric],
                  lo=float(np.quantile(delta,.025)),hi=float(np.quantile(delta,.975))))
    save_json(out/"paired_measurement.json",am)
    print("Statistics complete",len(rows),"fold/seed/model estimates",flush=True)


if __name__=="__main__":summarize()
