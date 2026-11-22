"""Reload selected weights. Reproduce both calibration and held predictions."""
import argparse
import json
import multiprocessing as mp
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from baseline import DEEP, TREES, UP, build, views
from backbones import predict as reference_predict
from train import CFG, indices, load, predict
from model import Unified
from registry_data import ROOT,save_json
from analyze import folder


def check(job):
    f,s,name=job
    home=folder(f,s,name)
    out=ROOT/"replay"/f"dev{f}"/str(s)/f"{name}.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():return
    info=json.loads((home/"complete.json").read_text())
    tr,ca,ev=indices(d,f)
    with np.load(home/"predictions.npz") as z:
        old=[z["cal"],z["p"]]
        if name in "ABCD" and len(name)==1:
            net=Unified(name)
            net.load_state_dict(torch.load(home/"best.pt",weights_only=True))
            values=[predict(net,d,k) for k in (ca,ev)]
            if name!="A":
                p,measures,probs=predict(net,d,ev,True)
                mask=d["mask"][ev].numpy()
                np.testing.assert_array_equal(p,values[1])
                np.testing.assert_array_equal(z["sizes"],measures[mask])
                np.testing.assert_array_equal(z["classes"],probs[mask])
        else:
            data={k:v.numpy() for k,v in d.items()}
            sc,x,features=views(data,tr.numpy())
            with np.load(home/"scaling.npz") as scale:
                np.testing.assert_array_equal(scale["mean"],sc.mean_)
                np.testing.assert_array_equal(scale["scale"],sc.scale_)
            if name in TREES:
                key,norm,model=joblib.load(home/"best.joblib")
                values=[]
                for k in (ca,ev):
                    raw=features[key][k.numpy()]
                    values.append(model.predict_proba(norm.transform(raw) if norm else raw)[:,1])
            else:
                nntr=tr.numpy();lead=data["lead"][nntr]
                event=np.where(np.isin(lead,[1,2]),lead,0)
                torch.manual_seed(s)
                net=build(name,x[nntr],event,UP,"cpu")
                net.load_state_dict(torch.load(home/"best.pt",weights_only=True),assign=name=="SSSD")
                values=[reference_predict(net,name in ("iTransformer","TimeXer"),torch.from_numpy(x[k.numpy()])) for k in (ca,ev)]
        errors=[float(np.max(np.abs(a-b))) for a,b in zip(values,old)]
        for a,b in zip(values,old):np.testing.assert_allclose(a,b,rtol=1e-5,atol=2e-6)
    save_json(out,dict(model=name,fold=f,seed=s,cal_rows=len(ca),eval_rows=len(ev),max_abs=errors))
    print("replayed",name,f,s,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=4);parser.add_argument("--available",action="store_true")
    parser.add_argument("--names",nargs="+",default=list("ABCD")+TREES+DEEP);args=parser.parse_args()
    d=load()
    jobs=[(f,s,n) for f in range(3) for s in CFG["seeds"] for n in args.names]
    if args.available:jobs=[j for j in jobs if (folder(*j)/"complete.json").exists()]
    with mp.get_context("fork").Pool(args.workers) as pool:list(pool.imap_unordered(check,jobs))
