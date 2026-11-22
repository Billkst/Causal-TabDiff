"""Paired internal-development inference for the single frozen candidate."""
import itertools
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from repair import ARCHIVE, ROOT, Repaired
from engine import cutoff, predict, read_role, save_json, state
from analyze import NINE, WeightedAP, boot_people, endpoints, grouping
from secondary import recalculated
sys.path.insert(0, str(ROOT))
from run import PROTOCOL, SCREEN, final_variant, fit_path, screen_path

STATS = ROOT/"statistics"
SEEDS = PROTOCOL["validation"]["seeds"]
EDGES = [s.split("-") for s in PROTOCOL["validation"]["ap_edges"]]
SECONDARY = ["auc","f1","mcc","coverage1","enrichment1","coverage5","enrichment5","ppv5"]
AUX_METRICS = ["whole_mae","site_mae","growth_mae","presence_brier"]


def measurement_draws(name, f, s, data, ev, inv, weights):
    current = name in ("R","N")
    home = (ROOT/"private"/"validation" if current else ARCHIVE/"runs")/f"dev{f}"/str(s)/name
    mask = data["mask"].numpy()
    with np.load(home/"predictions.npz") as predictions:
        np.testing.assert_array_equal(ev[mask],predictions["aux_rows"])
        size = predictions["sizes_mm"] if current else np.expm1(predictions["sizes"])
        presence = 1-predictions["classes"][:,0]
    truth = data["aux"][mask].expm1().numpy()
    err = abs(truth-size).astype(float)
    values = np.column_stack([err[:,0],err[:,1:7].mean(1),err[:,7],(presence-(truth[:,0]>0))**2])
    who = inv[mask]
    n = weights.shape[0]
    den = np.bincount(who,minlength=n)@weights
    sums = np.stack([np.bincount(who,weights=x,minlength=n) for x in values.T])
    return values.mean(0), (sums@weights)/den


def obj(path):
    return json.loads(path.read_text())


def read_scores(name, f, s, ev, ca, cal_y):
    if name in ("R","N"):
        base = ROOT/"private"/"validation"/f"dev{f}"/str(s)/name
        fit = fit_path(f,s,name)
        meta = obj(fit/"fit.json")
        with np.load(fit/"calibration.npz") as c:
            np.testing.assert_array_equal(c["rows"],ca)
            threshold = cutoff(cal_y,c["p"])
        with np.load(base/"predictions.npz") as test:
            np.testing.assert_array_equal(test["rows"],ev)
            p = test["p"]
    else:
        base = ARCHIVE/("runs" if name in list("ABCD") else "baselines")/f"dev{f}"/str(s)/name
        meta = obj(base/"complete.json")
        with np.load(base/"predictions.npz") as test:
            np.testing.assert_array_equal(test["rows"],ev)
            np.testing.assert_array_equal(test["cal_rows"],ca)
            threshold = cutoff(cal_y,test["cal"])
            p = test["p"]
    assert threshold == meta["threshold"] and len(p)==len(ev) and np.isfinite(p).all()
    return p,threshold


def diagnostic_rows():
    groups = [("screen",0,42,name,screen_path(name)) for name in SCREEN]
    groups += [("validation",f,s,name,fit_path(f,s,name)) for f,s,name in itertools.product(range(3),SEEDS,("R","N"))]
    gradients, dynamics, budgets = [],[],[]
    for stage,f,s,name,path in groups:
        tags = dict(stage=stage,fold=f,seed=s,model=name)
        meta = obj(path/"fit.json")
        grad = pd.read_csv(path/"gradients.csv").assign(**tags)
        log = pd.read_csv(path/"training.csv").assign(**tags)
        assert np.isfinite(log.select_dtypes(include=np.number)).all().all()
        gradients.append(grad)
        dynamics.append(log)
        budgets.append(dict(**tags,epochs=meta["epochs"],selected=meta["selected"],seconds=meta["seconds"],
                            positive=int(log.positive.sum()),negative=int(log.negative.sum()),aux=int(log.aux.sum()),
                            positive10=int(log.positive.head(10).sum()),negative10=int(log.negative.head(10).sum()),
                            parameters=meta["parameters"],threshold=meta["threshold"]))
    pd.concat(gradients,ignore_index=True).to_csv(STATS/"gradients.csv",index=False)
    pd.concat(dynamics,ignore_index=True).to_csv(STATS/"dynamics.csv",index=False)
    pd.DataFrame(budgets).to_csv(STATS/"budgets.csv",index=False)
    prelogs=[]
    for path in sorted((ROOT/"private"/"pretraining").glob("dev*/*/pretrain.csv")):
        prelogs.append(pd.read_csv(path).assign(fold=int(path.parents[1].name[3:]),seed=int(path.parent.name)))
    pd.concat(prelogs,ignore_index=True).to_csv(STATS/"pretraining.csv",index=False)


def run():
    STATS.mkdir(exist_ok=True)
    diagnostic_rows()
    names = list("ABCD")+NINE+["R","N"]
    key_names = ["R","N","C","D","A","iTransformer"]
    rng = np.random.default_rng(PROTOCOL["validation"]["paired_seed"])
    reps = PROTOCOL["validation"]["paired_draws"]
    ap_draws = {n:np.zeros(reps) for n in names}
    other_draws = {n:np.zeros((len(SECONDARY),reps)) for n in key_names}
    aux_models = ["B","C","D","R","N"]
    aux_draws = {n:np.zeros((len(AUX_METRICS),reps)) for n in aux_models}
    aux_points = {n:np.zeros(len(AUX_METRICS)) for n in aux_models}
    fixed_rows, fixed_paired = [], []
    fixed_draws = {n:np.zeros(reps) for n in ("C","D","R","N")}
    rows, aux, replay = [], [], []
    old = pd.read_csv(ARCHIVE/"statistics"/"metrics.csv").set_index(["model","fold","seed"])
    with np.load(ARCHIVE/"private"/"registry.npz") as bank:
        folds,y_all,people = (bank[k] for k in ("folds","y","who"))
    assert len(np.unique(people))==10594 and len(y_all)==31587
    assert np.all((folds==2).sum(0)==1)
    for f in range(3):
        fit,ca,ev = [np.flatnonzero(folds[f]==role) for role in range(3)]
        assert sum(len(np.unique(people[part])) for part in (fit,ca,ev))==10594
        data,_ = read_role(f,2)
        y = y_all[ev].astype("int64")
        inv,case = grouping(people[ev],y)
        weights = boot_people(case,reps,rng)
        w = np.ascontiguousarray(weights.T)
        for s in SEEDS:
            for name in names:
                p,th = read_scores(name,f,s,ev,ca,y_all[ca])
                scores = endpoints(y,p,th)
                tags = dict(model=name,fold=f,seed=s)
                rows.append(dict(**tags,**scores))
                if name not in ("R","N"):
                    np.testing.assert_allclose(old.loc[(name,f,s),list(scores)].to_numpy(),list(scores.values()),atol=1e-12,rtol=0)
                else:
                    dest = ROOT/"private"/"validation"/f"dev{f}"/str(s)/name
                    aux.append(dict(**tags,**obj(dest/"evaluation.json")["auxiliary"]))
                    net = Repaired(final_variant(name))
                    net.load_state_dict(state(fit_path(f,s,name)/"best.pt"))
                    p2 = predict(net,data,torch.arange(len(ev)))
                    np.testing.assert_array_equal(p2,p)
                    replay.append(dict(**tags,max_abs_risk=float(np.max(abs(p2-p)))))
                func = WeightedAP(people[ev],y,p)
                b = np.concatenate([func(weights[:,i:i+128]) for i in range(0,reps,128)])
                assert abs(b[0]-average_precision_score(y,p,sample_weight=w[0,inv]))<1e-12
                ap_draws[name] += b/9
                if name in aux_models:
                    observed_aux, samples = measurement_draws(name,f,s,data,ev,inv,weights)
                    aux_points[name] += observed_aux/9
                    aux_draws[name] += samples/9
                if name in fixed_draws:
                    if name in ("R","N"):
                        net.load_state_dict(state(fit_path(f,s,name)/"epoch10.pt"))
                        fixed = predict(net,data,torch.arange(len(ev)))
                    else:
                        with np.load(ARCHIVE/"runs"/f"dev{f}"/str(s)/name/"predictions.npz") as checkpoint:
                            fixed = checkpoint["fixed10"]
                    fixed_rows.append(dict(**tags,ap=average_precision_score(y,fixed)))
                    func10 = WeightedAP(people[ev],y,fixed)
                    fixed_draws[name] += np.concatenate([func10(weights[:,i:i+128]) for i in range(0,reps,128)])/9
                if name in key_names:
                    order=np.argsort(-p,kind="stable")
                    score = p[order].astype("d")
                    secondary=recalculated(y[order],score,inv[order],th,w)
                    unit=recalculated(y[order],score,inv[order],th,np.ones((1,len(case))))[:,0]
                    np.testing.assert_allclose(unit,[scores[k] for k in SECONDARY],atol=1e-12)
                    other_draws[name] += secondary/9
        print("paired development fold",f,flush=True)
    raw = pd.DataFrame(rows)
    means=raw.groupby("model")[list(scores)].mean()
    means["ap_lo"]=[np.quantile(ap_draws[k],.025) for k in means.index]
    means["ap_hi"]=[np.quantile(ap_draws[k],.975) for k in means.index]
    raw.to_csv(STATS/"metrics.csv",index=False)
    means.to_csv(STATS/"means.csv")
    pd.DataFrame(aux).to_csv(STATS/"measurement.csv",index=False)
    grid=raw.pivot(index=["fold","seed"],columns="model",values="ap")
    diff=np.stack([ap_draws[a]-ap_draws[b] for a,b in EDGES])
    point=np.array([(grid[a]-grid[b]).mean() for a,b in EDGES])
    sd=diff.std(1,ddof=1)
    critical=np.quantile(((diff-point[:,None])/sd[:,None]).max(0),.95,method="higher")
    pairs, secondary = [],[]
    for i,(a,b) in enumerate(EDGES):
        v=grid[a]-grid[b]
        pairs.append(dict(pair=a+"-"+b,delta=point[i],lo=np.quantile(diff[i],.025),hi=np.quantile(diff[i],.975),
                          simultaneous_lower=point[i]-critical*sd[i],positive_cells=int((v>0).sum()),
                          fold_means=v.groupby("fold").mean().tolist(),seed_means=v.groupby("seed").mean().tolist()))
        if a in key_names and b in key_names:
            for j,k in enumerate(SECONDARY):
                interval=np.quantile(other_draws[a][j]-other_draws[b][j],[.025,.975])
                secondary.append(dict(pair=a+"-"+b,metric=k,delta=means.loc[a,k]-means.loc[b,k],lo=interval[0],hi=interval[1]))
    save_json(STATS/"paired.json",dict(draws=reps,simultaneous_critical=critical,edges=pairs))
    pd.DataFrame(secondary).to_csv(STATS/"secondary.csv",index=False)
    aux_contrasts = []
    for a,b in [("R","C"),("N","D"),("R","N"),("C","B")]:
        for j,k in enumerate(AUX_METRICS):
            lo,hi=np.quantile((aux_draws[a]-aux_draws[b])[j],[.025,.975])
            aux_contrasts.append(dict(pair=a+"-"+b,metric=k,delta=aux_points[a][j]-aux_points[b][j],lo=lo,hi=hi))
    pd.DataFrame(aux_contrasts).to_csv(STATS/"paired_measurement.csv",index=False)
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(STATS/"epoch10.csv",index=False)
    centers=fixed.groupby("model").ap.mean()
    for a,b in [("R","N"),("R","C"),("N","D"),("C","D")]:
        lo,hi=np.quantile(fixed_draws[a]-fixed_draws[b],[.025,.975])
        fixed_paired.append(dict(pair=a+"-"+b,delta=centers[a]-centers[b],lo=lo,hi=hi))
    pd.DataFrame(fixed_paired).to_csv(STATS/"epoch10_paired.csv",index=False)
    save_json(ROOT/"verification.json",dict(development_patients=10594,development_landmarks=len(y_all),positive=int(y_all.sum()),
              historical_metrics_verified=len(old),frozen_weight_replays=replay,all_new_predictions_exact=True,
              registered_ap_contrasts=len(EDGES),risk_models=len(names),new_fits=len(replay),
              private_predictions="local only",gpu_hours=0))
    print("All scores and frozen-weight replays verified",flush=True)


if __name__=="__main__":
    run()
