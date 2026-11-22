"""Separate calibration, lock and external-fold evaluation entry points."""
import argparse
import json
import multiprocessing as mp
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from repair import ROOT, ARCHIVE, Variant, Repaired
from engine import SCREEN, read_role, train_model, predict, measurement, save_json, state

PROTOCOL = json.loads((ROOT/"protocol.json").read_text())


def screen_path(name):
    return ROOT/"private"/"screen"/name


def screen_fit(name):
    train_model(0,42,SCREEN[name],screen_path(name))
    print("calibration complete",name,flush=True)


def lock():
    fits = {key:json.loads((screen_path(key)/"fit.json").read_text()) for key in SCREEN}
    for arm, archived in [("C0","C"),("D0","D")]:
        with np.load(ARCHIVE/"runs"/"dev0"/"42"/archived/"predictions.npz") as old, np.load(screen_path(arm)/"calibration.npz") as new:
            np.testing.assert_array_equal(old["cal_rows"],new["rows"])
            np.testing.assert_array_equal(old["cal"],new["p"])
    c = fits["C0"]
    choices = [k for k in ("sample_only","loss_only") if fits[k]["cal_ap"]>c["cal_ap"] and fits[k]["cal_ap10"]>=c["cal_ap10"]]
    best = max(choices,key=lambda k:fits[k]["cal_ap"]) if choices else "C0"
    p = fits["P2"]
    sparse = (p["auxiliary"]["zero_specificity"]>c["auxiliary"]["zero_specificity"] and
              p["auxiliary"]["nonzero_sensitivity"]>0 and p["cal_ap"]>=c["cal_ap"]-.003)
    decision = dict(variant=asdict(Variant(prefix=True,weighting=SCREEN[best].weighting,sparse=sparse)),
                    rule=PROTOCOL["selection"], calibration={k:{m:v[m] for m in ("selected","epochs","cal_ap","cal_ap10","auxiliary")} for k,v in fits.items()},
                    archive_calibration_replay_exact=["C","D"],screen_fold=0,screen_seed=42)
    dest = ROOT/"locked.json"
    if dest.exists():
        assert json.loads(dest.read_text())==decision
    else:
        save_json(dest,decision)
    stamp=ROOT/"freeze_record.json"
    if stamp.exists():
        assert json.loads(stamp.read_text())["variant"]==decision["variant"]
    else:
        save_json(stamp,dict(utc=datetime.now(timezone.utc).isoformat(),variant=decision["variant"],
                             parent=PROTOCOL["audit_commit"]))
    print("locked",decision["variant"],flush=True)


def final_variant(name):
    spec=json.loads((ROOT/"locked.json").read_text())["variant"]
    return Variant(**dict(spec,diffusion=name=="R"))


def fit_path(f,s,name):
    v=final_variant(name)
    if (f,s)==(0,42):
        for label,arm in SCREEN.items():
            if arm==v:
                return screen_path(label)
    return ROOT/"private"/"validation"/f"dev{f}"/str(s)/name


def final_job(job):
    f,s,name=job
    v=final_variant(name)
    path=fit_path(f,s,name)
    train_model(f,s,v,path)
    dest=ROOT/"private"/"validation"/f"dev{f}"/str(s)/name
    dest.mkdir(exist_ok=True,parents=True)
    if (dest/"evaluation.json").exists():
        return
    net=Repaired(v)
    net.load_state_dict(state(path/"best.pt"))
    ev,rows=read_role(f,2)
    p=predict(net,ev,torch.arange(len(rows)))
    aux,output=measurement(net,ev)
    assert np.isfinite(p).all() and all(np.isfinite(a).all() for a in output.values())
    np.savez_compressed(dest/"predictions.npz",rows=rows,p=p,aux_rows=rows[ev["mask"].numpy()],**output)
    save_json(dest/"evaluation.json",dict(fold=f,seed=s,arm=name,fit=str(path.relative_to(ROOT)),auxiliary=aux))
    print("validation complete",f,s,name,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["screen","lock","final"])
    parser.add_argument("--workers",type=int,default=5)
    args=parser.parse_args()
    assert json.loads((ROOT/"checks.json").read_text())["pass_checks"]
    if args.stage=="lock":
        lock()
    elif args.stage=="screen":
        with mp.get_context("fork").Pool(args.workers) as pool:
            list(pool.imap_unordered(screen_fit,SCREEN))
    else:
        final_variant("R")
        assert json.loads((ROOT/"freeze_record.json").read_text())["variant"]==asdict(final_variant("R"))
        jobs=[(f,s,name) for f in PROTOCOL["validation"]["folds"] for s in PROTOCOL["validation"]["seeds"] for name in ("R","N")]
        with mp.get_context("fork").Pool(args.workers) as pool:
            list(pool.imap_unordered(final_job,jobs))


if __name__=="__main__":
    main()
