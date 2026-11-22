"""Create one development-only registry, using the frozen July input and endpoint."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
PROJECT = EXP.parents[1]
ORIGINAL = PROJECT / "server_repos/20260724/source/Causal-TabDiff"
DATA = PROJECT / "_audit_sources/20260724/Causal-TabDiff/data"
BANK = EXP / "clinical_paths/baseline_development_inputs"
SRC = EXP / "patient_diffusion/private"
PRIVATE = ROOT / "private"
SITES = ["right_upper", "right_middle", "right_lower", "left_upper", "left_lower", "other"]
os.umask(0o077)


def read_selected(name, columns, ids):
    file = DATA / "nlst.780.idc.delivery.052821" / f"nlst_780_{name}_idc_20210527.csv"
    return pd.concat([chunk.loc[chunk.pid.isin(ids)] for chunk in
                      pd.read_csv(file, usecols=["pid", *columns], chunksize=15000, low_memory=False)])


def save_json(file, obj):
    tmp = file.with_suffix(".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(file)


def prepare():
    sys.path.insert(0, str(ORIGINAL))
    from src.data.data_module_landmark import LandmarkDataset
    os.environ["ALPHA_TREATMENT_COLUMN"] = "baseline_cigsmok"
    patient_role = np.load(SRC / "cv0.npy")
    development = np.flatnonzero(patient_role == 2)
    assert len(development) == 10594 and len(patient_role)-len(development) == 21188
    person = np.load(SRC / "person.npy", mmap_mode="r")
    ix = np.flatnonzero(np.isin(person, development))
    ids = np.load(SRC / "ids.npy", mmap_mode="r")[development]
    who = np.searchsorted(development, person[ix])
    lm = np.load(SRC / "lm.npy", mmap_mode="r")[ix]
    folds = np.load(BANK / "folds.npy")
    y = np.load(BANK / "y.npy")
    lead = np.load(BANK / "lead.npy")
    assert folds.shape == (3,31587) and len(ix)==len(lm)==len(y)==31587
    assert np.array_equal(y, np.isin(lead, [1,2])) and np.all((folds==2).sum(0)==1)
    for role in folds:
        groups = [np.unique(who[role==k]) for k in range(3)]
        assert sum(map(len, groups))==len(ids)
    keys = pd.MultiIndex.from_arrays([ids[who], lm], names=["pid","landmark"])
    df = pd.read_pickle(DATA/"landmark_tables/unified_person_landmark_table.pkl").set_index(["pid","landmark"]).loc[keys].reset_index()
    src = LandmarkDataset(df, {0:0,1:1,2:2})
    cols = [src.baseline_cols + src.temporal_cols[f"t{t}"] for t in range(3)]
    x = np.stack([df[c].to_numpy(dtype="float32",na_value=np.nan) for c in cols], axis=1)
    valid = np.arange(3)<=lm[:,None]
    x[~valid] = 0
    x = np.nan_to_num(x)
    a = df.baseline_cigsmok.to_numpy(dtype="f")
    annual = np.stack(df.trajectory_target).astype("f")
    active = np.stack(df.trajectory_valid_mask).astype("f")
    np.testing.assert_array_equal(df.y_2year, y)
    np.testing.assert_array_equal(annual[:,:2].max(1), y)
    for i in np.linspace(0,len(ix)-1,97,dtype=int):
        sample = src[i]
        np.testing.assert_array_equal(sample["x"].numpy(), x[i,:lm[i]+1])
        assert sample["alpha_target"].item()==a[i]
    native = np.load(BANK/"native.npy",mmap_mode="r").reshape(len(ix),3,-1)
    np.testing.assert_array_equal(native[:,:,-1].sum(1), lm+1)
    names = json.loads((EXP/"conditional_trajectories/cohort_reference.json").read_text())["features"]
    np.testing.assert_allclose(native[np.arange(len(ix)),lm,names.index("age")],(x[:,0,0]-62)/6)
    np.testing.assert_array_equal(native[:,0,names.index("smoking")], a)
    # Only CT with a documented read qualifies. Unknown size in any recorded nodule
    # leaves the burden unknown. Other imaging and a missing CT do not yield a zero label.
    keys2=["pid","study_yr"]
    ct=read_selected("ctab",["study_yr","sct_ab_num","sct_ab_desc","sct_epi_loc","sct_long_dia"],ids)
    lesions=ct.loc[ct.sct_ab_desc==51].copy()
    ch=read_selected("ctabc",["study_yr","sct_ab_num","sct_ab_gwth","sct_ab_code"],ids)
    lesions=lesions.merge(ch, on=keys2+["sct_ab_num"],how="left",validate="one_to_one")
    lesions["unknown"]=~(lesions.sct_long_dia>0)
    lesions["conflict"]=lesions.sct_ab_code.notna() & (lesions.sct_ab_code!=51)
    lesions["site"]=lesions.sct_epi_loc.map({1:0,2:1,3:2,4:3,5:3,6:4}).fillna(5).astype(int)
    lesions["growing"]=lesions.sct_long_dia.where(lesions.sct_ab_gwth==2,0)
    idx=pd.MultiIndex.from_arrays([ids[who],lm],names=keys2)
    sites=lesions.groupby(keys2+["site"]).sct_long_dia.max().unstack().reindex(index=idx,columns=range(6)).fillna(0).to_numpy()
    grouped=lesions.groupby(keys2)
    bad=grouped[["unknown","conflict"]].max().reindex(idx).eq(True).any(axis=1).to_numpy()
    grow=grouped.growing.max().reindex(idx).fillna(0).to_numpy()
    ct_present=native[np.arange(len(ix)),lm,names.index("ct_available")]>0
    read=native[np.arange(len(ix)),lm,names.index("valid_read")]>0
    persons=read_selected("prsn",["scr_days0","scr_days1","scr_days2","candx_days"],ids).set_index("pid").loc[ids]
    tumors=read_selected("canc",["candx_days"],ids).groupby("pid").candx_days.min().reindex(ids)
    dxday=pd.concat([persons.candx_days,tumors],axis=1).min(axis=1).to_numpy()[who]
    day=persons[["scr_days0","scr_days1","scr_days2"]].to_numpy()[who,lm]
    dated=np.isfinite(day)&(~np.isfinite(dxday)|(day<dxday))
    pos=np.arange(len(ix))
    nxt=np.minimum(pos+1,len(pos)-1)
    adjacent=(who[nxt]==who)&(lm[nxt]==lm+1)
    usable=adjacent&ct_present[nxt]&read[nxt]&dated[nxt]&~bad[nxt]
    mm=np.column_stack([sites.max(1),sites,grow])
    assert np.all(mm>=0) and np.all(mm[:,-1]<=mm[:,0])
    target=np.zeros_like(mm,dtype="f")
    target[usable]=np.log1p(mm[nxt[usable]])
    dom=np.where(target[:,0]>0, target[:,1:7].argmax(1)+1, 0)
    assert np.all(folds[:,pos[usable]]==folds[:,nxt[usable]]) and not usable[lm==2].any()
    PRIVATE.mkdir(exist_ok=True,mode=0o700)
    np.savez(PRIVATE/"registry.npz", x=x,a=a,lm=lm,who=who,folds=folds,y=y,annual=annual,active=active,lead=lead,
             aux=target,dom=dom,mask=usable,ct=ct_present)
    counts=[]
    for t in range(3):
        sel=lm==t
        counts.append(dict(landmark=t,rows=int(sel.sum()),positive=int(y[sel].sum()),eligible_pair=int((sel&adjacent).sum()),
                           next_ct=int((sel&adjacent&ct_present[nxt]).sum()),auxiliary=int((sel&usable).sum()),
                           positive_auxiliary=int((sel&usable&(y==1)).sum()),positive_size=int((sel&usable&(dom>0)).sum())))
    save_json(ROOT/"population.json",dict(people=len(ids),rows=len(ix),positive=int(y.sum()),positive_people=len(np.unique(who[y==1])),
             sites=SITES,columns=cols,landmarks=counts,roles=[[int(sum(f==k)) for k in range(3)] for f in folds]))
    print("Prepared development cohort",len(ix),int(usable.sum()),flush=True)


if __name__=="__main__":
    prepare()
