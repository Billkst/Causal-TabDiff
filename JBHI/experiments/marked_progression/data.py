"""Clinical observations and fit-only auxiliary targets are constructed separately."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'screen_dynamics'))
from cohort import DATA,TABLE_SHA,digest,eligible,sha,split

RAW=DATA/'nlst.780.idc.delivery.052821'
LOBE_NAMES=['right_upper','right_middle','right_lower','left_upper','left_lower','other']
LOC_FLAGS=['locrup','locrmid','locrlow','loclup','locllow']
BASE=['age','gender','race','cigsmok']
KEY=['pid','study_yr','sct_ab_num']
LOC_MAP={1:0,2:1,3:2,4:3,5:3,6:4,8:5}

def read(name,cols,ids):
    path=RAW/f'nlst_780_{name}_idc_20210527.csv'
    df=pd.read_csv(path,usecols=['pid',*cols],low_memory=False)
    return df[df.pid.isin(ids)].copy()

def clinical(ids):
    p=read('prsn',BASE+[f'{key}{t}' for t in range(3) for key in ['scr_res','scr_iso','scr_days']],ids).set_index('pid').loc[ids]
    visits=pd.MultiIndex.from_product([ids,range(3)],names=KEY[:2])
    pp=pd.DataFrame(index=visits)
    for key in ['scr_res','scr_iso','scr_days']:
        pp[key]=p[[f'{key}{t}' for t in range(3)]].to_numpy().reshape(-1)
    valid=np.isin(pp.scr_res,range(1,7))|np.isin(pp.scr_iso,range(1,7))
    pp['exam_day']=pp.scr_days.where(valid)
    prev_day=pp.exam_day.groupby(level=0).ffill().groupby(level=0).shift(1)
    gap=(pp.exam_day-prev_day)/365.25
    scr=read('screen',['study_yr','ctdxqual','techpara_kvp','techpara_ma','techpara_fov'],ids)
    scr=scr.groupby(KEY[:2]).mean().reindex(visits)
    ct=read('ctab',KEY[1:]+['sct_ab_desc','sct_epi_loc','sct_long_dia','sct_perp_dia','sct_pre_att','sct_margins','sct_found_after_comp'],ids)
    change=read('ctabc',KEY[1:]+['sct_ab_attn','sct_ab_gwth','sct_ab_preexist','sct_ab_invg','visible_days'],ids)
    ct=ct.merge(change,on=KEY,how='left',validate='one_to_one').join(pp.exam_day,on=KEY[:2])
    lesions=ct[ct.sct_ab_desc==51].copy()
    lesions['site']=lesions.sct_epi_loc.map(LOC_MAP).fillna(5).astype(int)
    # The new read may identify a lesion on earlier films; it does not create a prior measurement.
    age=(lesions.exam_day-lesions.visible_days)/365.25
    measured=lesions.sct_long_dia.where(lesions.sct_long_dia>0)
    size=np.log1p(measured.clip(upper=120))/2
    lesions['n']=1.
    lesions['size']=size
    lesions['short']=np.log1p(lesions.sct_perp_dia.clip(0,120))/2
    lesions['spiculated']=size.where(lesions.sct_margins==1,0)
    lesions['subsolid']=size.where(lesions.sct_pre_att.isin([2,3]),0)
    lesions['growing']=size.where(lesions.sct_ab_gwth==2,0)
    lesions['new']=size.where(lesions.sct_ab_preexist==1,0)
    lesions['duration']=age.where((lesions.sct_ab_preexist==2)&age.ge(0)).clip(0,6)
    lesions['compared_only']=size.where(lesions.sct_found_after_comp==1,0)
    names=['n','size','short','spiculated','subsolid','growing','new','duration','compared_only']
    by=lesions.groupby(KEY[:2]+['site']).agg({k:('sum' if k=='n' else 'max') for k in names})
    sites=pd.MultiIndex.from_product([ids,range(3),range(6)],names=[*KEY[:2],'site'])
    loc=by.reindex(sites).fillna(0).to_numpy(dtype=np.float32).reshape(len(ids),3,6,len(names))
    loc[:,:,:,0]=np.log1p(loc[:,:,:,0])
    blocks={}
    def add(key,values):
        blocks[key]=np.broadcast_to(values,(len(ids),3)).astype(np.float32)
    add('age',(p.age.to_numpy()[:,None]-62)/6)
    add('sex',p.gender.to_numpy()[:,None]-1)
    add('smoking',p.cigsmok.to_numpy()[:,None])
    for k in (1,2,3,4,5,6,99):add(f'race_{k}',(p.race.to_numpy()[:,None]==k))
    for k in range(3):add(f'year_{k}',np.arange(3)==k)
    def annual(v):return np.asarray(v).reshape(len(ids),3)
    add('ct_available',annual(scr.ctdxqual.notna()))
    for key in ['scr_iso','scr_res']:
        for k in range(1,7):add(f'{key}_{k}',annual(pp[key]==k))
    add('valid_read',annual(valid))
    add('screen_day',annual(pp.exam_day.fillna(0).clip(-180,1400))/365.25)
    add('known_exam_gap',annual(gap.notna()))
    add('exam_gap',annual(gap.fillna(0).clip(0,4)))
    for name,den in [('ctdxqual',3),('techpara_kvp',140),('techpara_ma',100),('techpara_fov',400)]:
        add(name,annual(scr[name].fillna(0))/den)
    counts=ct.groupby(KEY[:2]+['sct_ab_desc']).size().unstack(fill_value=0).reindex(visits,columns=range(51,66),fill_value=0)
    for k in counts:add(f'finding_{k}',np.log1p(annual(counts[k])))
    other=ct.assign(investigate=ct.sct_ab_invg.eq(2),changed=ct.sct_ab_attn.eq(2))
    ev=other.groupby(KEY[:2])[['investigate','changed']].max().reindex(visits,fill_value=False)
    add('other_investigate',annual(ev.investigate))
    add('changed_attenuation',annual(ev.changed))
    for j,k in enumerate(names):
        add('global_'+k, loc[:,:,:,j].max(2))
    for s,label in enumerate(LOBE_NAMES):
        for j,k in enumerate(names):
            add(label+'_'+k,loc[:,:,s,j])
    return np.stack(list(blocks.values()),axis=-1),list(blocks),names

def fit_marks(ids):
    p=read('prsn',['can_scr',*LOC_FLAGS,'loclin','locoth','locmed','loclhil','locrhil','loclmsb','locrmsb','loccar','locunk'],ids).set_index('pid').reindex(ids)
    c=read('canc',['first_lc','study_yr','candx_days','lc_order'],ids)
    counts=c.groupby('pid').size().reindex(ids,fill_value=0)
    flags=p[LOC_FLAGS].eq(1).to_numpy()
    flags[:,3]|=p.loclin.eq(1).to_numpy()
    other=p[['locoth','locmed','loclhil','locrhil','loclmsb','locrmsb','loccar']].eq(1).any(axis=1).to_numpy()
    flags=np.column_stack([flags,other])
    # Missing or multiple-primary marks remain compatible with all sites.
    unknown=(flags.sum(1)==0)|(counts.to_numpy()!=1)|p.locunk.eq(1).to_numpy()
    flags[unknown]=True
    route=p.can_scr.to_numpy(float)
    route=np.where(route==1,0,np.where(np.isin(route,[2,3]),1,np.where(route==4,2,-1)))
    return flags,route

def history(panel,person,lm):
    observed=np.arange(3)[None,:]<=lm[:,None]
    ans=panel[person]*observed[:,:,None]
    return np.concatenate([ans,np.broadcast_to(observed[:,:,None],(*observed.shape,1))],axis=2).astype('float32')

def prepare(out):
    os.umask(0o077)
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    clean=eligible()
    asset=DATA/'landmark_tables/unified_person_landmark_table.pkl'
    assert sha(asset)==TABLE_SHA
    df=pd.read_pickle(asset)
    non_test,_=split(df,42)
    y=df.groupby('pid').y_2year.max().reindex(non_test)
    ids,origval=train_test_split(non_test,test_size=.25,stratify=y,random_state=42)
    orig=json.loads((HERE.parent/'repair_validation/verified.json').read_text())
    assert [digest(ids),digest(origval)]==orig['manifest']['split_pid_sha']
    train_ids=np.setdiff1d(ids,clean.pid.unique())
    assert len(train_ids)==29664 and digest(train_ids)=='5b52c1ac1b191cdf5fb64800d79ba449ddc1beb18f346e02b17fae294e3bc6cc'
    clean_ids=np.unique(clean.pid)
    person_ids=np.r_[train_ids,clean_ids]
    frame=pd.concat([df[df.pid.isin(train_ids)].sort_values(['pid','landmark']),clean],ignore_index=True)
    mapping=pd.Index(person_ids).get_indexer(frame.pid)
    assert (mapping>=0).all()
    lm=frame.landmark.to_numpy(int)
    fit=mapping<len(train_ids)
    y=frame.y_2year.to_numpy(int)
    lead=frame.cancyr.to_numpy()-lm
    np.testing.assert_array_equal(y,(lead==1)|(lead==2))
    labels=np.nan_to_num(lead[fit],nan=0).astype(int)
    assert ((labels>=0)&(labels<=7-lm[fit])).all()
    clean_y=clean.groupby('pid').y_2year.max().reindex(clean_ids)
    fold=np.full(len(person_ids),-1)
    for j,(_,v) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261120).split(clean_ids,clean_y)):
        fold[len(train_ids)+v]=j
    panel,names,fields=clinical(person_ids)
    flags,route=fit_marks(train_ids)
    np.savez_compressed(out/'clinical.npz',panel=panel,pid=person_ids,person=mapping,lm=lm,fold=fold[mapping])
    np.savez_compressed(out/'fit_targets.npz',y=y[fit],lead=labels,sites=flags[mapping[fit]],route=route[mapping[fit]])
    np.savez_compressed(out/'development_targets.npz',y=y[~fit])
    raw_checks=json.loads((HERE.parent/'final_validation/availability.json').read_text())['raw_sha']
    assert all(sha(RAW/k)==v for k,v in raw_checks.items())
    # Check row-level logical horizon exclusion independently of the history constructor.
    take=np.linspace(0,len(lm)-1,2500,dtype=int)
    xp=history(panel,mapping[take],lm[take])
    for t in (0,1):
        alter=panel.copy()
        alter[:,t+1:]=831
        m=take[lm[take]==t]
        np.testing.assert_array_equal(history(alter,mapping[m],lm[m]),history(panel,mapping[m],lm[m]))
    assert np.isfinite(xp).all()
    label_mask=y[fit]==1
    info=dict(features=names,site_fields=fields,sites=LOBE_NAMES,train_rows=int(fit.sum()),
              train_patients=len(train_ids),dev_rows=int((~fit).sum()),dev_patients=len(clean_ids),
              fit_positive_rows=int(label_mask.sum()),fit_positive_patients=int(len(np.unique(frame.pid[fit][label_mask]))),
              fit_mark_known_site_rows=int((flags[mapping[fit]][label_mask].sum(1)<6).sum()),
              fit_known_route_rows=int((route[mapping[fit]][label_mask]>=0).sum()),
              raw_sha=raw_checks,table_sha=TABLE_SHA,source_sha=sha(Path(__file__)),
              files={p.name:sha(p) for p in out.glob('*.npz')})
    (out/'schema.json').write_text(json.dumps(info,indent=2))
    print({k:info[k] for k in ('train_patients','dev_patients','fit_positive_rows','fit_mark_known_site_rows','fit_known_route_rows')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    prepare(p.parse_args().out)
