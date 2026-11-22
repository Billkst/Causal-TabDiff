"""Common patient assignments, feature views and registered-prediction storage."""
import json
import os
from pathlib import Path
import shutil
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve
from data import HERE,history,sha

SEEDS=(42,52,62)

@dataclass
class Study:
    x: np.ndarray
    pid: np.ndarray
    lm: np.ndarray
    fit: np.ndarray
    dev: np.ndarray
    folds: np.ndarray
    target: dict
    schema: dict

    def columns(self,names):
        return [self.schema['features'].index(n) for n in names]

def load(data):
    schema=json.loads((data/'schema.json').read_text())
    if any(sha(data/p)!=v for p,v in schema['files'].items()):
        raise ValueError('Frozen data integrity check failed')
    with np.load(data/'clinical.npz') as ar:
        pid=ar['pid'][ar['person']]
        lm=ar['lm']
        fold=ar['fold']
        x=history(ar['panel'],ar['person'],lm)
    fit=np.flatnonzero(fold==-1)
    dev=np.flatnonzero(fold>=0)
    assert np.array_equal(fit,np.arange(len(fit))) and not np.intersect1d(pid[fit],pid[dev]).size
    with np.load(data/'fit_targets.npz') as ar:
        target=dict(ar)
    assert len(target['y'])==len(fit)
    return Study(x,pid,lm,fit,dev,fold[dev],target,schema)

def clinical_views(s):
    x,lm=s.x,s.lm
    chosen=[c for c in s.schema['features'] if c.startswith(('scr_','global_','finding_')) or
            c in ('age','sex','smoking','ct_available','valid_read','exam_gap','screen_day','other_investigate','changed_attenuation')]
    idx=s.columns(chosen)
    h=x[:, :, idx]
    row=np.arange(len(x))
    cur=h[row,lm]
    prev=h[row,np.maximum(0,lm-1)]*(lm>0)[:,None]
    size=cur[:,chosen.index('global_size')]
    diff=size-prev[:,chosen.index('global_size')]
    cap=np.column_stack([np.maximum(size-np.log1p(t)/2,0) for t in (4,6,8,12,20)])
    current_ct=cur[:,chosen.index('ct_available')]
    former_pos=prev[:,[chosen.index('scr_res_'+str(i)) for i in (4,5,6)]].sum(1)
    past=x[:,:,s.columns(['global_size','scr_res_4','scr_res_5','scr_res_6','ct_available'])].max(1)
    compact=np.column_stack([cur,prev,past,np.eye(3)[lm],cap,diff,
                             former_pos*size,current_ct*cur[:,chosen.index('age')],size*cur[:,chosen.index('global_growing')]]).astype('float32')
    # The complete native sequence, the compact view and the distinct-site histories
    # are all available to the learning algorithms, from exactly the same clinical data.
    return {'compact':compact,'native':x.reshape(len(x),-1),
            'combined':np.column_stack([compact,x.reshape(len(x),-1)]).astype('float32')}

def start(out,method,seed,files,schema=None):
    os.umask(0o077)
    out.mkdir(parents=True,mode=0o700,exist_ok=False)
    snap=out/'source'
    snap.mkdir()
    common=['data.py','study.py','protocol.json']
    for path in dict.fromkeys([*common,*files]):
        shutil.copyfile(HERE/path,snap/path)
    info=dict(method=method,seed=seed,sources={p.name:sha(p) for p in snap.iterdir()})
    if schema is not None:
        info['data_fingerprints']=schema['files']
        if 'localization' in schema:
            info['detector_sha']=schema['localization']['detector_sha']
    (out/'manifest.json').write_text(json.dumps(info,indent=2))
    return (out/'training.log').open('x',buffering=1)

def say(log,message):
    print(message,flush=True)
    print(message,file=log,flush=True)

def finish(out):
    files=sorted((out/'predictions').glob('*.npy'))
    if not files:
        raise ValueError('Empty prediction inventory')
    (out/'complete.json').write_text(json.dumps({'predictions':{p.name:sha(p) for p in files}},indent=2))

def save_pred(out,setting,scores):
    scores=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(scores).all() or np.max(scores)>1 or np.min(scores)<0:
        raise ValueError('Invalid predicted probability')
    folder=out/'predictions'
    folder.mkdir(exist_ok=True)
    with (folder/(setting+'.npy')).open('xb') as f:
        np.save(f,scores)

def threshold(y,p):
    precision,recall,cuts=precision_recall_curve(y,p)
    v=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return cuts[np.flatnonzero(v==v.max())[-1]]

def select(out,y,fold):
    manifest=json.loads((out/'complete.json').read_text())['predictions']
    items=[]
    for filename,digest in manifest.items():
        path=out/'predictions'/filename
        assert sha(path)==digest
        p=np.load(path)
        assert p.shape==y.shape
        items.append((path.stem,p))
    scores=np.empty(len(y))
    binary=np.empty(len(y),dtype=bool)
    chosen=[]
    for f in range(3):
        cal=fold!=f
        choices=[average_precision_score(y[cal],p[cal]) for _,p in items]
        best=int(np.argmax(choices))
        key,p=items[best]
        cut=threshold(y[cal],p[cal])
        scores[~cal]=p[~cal]
        binary[~cal]=p[~cal]>=cut
        chosen.append(dict(fold=f,setting=key,threshold=float(cut),cal_ap=float(choices[best])))
    return scores,binary,chosen
