"""Fixed-bank baselines. Every candidate configuration saves weights and predictions."""
import argparse
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from study import clinical_views,finish,load,save_pred,say,start,SEEDS

def models(name,seed,prevalence):
    if name=='LR':
        return [(f'c{i}',LogisticRegression(C=c,max_iter=1400,tol=3e-5,random_state=seed))
                for i,c in enumerate((.01,.1,1.,3.))]
    if name=='RF':
        return [(f'leaf{leaf}',RandomForestClassifier(n_estimators=160,min_samples_leaf=leaf,
                  max_features=frac,class_weight=weight,random_state=seed,n_jobs=2,max_samples=.8))
                for leaf,frac,weight in [(1,'sqrt','balanced_subsample'),(12,.5,None),(50,.5,None)]]
    if name=='XGBoost':
        return [(f'd{d}w{int(b)}',XGBClassifier(n_estimators=n,max_depth=d,min_child_weight=m,
                  learning_rate=eta,scale_pos_weight=b,reg_lambda=reg,subsample=.9,random_state=seed,n_jobs=2))
                for n,d,m,eta,b,reg in [(200,2,6,.04,1,8),(240,3,8,.035,1,8),
                                          (150,4,6,.06,5,4),(100,6,1,.3,1/prevalence-1,1)]]
    if name=='HGB':
        return [(f'n{n}l{l}',HistGradientBoostingClassifier(max_iter=n,max_leaf_nodes=l,
                  min_samples_leaf=40,l2_regularization=1,learning_rate=.05,early_stopping=False,random_state=seed))
                for n,l in [(60,7),(140,15),(260,15)]]
    raise ValueError(name)

def run(args):
    s=load(args.data)
    views=clinical_views(s)
    y=s.target['y']
    for seed in SEEDS:
        out=args.root/args.name/str(seed)
        with start(out,args.name,seed,['trees.py'],s.schema) as log,threadpool_limits(limits=1):
            # Compact versus combined retains a genuinely simpler predictor even when native noise dimensions are many.
            for view,mat in views.items():
                if view=='native' and args.name!='LR':
                    continue
                for key,model in models(args.name,seed,y.mean()):
                    clock=time.monotonic()
                    model.fit(mat[s.fit],y)
                    if args.name=='LR' and model.n_iter_.max()>=model.max_iter:
                        raise RuntimeError('Logistic optimization did not converge')
                    setting=f'{view}_{key}'
                    joblib.dump(model,out/(setting+'.joblib'),compress=3)
                    save_pred(out,setting,model.predict_proba(mat[s.dev])[:,1])
                    say(log,f'{args.name} seed={seed} view={view} settings={model.get_params()} elapsed={time.monotonic()-clock:.1f}')
            finish(out)

if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('name',choices=['LR','RF','HGB','XGBoost'])
    q.add_argument('--data',type=Path,default=Path('private_data'))
    q.add_argument('--root',type=Path,default=Path('runs'))
    run(q.parse_args())
