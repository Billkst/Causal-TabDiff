"""Frozen threshold replay, single-factor audit fits and clustered paired inference."""
import argparse
import itertools
import multiprocessing as mp
import time
from common import *
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from model import UncertaintyWeightedLoss,focal_loss_with_logits


def read_archive(name):
    return pd.read_csv(FIX/'statistics'/name,float_precision='round_trip')


def preflight():
    from frozen_inputs import verify
    verify()
    with np.load(OLD/'private/registry.npz') as bank:
        assert len(np.unique(bank['who']))==10594 and len(bank['y'])==31587
        assert (bank['folds']==2).sum(axis=0).min()==(bank['folds']==2).sum(axis=0).max()==1
    archive=read_archive('measurement.csv').set_index(['model','fold','seed'])
    risk=read_archive('metrics.csv').set_index(['model','fold','seed'])
    rows=[]
    max_measure=max_risk=0.
    for f in range(3):
        d,ev=engine.read_role(f,2)
        cal,ca=engine.read_role(f,1)
        for s in SEEDS:
            net=load_net(f,s,scan=True)
            with np.load(source(f,s)/'predictions.npz') as z:
                np.testing.assert_array_equal(ev,z['rows'])
                p=z['p']
            with np.load(source(f,s)/'calibration.npz') as z:
                np.testing.assert_array_equal(ca,z['rows'])
                th=cutoff(cal['y'],z['p'])
            assert th==obj(source(f,s)/'fit.json')['threshold']
            scores=endpoints(d['y'].numpy().astype('int64'),p,th)
            for key,val in scores.items():
                max_risk=max(max_risk,abs(val-risk.loc[('R',f,s),key]))
            for tau in TAUS:
                net.tau=tau
                vals,_=engine.measurement(net,d)
                if tau==.5:
                    diff={k:abs(v-archive.loc[('R',f,s),k]) for k,v in vals.items()}
                    max_measure=max(max_measure,max(diff.values()))
                    print('anchor',f,s,{k:v for k,v in diff.items() if v},flush=True)
                rows.append(dict(kind='cell',fold=f,seed=s,tau=tau,**vals))
    out=pd.DataFrame(rows)
    means=out.drop(columns=['kind','fold','seed']).groupby('tau',as_index=False).mean()
    save_json(PRIVATE/'replay.json',dict(max_measure=max_measure,max_risk=max_risk))
    assert max_measure<=1e-12 and max_risk==0,(max_measure,max_risk)
    pd.concat([out,means.assign(kind='mean',fold='all',seed='all')]).assign(notice=NOTICE).to_csv(STATS/'tau_sweep.csv',index=False)
    print('Nine archive anchors verified',flush=True)


def train(job):
    f,s,name=job
    path=PRIVATE/'fits'/group(f,s)/name
    path.mkdir(exist_ok=True,parents=True)
    weight,gamma=ARMS[name]
    if (path/'evaluation.json').is_file():
        assert obj(path/'evaluation.json')['setting']==[weight,gamma]
        return
    d,_=engine.read_role(f,0)
    cal,ca=engine.read_role(f,1)
    torch.manual_seed(s)
    net=Repaired(spec())
    net.core.load_state_dict(engine.state(FIX/'private/pretraining'/group(f,s)/'pretrained.pt'))
    core=net.core
    uw=UncertaintyWeightedLoss(3)
    with torch.no_grad():
        uw.log_sigma[2]=-1
    opt=torch.optim.Adam([dict(params=m.parameters(),lr=lr) for m,lr in
        [(core.risk_head,.001),(core.trajectory_head,.0005),(core.base_model,.0001),(uw,.01),
         (net.auxiliary,.001),(net.presence,.001)]],weight_decay=.0001)
    labels=d['y'].float()
    ratio=(len(labels)-labels.sum())/labels.sum()
    gen=torch.Generator().manual_seed(s+1)
    best=-1.
    start=time.monotonic()
    with (LOGS/f'{name}-dev{f}-{s}.log').open('w',buffering=1) as stream:
        for ep in range(1,CFG['finetune_epochs']+1):
            ep_start=time.monotonic()
            net.train()
            total=0.
            order=engine.take_order(labels,ratio,'double',gen)
            for j,b in enumerate(order.split(CFG['batch'])):
                phase=engine.random_phase(s,ep,j)
                torch.manual_seed(phase+330001)
                den,smoking=net.history_terms(*inputs(d,b))
                torch.manual_seed(phase)
                h=net.encode(*inputs(d,b))
                annual=core.compute_trajectory_loss(core.trajectory_head(h),d['annual'][b],d['active'][b])
                risk=focal_loss_with_logits(core.risk_head(h),d['y'][b,None].float(),gamma,ratio.reshape(1))
                aux=net.auxiliary_loss(h,d['aux'][b],d['dom'][b],d['mask'][b])
                loss=uw([den+.5*smoking,annual,risk])+weight*aux
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                total+=loss.item()*len(b)
            pc=predict(net,cal,torch.arange(len(ca)))
            ap=float(average_precision_score(cal['y'],pc))
            if ap>best:
                chosen,best=ep,ap
                torch.save(net.state_dict(),path/'best.pt')
            line=(f'{name} dev{f} seed={s} epoch={ep} lr_risk=.001 loss={total/len(order):.8f} '
                  f'cal_ap={ap:.8f} best_ap={best:.8f} cal_auc={roc_auc_score(cal["y"],pc):.8f} '
                  f'epoch_seconds={time.monotonic()-ep_start:.3f}')
            print(line,flush=True)
            print(line,file=stream,flush=True)
            if ep-chosen>=CFG['patience']:
                break
    net.load_state_dict(engine.state(path/'best.pt'))
    threshold=cutoff(cal['y'],predict(net,cal,torch.arange(len(ca))))
    d,rows=engine.read_role(f,2)
    p=predict(net,d,torch.arange(len(rows)))
    aux,_=engine.measurement(net,d)
    np.savez_compressed(path/'predictions.npz',rows=rows,p=p)
    save_json(path/'evaluation.json',dict(setting=[weight,gamma],epoch=ep,selected_epoch=chosen,threshold=threshold,
              seconds=time.monotonic()-start,metrics=endpoints(d['y'].numpy().astype('int64'),p,threshold),measurement=aux))
    print('completed',f,s,name,flush=True)


def summarize():
    protocol=obj(FIX/'protocol.json')['validation']
    rng=np.random.default_rng(protocol['paired_seed'])
    reps=protocol['paired_draws']
    assert reps==2000
    draws={k:np.zeros(reps) for k in ARMS}
    risk=read_archive('metrics.csv').set_index(['model','fold','seed'])
    measurements=read_archive('measurement.csv').set_index(['model','fold','seed'])
    rows=[]
    with np.load(OLD/'private/registry.npz') as bank:
        folds,y,who=(bank[k] for k in ('folds','y','who'))
    for f in range(3):
        ev=np.flatnonzero(folds[f]==2)
        patient,inv=np.unique(who[ev],return_inverse=True)
        yy=y[ev].astype('int64')
        case=np.zeros(len(patient),dtype='int8')
        np.maximum.at(case,inv,yy)
        w=boot_people(case,reps,rng)
        for s,name in itertools.product(SEEDS,ARMS):
            if name=='R':
                path=source(f,s)
                scores=risk.loc[('R',f,s)].to_dict()
                aux=measurements.loc[('R',f,s)].to_dict()
                epoch=obj(path/'fit.json')['selected']
            else:
                path=PRIVATE/'fits'/group(f,s)/name
                record=obj(path/'evaluation.json')
                scores,aux,epoch=record['metrics'],record['measurement'],record['selected_epoch']
            with np.load(path/'predictions.npz') as pred:
                np.testing.assert_array_equal(pred['rows'],ev)
                p=pred['p']
            ap=WeightedAP(who[ev],yy,p)
            sample=np.concatenate([ap(w[:,i:i+128]) for i in range(0,reps,128)])
            for j in (0, 999, 1999):
                assert abs(sample[j]-average_precision_score(yy,p,sample_weight=w[inv,j]))<1e-12
            draws[name]+=sample/9
            rows.append(dict(arm=name,fold=f,seed=s,aux_weight=ARMS[name][0],gamma=ARMS[name][1],
                             selected_epoch=epoch,**scores,**aux,notice=NOTICE))
    df=pd.DataFrame(rows)
    assert len(df)==len(df.drop_duplicates(['arm','fold','seed']))==63 and not df.isna().any().any()
    df.to_csv(STATS/'param_sensitivity_metrics.csv',index=False)
    p=df.pivot(index=['fold','seed'],columns='arm',values='ap')
    pairs=[]
    for name in list(ARMS)[1:]:
        lo,hi=np.quantile(draws[name]-draws['R'],[.025,.975])
        pairs.append(dict(arm=name,delta=(p[name]-p.R).mean(),lo=lo,hi=hi))
    save_json(STATS/'param_sensitivity_paired.json',dict(notice=NOTICE,draws=reps,seed=protocol['paired_seed'],contrasts=pairs,
              absolute_ap={k:dict(mean=p[k].mean(),lo=np.quantile(d,.025),hi=np.quantile(d,.975)) for k,d in draws.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('step',choices=['preflight','train','summarize'])
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.step=='train':
        assert (STATS/'tau_sweep.csv').is_file()
        with mp.get_context('fork').Pool(args.workers) as pool:
            list(pool.imap_unordered(train,itertools.product(range(3),SEEDS,list(ARMS)[1:])))
    elif args.step=='preflight':
        preflight()
    else:
        summarize()
