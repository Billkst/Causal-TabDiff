"""Time a single CPU batch of 1,000 already-encoded landmark inputs."""
import argparse
import platform
import subprocess
import time
from common import *
import pandas as pd
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from model import Unified,focal_loss_with_logits
from registry_data import ORIGINAL
sys.path.insert(0,str(ORIGINAL/'external/TSLib'))
sys.path.insert(0,str(EXP/'conditional_trajectories'))
from backbones import build

MODELS=(*'ABCDRN',*DEEP)


class ParameterReads(TorchDispatchMode):
    """Parameters actually read by the deployment route, including internal guided steps."""
    def __init__(self,model):
        self.names={id(p):n for n,p in model.named_parameters()}
        self.used=set()

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        for item in tree_leaves((args,kwargs)):
            if id(item) in self.names:
                self.used.add(self.names[id(item)])
        return func(*args,**(kwargs or {}))


def train_parameters(model,loss):
    named=list(model.named_parameters())
    grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
    return {n for (n,_),g in zip(named,grads) if g is not None}


def common_route(name):
    d,_=engine.read_role(0,2)
    path=source(0,42,name) if name in 'RN' else OLD/'runs/dev0/42'/name
    net=load_net(0,42,name) if name in 'RN' else Unified(name)
    net.load_state_dict(engine.state(path/'best.pt'))
    net.train()
    # Only model architecture is probed by this backward pass. No optimizer update.
    fit,_=engine.read_role(0,0)
    b=torch.cat([torch.where(fit['mask'] & (fit['dom']>0))[0][:6],torch.arange(10)])
    x=inputs(fit,b)
    train_read=ParameterReads(net)
    with train_read:
        h=net.encode(*x)
        loss=focal_loss_with_logits(net.core.risk_head(h),fit['y'][b,None].float(),2.,None)
        loss+=net.core.compute_trajectory_loss(net.core.trajectory_head(h),fit['annual'][b],fit['active'][b])
        if name!='A':
            loss+=net.auxiliary_loss(h,fit['aux'][b],fit['dom'][b],fit['mask'][b])
        if name in 'ABCR':
            den,cond=net.history_terms(*x) if name=='R' else net.core.base_model(x[0],x[1])
            loss+=den+.5*cond
    training=train_parameters(net,loss)
    x=inputs(d,torch.arange(1000))
    net.eval()
    def forward(i):
        return net.risk(*(a[:i] for a in x)).sigmoid()
    return net,training,train_read.used,forward,path


def baseline_route(name):
    with np.load(OLD/'private/registry.npz') as bank:
        x,a,t,fold,lead=(bank[k] for k in ('x','a','lm','folds','lead'))
    fit=np.flatnonzero(fold[0]==0)
    ev=np.flatnonzero(fold[0]==2)[:1000]
    path=OLD/'baselines/dev0/42'/name
    with np.load(path/'scaling.npz') as scaler:
        valid=np.arange(3)<=t[:,None]
        z=((x-scaler['mean'])/scaler['scale']).astype('float32')*valid[:,:,None]
    z=np.concatenate([z,valid[:,:,None],(a[:,None]*valid)[:,:,None]],axis=2).astype('float32')
    event=np.where(np.isin(lead[fit],[1,2]),lead[fit],0)
    torch.manual_seed(42)
    np.random.seed(42)
    net=build(name,z[fit],event,EXP/'marked_progression/assets/upstream','cpu')
    ckpt=engine.state(path/'best.pt')
    net.load_state_dict(ckpt,assign=name=='SSSD')
    subset=np.r_[np.flatnonzero(event)[:4],np.flatnonzero(event==0)[:12]]
    batch=torch.from_numpy(z[fit[subset]])
    ee=torch.from_numpy(event[subset])
    disc=name in ('iTransformer','TimeXer')
    train_read=ParameterReads(net)
    with train_read:
        loss=F.binary_cross_entropy_with_logits(net(batch),(ee>0).float()) if disc else net.loss(batch,ee)
    training=train_parameters(net,loss)
    net.load_state_dict(ckpt,assign=name=='SSSD')
    net.eval()
    test=torch.from_numpy(z[ev])
    def forward(i):
        return net(test[:i]).sigmoid() if disc else net.risk(test[:i])
    return net,training,train_read.used,forward,path


def budget(name):
    fine,full=[],[]
    for f in range(3):
        for s in SEEDS:
            k=group(f,s)
            if name in 'RN':
                budgets=pd.read_csv(FIX/'statistics/budgets.csv')
                selected=budgets[(budgets.stage=='validation') & (budgets.model==name)&(budgets.fold==f)&(budgets.seed==s)]
                sec=selected.seconds.item()
                pre=(pd.read_csv(FIX/'private/pretraining'/k/'pretrain.csv').seconds.iloc[-1] if name=='R' else 0)
            elif name in 'ABCD':
                sec=obj(OLD/'runs'/k/name/'complete.json')['seconds_fine']
                pre=float((OLD/'runs'/k/'pretrain.log').read_text().splitlines()[-1].split()[-1]) if name!='D' else 0
            else:
                sec=obj(OLD/'baselines'/k/name/'complete.json')['seconds']
                pre=0
            fine.append(sec)
            full.append(sec+pre)
    src='repair/statistics/budgets.csv + pretrain.csv' if name in 'RN' else ('runs/complete.json + pretrain.log' if name in 'ABCD' else 'baselines/complete.json')
    return dict(fit_seconds_median=float(np.median(full)),fine_seconds_median=float(np.median(fine)),timing_source=src)


def bench(name,probe):
    net,training,train_read,forward,path=baseline_route(name) if name in DEEP else common_route(name)
    params=dict(net.named_parameters())
    with torch.no_grad():
        read=ParameterReads(net)
        with read:
            forward(8)
        with np.load(path/'predictions.npz') as scores:
            stored=scores['p'][:1000]
        result=forward(1000).numpy()
        err=float(np.max(abs(stored-result)))
        assert result.shape==(1000,) and np.isfinite(result).all()
        np.testing.assert_allclose(result,stored,atol=2e-5,rtol=2e-5)
        training_n=sum(params[k].numel() for k in training)
        deploy_n=sum(params[k].numel() for k in read.used)
        assert 0<deploy_n<=sum(p.numel() for p in params.values())
        if name in 'ABCDRN':
            assert deploy_n==5133,(name,deploy_n,read.used)
        record=dict(model=name,registered_parameters=sum(p.numel() for p in params.values()),
                    training_network_parameters=training_n,training_forward_network_parameters=sum(params[k].numel() for k in train_read),loss_parameters=(2 if name in 'DN' else 3) if name in 'ABCDRN' else 0,
                    deployment_parameters=deploy_n,replay_batch_max_abs=err,**budget(name))
        record['training_graph_parameters']=training_n+record['loss_parameters']
        save_json(PRIVATE/f'graph-{name}.json',dict(train=sorted(training),training_forward=sorted(train_read),deploy=sorted(read.used)))
        if probe:
            print('graph',name,training_n,deploy_n,err,flush=True)
            return
        for _ in range(5):
            forward(1000)
        times=[]
        for _ in range(20):
            start=time.perf_counter_ns()
            got=forward(1000)
            times.append((time.perf_counter_ns()-start)/1e6)
            np.testing.assert_array_equal(got.numpy(),result)
    q=np.quantile(times,[.25,.5,.75])
    save_json(PRIVATE/f'timing-{name}.json',dict(times_ms=times))
    print('benchmark',name,*q,flush=True)
    return dict(**record,batch_size=1000,threads=1,warmups=5,repeats=20,q25_ms=q[0],median_ms=q[1],q75_ms=q[2],iqr_ms=q[2]-q[0],notice=NOTICE)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    affinity=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,[affinity[0]])
    env=dict(cpu=platform.processor(),lscpu=subprocess.check_output(['lscpu','-J'],text=True),
             torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
             threads=1,interop_threads=1,pinned_logical_cpu=affinity[0])
    save_json(PRIVATE/'benchmark_env.json',env)
    rows=[bench(n,args.probe) for n in args.models]
    if not args.probe:
        assert tuple(args.models)==MODELS
        pd.DataFrame(rows).to_csv(STATS/'complexity_benchmark.csv',index=False)
