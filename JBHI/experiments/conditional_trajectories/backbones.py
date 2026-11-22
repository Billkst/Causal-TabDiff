import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models import iTransformer, TimeXer


class Benchmark(nn.Module):
    def __init__(self,kind,channels):
        super().__init__()
        self.kind=kind
        cfg=SimpleNamespace(task_name='classification',seq_len=3,pred_len=1,embed='timeF',freq='h',dropout=.1,
            d_model=16,n_heads=2,e_layers=1,d_ff=48,activation='gelu',factor=3,output_attention=False,
            num_class=1,enc_in=channels,features='MS',patch_len=1,use_norm=False)
        self.model=iTransformer.Model(cfg) if kind=='iTransformer' else TimeXer.Model(cfg)
    def forward(self,x):
        if self.kind=='iTransformer':
            return self.model.classification(x,None).flatten()
        # Long diameter as the endogenous series; all other information is exogenous.
        ordered=torch.cat([x[:,:,:8],x[:,:,9:],x[:,:,8:9]],axis=-1)
        return self.model.forecast(ordered,None,None,None).flatten()

class GenerativeBenchmark(nn.Module):
    def __init__(self,root,kind,channels):
        super().__init__()
        self.kind=kind
        self.channels=channels+1
        sys.path.insert(0,str(root/kind.lower()))
        if kind=='SSSD':
            from imputers.SSSDS4Imputer import SSSDS4Imputer
            self.net=SSSDS4Imputer(self.channels,16,16,self.channels,2,16,32,32,4,16,0.,True,True)
        else:
            from architecture import BackboneModel
            self.net=BackboneModel(self.channels,24,self.channels,16,2,0,init_skip=False)
        self.register_buffer('alpha',torch.cos(torch.linspace(.02,1.35,32)).square())
        g=torch.Generator().manual_seed(9201)
        noise=torch.randn(2,4,self.channels,generator=g)
        self.register_buffer('initial',torch.cat([noise,-noise]))

    def layout(self,x):
        obs=F.pad(x,(0,1,0,1))
        mask=torch.ones_like(obs)
        mask[:,-1,-1]=0.
        return obs,mask

    def denoise(self,z,t,obs,mask):
        if self.kind=='TSDiff':
            return self.net(z,t)
        return self.net((z.transpose(1,2),obs.transpose(1,2),mask.transpose(1,2),t[:,None])).transpose(1,2)

    def loss(self,x,event):
        obs,mask=self.layout(x)
        clean=obs.clone()
        clean[:,-1,-1]=2*(event>0).float()-1
        t=torch.randint(len(self.alpha),(len(x),),device=x.device)
        a=self.alpha[t,None,None]
        eps=torch.randn_like(clean)
        z=a.sqrt()*clean+(1-a).sqrt()*eps
        if self.kind=='SSSD':
            z=z*(1-mask)+obs
        pred=self.denoise(z,t,obs,mask)
        errors=(pred-eps).square()
        if self.kind=='SSSD':
            return errors[:,-1,-1].mean()
        return .5*(errors.mean()+errors[:,-1,-1].mean())

    def risk(self,x):
        obs,mask=self.layout(x)
        output=[]
        times=[31,26,21,16,11,6,2,0]
        for template in self.initial:
            z=template.expand_as(obs)
            for step,next_step in zip(times,times[1:]+[-1]):
                a=self.alpha[step]
                prev=self.alpha[next_step] if next_step>=0 else torch.ones_like(a)
                t=torch.full((len(x),),step,device=x.device,dtype=torch.long)
                if self.kind=='SSSD':
                    z=obs+z*(1-mask)
                    noise=self.denoise(z,t,obs,mask)
                else:
                    # Observation self-guidance using the MSE score of the denoised clinical prefix.
                    with torch.enable_grad():
                        z=z.detach().requires_grad_(True)
                        pred=self.denoise(z,t,obs,mask)
                        clean=(z-(1-a).sqrt()*pred)/a.sqrt()
                        grad=torch.autograd.grad(((clean-obs).square()*mask).sum(),z)[0]
                    noise=pred.detach()+.05*(1-a).sqrt()*grad.detach()
                clean=(z.detach()-(1-a).sqrt()*noise)/a.sqrt()
                z=prev.sqrt()*clean+(1-prev).sqrt()*noise
            output.append(z[:,-1,-1])
        return torch.stack(output).mean(0).sigmoid()

class SurvivalBenchmark(nn.Module):
    def __init__(self,root,device,x,event):
        super().__init__()
        default=torch.get_default_dtype()
        sys.path.insert(0,str(root/'survtraj'))
        from surv_traj.model import SurvTraj
        torch.set_default_dtype(default)
        m=SurvTraj(dict(latent_dim=8,regular_coef=.01,sigma_z=1.),samples_num=8,
                   traj_penalty_points_n=5,device=device)
        record=np.empty(len(event),dtype=[('censor','?'),('time','<f8')])
        record['censor']=event>0
        record['time']=np.where(event==1,1,2)
        m._lazy_init(x.reshape(len(x),-1),record)
        self.net=m
        self.net.float()

    def loss(self,x,event):
        x=x.flatten(1)
        m=self.net
        t=torch.where(event==1,1.,2.)
        dt=(event>0).long()
        mu,sigma=m.vae.get_mu_sigma(x)
        sf,prob=m.beran(mu)
        generated=m.gen_evt_time(prob,m.T_bg)
        proto,z=m.calc_prototype(mu,sigma,generated,f_return_z_i=True)
        tx=(generated-m.T_mean)/m.T_std
        recon=m.vae.decoder(torch.cat([proto,tx[:,None]],1))
        vae,_,_=m.vae_loss(x,z,recon)
        case=x[dt.bool()]
        traj=m.calc_tr_llh(case,torch.ones(len(case),device=x.device),torch.full((len(case),),2.,device=x.device),5) if len(case) else x.new_zeros(())
        return vae-m.calc_surv_llh(sf,prob,m.T_bg,t,dt)-traj

    def risk(self,x):
        mu,_=self.net.vae.get_mu_sigma(x.flatten(1))
        sf,_=self.net.beran(mu)
        return 1-sf[:,-1]

def input_order(schema):
    order=np.arange(len(schema['features'])+1)
    i=schema['features'].index('global_size')
    order[i],order[8]=order[8],order[i]
    return order

def build(kind,x,event,upstream,device):
    if kind in ('iTransformer','TimeXer'):
        return Benchmark(kind,x.shape[-1])
    if kind=='SurvTraj':
        return SurvivalBenchmark(upstream,device,x,event)
    return GenerativeBenchmark(upstream,kind,x.shape[-1])

@torch.no_grad()
def predict(net,disc,dev):
    net.eval()
    rows=[(net(batch).sigmoid() if disc else net.risk(batch)).cpu().numpy()
          for batch in dev.split(128)]
    return np.concatenate(rows)
