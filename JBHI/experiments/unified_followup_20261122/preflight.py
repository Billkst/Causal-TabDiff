"""Development-only checks before the first training job."""
import numpy as np
import torch
from train import load, indices, inputs, predict, history_loss
from model import Unified
from registry_data import ROOT, save_json


def check():
    d=load()
    fit,cal,ev=indices(d,0)
    b=torch.cat([fit[d["mask"][fit]&d["dom"][fit].gt(0)][:20],fit[d["mask"][fit]&d["dom"][fit].eq(0)][:20],
                 fit[~d["mask"][fit]][:20]])
    assert len(b)==60 and not d["mask"][d["lm"]==2].any()
    assert torch.all(d["x"][torch.arange(3)>d["lm"][:,None]]==0)
    report={}
    for arm in "ABCD":
        torch.manual_seed(42)
        net=Unified(arm)
        net.eval()
        x,a,length=inputs(d,b)
        ref=net.core(x,a,history_length=length)["risk_2year_logit"].flatten()
        torch.testing.assert_close(net.risk(x,a,length),ref,rtol=0,atol=0)
        r=net.risk(x,a,length).detach().clone()
        shadow=dict(d,aux=d["aux"].flip(0),dom=d["dom"].flip(0),annual=d["annual"].flip(0),y=d["y"].flip(0))
        np.testing.assert_array_equal(predict(net,d,b),predict(net,shadow,b))
        grad={}
        if arm!="A":
            loss=net.auxiliary_loss(net.encode(x,a,length),d["aux"][b],d["dom"][b],d["mask"][b])
            alt=d["aux"][b].clone();alt[~d["mask"][b]]=100.
            torch.testing.assert_close(loss,net.auxiliary_loss(net.encode(x,a,length),alt,d["dom"][b],d["mask"][b]))
            loss.backward()
            for name,module in [("time",net.core.base_model.block1.time_attn),("feature",net.core.base_model.block1.feat_attn),
                                ("condition",net.core.base_model.cond_mlp)]:
                grad[name]=float(sum(p.grad.detach().square().sum() for p in module.parameters()).sqrt())
                assert grad[name]>0 and np.isfinite(grad[name])
            assert all(p.grad is None for p in net.core.risk_head.parameters())
            net.zero_grad(set_to_none=True)
            zero=net.auxiliary_loss(net.encode(x,a,length),alt,d["dom"][b],torch.zeros_like(d["mask"][b]))
            zero.backward()
            assert zero.item()==0 and all(p.grad is None or not p.grad.any() for p in net.parameters())
            _,sizes=net.report(net.encode(x,a,length),d["dom"][b])
            if arm in "CD":
                torch.testing.assert_close(sizes[:,0],sizes[:,1:7].amax(1),rtol=0,atol=0)
                assert (sizes[:,-1]<=sizes[:,0]).all()
        opt=torch.optim.Adam(net.parameters(),lr=.0001)
        for iteration in range(2):
            opt.zero_grad()
            h=net.encode(x,a,length)
            loss=net.core.risk_head(h).square().mean()
            if arm!="D":loss=loss+history_loss(net,d,b)
            if arm!="A":loss=loss+net.auxiliary_loss(h,d["aux"][b],d["dom"][b],d["mask"][b])
            loss.backward()
            if arm=="D":assert all(p.grad is None for p in net.core.base_model.noise_head.parameters())
            opt.step()
        assert not torch.equal(net.risk(x,a,length).detach(),r)
        report[arm]=dict(gradients=grad,mini_updates=2,risk_matches_original=True)
    save_json(ROOT/"preflight.json",report)
    print(report,flush=True)


if __name__=="__main__":check()
