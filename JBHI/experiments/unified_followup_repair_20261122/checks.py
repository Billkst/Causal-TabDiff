"""Deterministic support, objective and gradient checks. Uses only dev0 fitting rows."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from repair import ROOT, Variant, Repaired, observed, prefix_block
from engine import ARCHIVE, CFG, inputs, read_role, save_json, take_order
from model import focal_loss_with_logits


def check():
    torch.manual_seed(42)
    x = torch.randn(6,3,14)
    a = torch.rand(6,1)
    length = torch.tensor([1,2,3,1,2,3])
    valid = observed(x,length)
    dirty = torch.where(valid.unsqueeze(2),x,150*torch.randn_like(x))
    old, new = Repaired(Variant()).eval(), Repaired(Variant(prefix=True)).eval()
    new.load_state_dict(old.state_dict())
    def gap(a,b):
        return float((a-b).abs().max().detach())
    clean_logits = new.risk(x,a,length)
    delta = gap(clean_logits,new.risk(dirty,a,length))
    old_delta = gap(old.risk(x,a,length),old.risk(dirty,a,length))
    assert delta == 0 and old_delta > 1e-6
    h = new.encode(x,a,length)
    assert gap(h,new.encode(dirty,a,length)) == 0
    x.requires_grad_()
    clean = new.risk(x,a,length).sum()
    dx = torch.autograd.grad(clean,x)[0]
    assert dx[~valid].abs().max()==0 and dx[valid].norm()>0
    old_pad = torch.autograd.grad(old.risk(x,a,length).sum(),x)[0][~valid].norm().item()
    assert old_pad > 0
    step = torch.tensor([0,12,50,99,2,19])
    noise = torch.randn_like(x).requires_grad_()
    den,smoking = new.history_terms(x,a,length,step=step,noise=noise)
    dirty_noise = torch.where(valid.unsqueeze(2),noise,-1000.)
    other = new.history_terms(dirty,a,length,step=step,noise=dirty_noise)
    assert gap(den,other[0]) == gap(smoking,other[1]) == 0
    dx,de = torch.autograd.grad(den+.5*smoking,(x,noise))
    assert dx[~valid].norm()==de[~valid].norm()==0 and de[valid].norm()>0
    solo = [new.history_terms(x[i:i+1],a[i:i+1],length[i:i+1],step=step[i:i+1],noise=noise[i:i+1]) for i in range(len(x))]
    np.testing.assert_allclose([den.item(),smoking.item()],[torch.stack(v).mean().item() for v in zip(*solo)],rtol=2e-6)
    full = length==3
    torch.testing.assert_close(new.encode(x[full],a[full],length[full]),old.encode(x[full],a[full],length[full]),atol=1e-5,rtol=1e-5)
    torch.manual_seed(81)
    od = old.history_terms(x[full],a[full],length[full])
    torch.manual_seed(81)
    nd = new.history_terms(x[full],a[full],length[full])
    for l,r in zip(od,nd):
        torch.testing.assert_close(l,r,atol=1e-5,rtol=1e-5)
    # Synthetic positive report, only dominant site present. Mask out one
    # deliberately corrupt target to distinguish absent CT from a measured zero.
    sparse = Repaired(Variant(sparse=True))
    hidden = sparse.encode(x,a,length)
    cls = torch.tensor([1,2,3,0,0,0])
    truth = torch.zeros(6,8)
    truth[:3,0] = 1.
    truth[torch.arange(3),cls[:3]] = 1.
    mask = torch.tensor([True,True,True,True,False,False])
    dense = old.report(hidden,cls)[1]
    with torch.no_grad():
        sparse.presence.bias.fill_(-100)
    _,v = sparse.report(hidden,cls)
    assert (v[:3]>0).sum()==6 and (dense[:3]>0).sum()==24
    assert torch.equal(v[:,0],v[:,1:7].amax(1)) and (v[:,7]<=v[:,0]).all()
    loss = sparse.auxiliary_loss(hidden,truth,cls,mask)
    corrupt = truth.clone()
    corrupt[~mask] = float("nan")
    assert gap(loss,sparse.auxiliary_loss(hidden,corrupt,cls,mask))==0
    assert sparse.auxiliary_loss(hidden,corrupt,cls,~torch.ones_like(mask)).item()==0
    # Check BCE from new gates alone reaches shared parameters at a nonsaturated operating point.
    with torch.no_grad():
        sparse.presence.bias.zero_()
    logits = sparse.sparse_report(hidden)[2]
    bce = F.binary_cross_entropy_with_logits(logits[:3],(truth[:3,1:]>0).float())
    gate_grad = torch.autograd.grad(bce, sparse.core.base_model.block1.ffn[0].weight)[0].norm().item()
    assert gate_grad>0
    d,_ = read_role(0,0)
    y = d["y"].float()
    ratio = (len(y)-y.sum())/y.sum()
    p1 = {}
    for name in ("double","sample_only","loss_only"):
        gen = torch.Generator().manual_seed(43)
        rows = take_order(y,ratio,name,gen)
        logits = torch.zeros(2,1,requires_grad=True)
        focal = focal_loss_with_logits(logits,torch.tensor([[1.],[0.]]),2.,None if name=="sample_only" else ratio.reshape(1))
        g = torch.autograd.grad(focal,logits)[0].abs().flatten()
        p1[name] = dict(epoch_positive=int(y[rows].sum()), epoch_n=len(rows),first_batch_positive=int(y[rows[:CFG["batch"]]].sum()),
                        focal_grad_ratio_at_equal_logits=float(g[0]/g[1]))
    sel = d["mask"]&(d["dom"]>0)
    sizes = d["aux"][sel]
    data = dict(fit_n=len(y),fit_positive=int(y.sum()),eligible=int(d["mask"].sum()),positive_reports=int(sel.sum()),
                single_site=int(((sizes[:,1:7]>0).sum(1)==1).sum()),zero_growth=int((sizes[:,7]==0).sum()),
                padded_cell_fraction=float((~observed(d["x"],d["lm"]+1)).float().mean()))
    save_json(ROOT/"checks.json",dict(legacy_padding_risk_logit_gap=old_delta,legacy_risk_padded_grad=old_pad,
              repaired_padding_logit_gap=delta,repaired_padded_history_input_grad=0,repaired_padded_noise_grad=0,
              full_length_encode_gap=gap(new.encode(x[full],a[full],length[full]),old.encode(x[full],a[full],length[full])),
              sparse_gate_trunk_grad=gate_grad,dense_positive_cells=int((dense[:3]>0).sum()),
              sparse_positive_cells=int((v[:3]>0).sum()),pass_checks=True,fit=data,sampling=p1))
    print("Passed deterministic prefix, equal-landmark loss, sparse supervision and weighting checks.",flush=True)


if __name__ == "__main__":
    check()
