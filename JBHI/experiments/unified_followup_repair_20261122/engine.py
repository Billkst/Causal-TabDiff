"""Calibration-only training and aggregate gradient instrumentation."""
import csv
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from repair import ARCHIVE, ROOT, Repaired, Variant
from train import CFG, cutoff, inputs, predict, random_phase
from model import UncertaintyWeightedLoss, focal_loss_with_logits
from registry_data import save_json

os.umask(0o077)
torch.set_num_threads(1)
FIELDS = ("x", "a", "lm", "y", "annual", "active", "dom", "aux", "mask")
SCREEN = dict(C0=Variant(), P0=Variant(prefix=True), sample_only=Variant(weighting="sample_only"),
              loss_only=Variant(weighting="loss_only"), P2=Variant(sparse=True), D0=Variant(diffusion=False))


def read_role(fold, role):
    with np.load(ARCHIVE/"private/registry.npz") as data:
        idx = np.flatnonzero(data["folds"][fold] == role)
        return {k: torch.from_numpy(data[k][idx]) for k in FIELDS}, idx


def take_order(labels, ratio, scheme, generator):
    pos = 5*ratio if scheme != "loss_only" else torch.ones_like(ratio)
    weights = torch.where(labels > 0, pos, 1.).double()
    return torch.multinomial(weights, len(labels), replacement=True, generator=generator)


def state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def pretrain(d, fold, seed, dest):
    output = dest/"pretrained.pt"
    if output.exists():
        return output
    torch.manual_seed(seed)
    net = Repaired(Variant(prefix=True))
    opt = torch.optim.Adam(net.core.base_model.parameters(), lr=.001, weight_decay=.0001)
    gen = torch.Generator().manual_seed(seed)
    start = time.monotonic()
    with (dest/"pretrain.csv").open("w", buffering=1) as stream:
        wr = csv.writer(stream)
        wr.writerow(["epoch", "denoise", "smoking", "seconds"])
        for ep in range(1, CFG["pretrain_epochs"]+1):
            totals = np.zeros(2)
            order = torch.randperm(len(d["y"]), generator=gen)
            for j, b in enumerate(order.split(CFG["batch"])):
                torch.manual_seed(random_phase(seed, ep, j))
                den, cond = net.history_terms(*inputs(d, b))
                opt.zero_grad()
                (den+.5*cond).backward()
                opt.step()
                totals += np.array([den.item(), cond.item()])*len(b)
            assert np.isfinite(totals).all()
            wr.writerow([ep, *(totals/len(order)), time.monotonic()-start])
    torch.save(net.core.state_dict(), output.with_suffix(".partial"))
    output.with_suffix(".partial").replace(output)
    return output


def grad_record(terms, net):
    base = net.core.base_model
    params = list(base.cond_mlp.parameters()) + list(base.block1.parameters())
    vectors = {}
    row = {}
    for key, val in terms.items():
        vec = torch.autograd.grad(val, params, retain_graph=True, allow_unused=True)
        vectors[key] = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                                  for g,p in zip(vec, params)])
        row["norm_"+key] = vectors[key].norm().item()
    for i, a in enumerate(vectors):
        for b in list(vectors)[i+1:]:
            row["cos_"+a+"_"+b] = F.cosine_similarity(vectors[a], vectors[b], dim=0).item()
    return row


@torch.no_grad()
def measurement(net, d):
    net.eval()
    active = torch.where(d["mask"])[0]
    sizes, labels, given = [], [], []
    for b in active.split(512):
        h = net.encode(*inputs(d,b))
        logits, vals = net.report(h)
        _, teacher = net.report(h, d["dom"][b])
        sizes.append(vals.expm1())
        labels.append(logits.softmax(1))
        given.append(teacher.expm1())
    v, q, forced = map(torch.cat,(sizes,labels,given))
    y, dom = d["aux"][active].expm1(), d["dom"][active]
    pos = dom > 0
    nondom = torch.cat((~F.one_hot((dom-1).clamp_min(0),6).bool(),torch.ones(len(dom),1,dtype=torch.bool)),1)
    include = pos[:,None] & nondom
    observed = y[:,1:][include] > 0
    decoded = forced[:,1:][include] > 0
    chosen = q.argmax(1)
    site = (chosen-1).clamp_min(0)
    err = (v-y).abs()
    stats = dict(eligible=len(dom), positive=int(pos.sum()), conditional_zero_n=int((~observed).sum()),
                 conditional_positive_n=int(observed.sum()),
                 zero_specificity=float((~decoded[~observed]).float().mean()),
                 nonzero_sensitivity=float(decoded[observed].float().mean()),
                 whole_mae=err[:,0].mean().item(), site_mae=err[:,1:7].mean().item(),
                 growth_mae=err[:,7].mean().item(), positive_whole_mae=err[pos,0].mean().item(),
                 presence_brier=((1-q[:,0]-pos.float())**2).mean().item(),
                 presence_ap=float(average_precision_score(pos,1-q[:,0])),
                 predicted_positive=int((chosen>0).sum()),
                 max_violation=int(((v[:,1:7].amax(1)-v[:,0]).abs()>1e-5).sum()),
                 dominant_violation=int(((v[torch.arange(len(v)),site+1]-v[:,0]).abs()>1e-5).sum()),
                 growth_violation=int((v[:,7]>v[:,0]+1e-5).sum()),
                 forced_max_violation=int(((forced[:,1:7].amax(1)-forced[:,0]).abs()>1e-5).sum()))
    return stats, dict(sizes_mm=v.numpy(), classes=q.numpy(), forced_mm=forced.numpy())


def train_model(fold, seed, variant, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/"fit.json").is_file():
        assert json.loads((dest/"fit.json").read_text())["variant"] == asdict(variant)
        return
    d, _ = read_role(fold, 0)
    cal, cal_rows = read_role(fold, 1)
    cal_indices = torch.arange(len(cal_rows))
    torch.manual_seed(seed)
    net = Repaired(variant)
    if variant.diffusion:
        if variant.prefix:
            common = ROOT/"private"/"pretraining"/f"dev{fold}"/str(seed)
            common.mkdir(parents=True, exist_ok=True)
            weight = pretrain(d, fold, seed, common)
        else:
            weight = ARCHIVE/"runs"/f"dev{fold}"/str(seed)/"pretrained.pt"
        net.core.load_state_dict(state(weight))
    core = net.core
    uw = UncertaintyWeightedLoss(3)
    with torch.no_grad():
        uw.log_sigma[2] = -1
    groups = [dict(params=core.risk_head.parameters(),lr=.001),dict(params=core.trajectory_head.parameters(),lr=.0005),
              dict(params=core.base_model.parameters(),lr=.0001),dict(params=uw.parameters(),lr=.01),
              dict(params=net.auxiliary.parameters(),lr=.001)]
    if variant.sparse:
        groups.append(dict(params=net.presence.parameters(),lr=.001))
    opt = torch.optim.Adam(groups,weight_decay=.0001)
    labels = d["y"].float()
    ratio = (len(labels)-labels.sum())/labels.sum()
    pos_weight = None if variant.weighting == "sample_only" else ratio.reshape(1)
    gen = torch.Generator().manual_seed(seed+1)
    best = -1.
    start = time.monotonic()
    with (dest/"training.csv").open("w",buffering=1) as log, (dest/"gradients.csv").open("w",buffering=1) as glog:
        writer = gwriter = None
        for ep in range(1,CFG["finetune_epochs"]+1):
            net.train()
            order = take_order(labels,ratio,variant.weighting,gen)
            counts = dict(positive=int(labels[order].sum()),negative=int((1-labels[order]).sum()),
                          unique_positive=int(order[labels[order]>0].unique().numel()),unique_negative=int(order[labels[order]==0].unique().numel()),
                          aux=int(d["mask"][order].sum()),annual_valid=int(d["active"][order].sum()))
            totals = dict(loss=0., risk=0., annual=0., aux_loss=0., denoise=0., smoking=0., clip_norm=0.)
            for j,b in enumerate(order.split(CFG["batch"])):
                stream = random_phase(seed,ep,j)
                if variant.diffusion:
                    torch.manual_seed(stream+330001)
                    den, smoking = net.history_terms(*inputs(d,b))
                    hist = den+.5*smoking
                torch.manual_seed(stream)
                h = net.encode(*inputs(d,b))
                annual = core.compute_trajectory_loss(core.trajectory_head(h),d["annual"][b],d["active"][b])
                risk = focal_loss_with_logits(core.risk_head(h),d["y"][b,None].float(),2.,pos_weight)
                aux = net.auxiliary_loss(h,d["aux"][b],d["dom"][b],d["mask"][b])
                weights = torch.exp(-uw.log_sigma)
                if variant.diffusion:
                    loss = uw([hist,annual,risk])
                else:
                    loss = (weights[1:]*torch.stack([annual,risk])+uw.log_sigma[1:]).sum()
                loss = loss+CFG["aux_weight"]*aux
                if j == 0:
                    parts = dict(risk=weights[2]*risk,annual=weights[1]*annual,aux=CFG["aux_weight"]*aux)
                    if variant.diffusion:
                        parts.update(history=weights[0]*hist,denoise=weights[0]*den,smoking=weights[0]*.5*smoking)
                    grad = dict(epoch=ep, batch_n=len(b), batch_positive=int(labels[b].sum()),
                                weight_risk=weights[2].item(),weight_annual=weights[1].item(),weight_history=weights[0].item(),
                                **grad_record(parts,net))
                    if gwriter is None:
                        gwriter = csv.DictWriter(glog, list(grad))
                        gwriter.writeheader()
                    gwriter.writerow(grad)
                opt.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                for k,v in dict(loss=loss,risk=risk,annual=annual,aux_loss=aux,clip_norm=norm).items():
                    totals[k] += v.item()*len(b)
                if variant.diffusion:
                    totals["denoise"] += den.item()*len(b)
                    totals["smoking"] += smoking.item()*len(b)
            pc = predict(net,cal,cal_indices)
            ap = float(average_precision_score(cal["y"],pc))
            if ap > best:
                best, chosen = ap, ep
                torch.save(net.state_dict(),dest/"best.pt")
            if ep == 10:
                fixed10 = ap
                torch.save(net.state_dict(),dest/"epoch10.pt")
            row = dict(epoch=ep,cal_ap=ap,**counts,**{k:v/len(order) for k,v in totals.items()},seconds=time.monotonic()-start)
            if writer is None:
                writer = csv.DictWriter(log, list(row))
                writer.writeheader()
            writer.writerow(row)
            if ep-chosen >= CFG["patience"]:
                break
    net.load_state_dict(state(dest/"best.pt"))
    pc = predict(net,cal,cal_indices)
    aux_stats,_ = measurement(net,cal)
    np.savez_compressed(dest/"calibration.npz",rows=cal_rows,p=pc)
    save_json(dest/"fit.json",dict(fold=fold,seed=seed,variant=asdict(variant),epochs=ep,selected=chosen,
                                  seconds=time.monotonic()-start,cal_ap=best,cal_ap10=fixed10,ratio=ratio.item(),
                                  fit_n=len(labels),fit_positive=int(labels.sum()),cal_n=len(pc),
                                  threshold=cutoff(cal["y"],pc),auxiliary=aux_stats,
                                  parameters=sum(p.numel() for p in net.parameters())))
