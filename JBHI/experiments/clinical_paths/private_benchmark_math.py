"""Exact endpoint and patient bootstrap definitions copied from patient_diffusion/estimate.py for execution beside the data."""

import numpy as np

from scipy.sparse import coo_matrix

from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

def top_mass(scores, fraction):
    n = fraction*len(scores)
    cut = np.partition(scores, len(scores)-int(np.ceil(n)))[len(scores)-int(np.ceil(n))]
    high, tie = scores > cut, scores == cut
    mass = high.astype(float)
    mass[tie] = (n-high.sum())/tie.sum()
    return mass

def endpoints(y, p, cut):
    cls = p >= cut
    ans = dict(ap=float(average_precision_score(y,p)), auc=float(roc_auc_score(y,p)),
               f1=float(f1_score(y,cls,zero_division=0)), mcc=float(matthews_corrcoef(y,cls)))
    for frac, k in ((.01,"1"),(.05,"5")):
        w = top_mass(p,frac)
        cases = np.dot(y,w)
        ans.update({f"ppv{k}": cases/w.sum(), f"coverage{k}": cases/y.sum(),
                    f"enrichment{k}": cases/(w.sum()*y.mean())})
    return ans

class WeightedAP:
    """Group at every distinct positive threshold. Patient weights are shared across all landmarks."""
    def __init__(self, pid, y, p):
        _, code = np.unique(pid, return_inverse=True)
        levels = -np.unique(p[y == 1])[::-1]
        bins = np.searchsorted(levels, -p, side="left")
        active = bins < len(levels)
        def mapping(mask):
            shp = (len(levels), int(code.max())+1)
            return coo_matrix((np.ones(mask.sum()), (bins[mask],code[mask])), shape=shp).tocsr()
        self.all_above = mapping(active)
        self.positives = mapping(y == 1)

    def __call__(self, patient_weights):
        events = self.positives @ patient_weights
        numer = np.cumsum(events,axis=0)
        denom = np.cumsum(self.all_above @ patient_weights,axis=0)
        precision = np.divide(numer,denom,where=denom>0,out=np.zeros_like(numer))
        return np.sum(events*precision,axis=0)/numer[-1]

def boot_people(target, n, rng):
    weights = np.zeros((len(target),n), dtype=float)
    for label in (0,1):
        where = np.flatnonzero(target==label)
        for j in range(n):
            pick = rng.integers(len(where),size=len(where))
            weights[where,j] = np.bincount(pick,minlength=len(where))
    return weights
