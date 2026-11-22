"""Run each independent fold/seed in its own worker. Re-entry skips finished arms."""
import argparse
import multiprocessing as mp
import os

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
from train import ROOT,CFG,load,fit_one


def run(job):
    f,s=job
    for arm in CFG["arms"]:fit_one(bank,f,s,arm)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=6)
    args=p.parse_args()
    if not (ROOT/"preflight.json").is_file():raise RuntimeError("Run preflight.py first")
    bank=load()
    with mp.get_context("fork").Pool(args.workers) as pool:
        list(pool.imap_unordered(run,[(f,s) for f in CFG["folds"] for s in CFG["seeds"]]))
