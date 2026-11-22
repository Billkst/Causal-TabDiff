"""Check the staged orphan review tree against the aggregate-only release allowlist."""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import subprocess

PREFIX='JBHI/experiments/unified_followup_audit_20261122/'
FILES=('.gitignore','benchmark.py','cases.py','check_review.py','common.py','frozen_inputs.py','report.py',
       'reproduce_audit.sh','sensitivity.py','verify_audit.py','论文写作补齐审计报告.md',
       'statistics/case_selection.json','statistics/complexity_benchmark.csv',
       'statistics/param_sensitivity_metrics.csv','statistics/param_sensitivity_paired.json',
       'statistics/tau_sweep.csv','statistics/verification.json')
RISK='ap auc f1 mcc ppv1 coverage1 enrichment1 ppv5 coverage5 enrichment5'
AUX=('eligible positive conditional_zero_n conditional_positive_n zero_specificity nonzero_sensitivity whole_mae '
     'site_mae growth_mae positive_whole_mae presence_brier presence_ap predicted_positive max_violation '
     'dominant_violation growth_violation forced_max_violation')
SCHEMAS={'tau_sweep.csv':('kind fold seed tau '+AUX+' notice',60),
         'param_sensitivity_metrics.csv':('arm fold seed aux_weight gamma selected_epoch '+RISK+' '+AUX+' notice',63),
         'complexity_benchmark.csv':('model registered_parameters training_network_parameters training_forward_network_parameters '
             'loss_parameters deployment_parameters replay_batch_max_abs fit_seconds_median fine_seconds_median timing_source '
             'training_graph_parameters batch_size threads warmups repeats q25_ms median_ms q75_ms iqr_ms notice',11)}
NOTICE='冻结后稳健性审计，不改变已锁定主模型 R'


def verify(repo):
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args])
    staged=git('ls-files','-z').decode().split('\0')[:-1]
    assert set(staged)=={PREFIX+f for f in FILES}
    assert all('/private/' not in p and '/logs/' not in p and Path(p).suffix not in ('.pt','.npz') for p in staged)
    for p in staged:
        raw=git('show',':'+p)
        assert b'\x00' not in raw
        t=raw.decode('utf8')
        name=Path(p).name
        if name in SCHEMAS:
            first,n=SCHEMAS[name]
            rows=list(csv.DictReader(io.StringIO(t)))
            assert len(rows)==n and set(rows[0])==set(first.split())
            assert all(r['notice']==NOTICE for r in rows)
        elif name.endswith('.json'):
            o=json.loads(t)
            assert o['notice']==NOTICE
            if name=='case_selection.json':
                assert set(o)=={'notice','counts','distinct_cases','dev0_seed42'}
                assert all(type(x)==int for x in [*o['counts'].values(),*o['dev0_seed42'].values()])
            elif name=='param_sensitivity_paired.json':
                assert set(o)=={'notice','draws','seed','contrasts','absolute_ap'} and len(o['contrasts'])==6
                assert all(set(r)=={'arm','delta','lo','hi'} for r in o['contrasts'])
                assert all(set(r)=={'mean','lo','hi'} for r in o['absolute_ap'].values())
            elif name=='verification.json':
                assert all(type(v) in (str,int,float,bool) for v in o.values())
        if p.endswith('.md') or '/statistics/' in p:
            assert not re.search(r'(?i)(?:\bPID\b|patient_id|patient_uid|case_display|真实日期)\s*[:=,|]\s*\d',t)
            assert not re.search(r'(?m)^\| (?:T[012]|基线条件|两年真实结局) \|',t)
            assert not re.search(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',t)
    print(f'git ls-files verified: {len(staged)} allowlisted text files, 3 aggregate CSVs, 3 aggregate JSONs',flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('repository',type=Path)
    verify(cli.parse_args().repository)
