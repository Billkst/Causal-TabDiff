"""Reconstruct the frozen exclusions, then load only the eligible development rows."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / '_audit_sources/20260724/Causal-TabDiff/data'
TABLE_SHA = 'b832240ffa207c3e8d1d40919a5ccca27ebb6e4abd969c299bce836f7e410847'
EXCLUDED_SHA = '03ec275dc0d2fdcfcb23065d8844237cada1432b4f93b1da25973edf1effc04d'
FULL_SEEDS = [0, 1, 2, 3, 42, 52, 62, 72, 82, 123, 456, 789, 1024]
DEBUG_SEEDS = [42, 52, 62, 72, 82]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(ids):
    return hashlib.sha256(np.sort(ids).astype('<i8').tobytes()).hexdigest()


def split(frame, seed):
    ids = frame.pid.unique()
    label = frame.groupby('pid').y_2year.max().reindex(ids)
    return train_test_split(ids, test_size=.2, random_state=seed, stratify=label)


def eligible():
    path = DATA / 'landmark_tables/unified_person_landmark_table.pkl'
    if sha(path) != TABLE_SHA:
        raise ValueError('Unexpected source landmark table')
    frame = pd.read_pickle(path)
    debug = pd.read_pickle(DATA / 'landmark_tables_debug_backup_1000persons/unified_person_landmark_table.pkl')
    frozen = [split(frame, s)[1] for s in FULL_SEEDS] + [split(debug, s)[1] for s in DEBUG_SEEDS]
    known = np.unique(np.concatenate(frozen))
    # Only identities are compared to the archived union digest. No prediction or
    # performance measure participates in recovery of the eight residual IDs.
    residual = np.setdiff1d(debug.pid.unique(), known)
    if (len(known), len(residual)) != (50131, 14):
        raise ValueError('Historical split reconstruction changed')
    for extra in itertools.combinations(residual, 8):
        excluded = np.sort(np.r_[known, np.asarray(extra)])
        if digest(excluded) == EXCLUDED_SHA:
            break
    else:
        raise ValueError('Historical exclusions do not match archive')
    non_test, _ = split(frame, 42)
    label = frame.groupby('pid').y_2year.max().reindex(non_test)
    train, _ = train_test_split(non_test, test_size=.25, stratify=label, random_state=42)
    expected = json.loads((HERE.parent/'repair_validation/verified.json').read_text())
    if digest(train) != expected['manifest']['split_pid_sha'][0]:
        raise ValueError('July training set differs from original')
    frame = frame.loc[frame.pid.isin(np.setdiff1d(train, excluded))].sort_values(['pid','landmark'])
    if (frame.pid.nunique(), len(frame), frame.loc[frame.y_2year == 1,'pid'].nunique(), frame.y_2year.sum()) != (2118,6319,41,61):
        raise ValueError('Strict development cohort differs from frozen counts')
    return frame.reset_index(drop=True)


def tables(ids):
    raw = DATA/'nlst.780.idc.delivery.052821'
    checks = json.loads((HERE.parent/'final_validation/availability.json').read_text())['raw_sha']
    def read(key, fields):
        path = raw/f'nlst_780_{key}_idc_20210527.csv'
        if sha(path) != checks[path.name]:
            raise ValueError(f'Unexpected source for {key}')
        table = pd.read_csv(path, usecols=['pid', *fields], low_memory=False)
        return table.loc[table.pid.isin(ids)].copy()
    p = read('prsn', ['age','gender','race','cigsmok',*[f'{k}{t}' for t in range(3)
                       for k in ('scr_iso','scr_res','scr_days')]]).set_index('pid')
    keys = ['pid','study_yr','sct_ab_num']
    ct = read('ctab', [*keys[1:],'sct_ab_desc','sct_epi_loc','sct_long_dia','sct_perp_dia',
                        'sct_margins','sct_pre_att'])
    change = read('ctabc', [*keys[1:],'sct_ab_gwth','sct_ab_attn','sct_ab_preexist'])
    nod = ct.loc[ct.sct_ab_desc==51].merge(change, on=keys, how='left', validate='one_to_one')
    return p, ct, nod
