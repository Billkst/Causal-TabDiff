"""Fingerprint allowlisted local development inputs without reading any source cohort."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
EXP = PROJECT / 'JBHI/experiments'
OLD = EXP / 'unified_followup_20261122'
FIX = EXP / 'unified_followup_repair_20261122'
SRC = PROJECT / 'server_repos/20260724/source/Causal-TabDiff'
DEEP = ('iTransformer', 'TimeXer', 'SSSD', 'TSDiff', 'SurvTraj')


def selection():
    paths = []
    def add(base, names):
        paths.extend(base / name for name in names.split())
    add(FIX, 'repair.py engine.py protocol.json freeze_record.json statistics/measurement.csv statistics/metrics.csv statistics/budgets.csv')
    add(OLD, 'model.py train.py registry_data.py protocol.json private/registry.npz population.json')
    add(EXP, 'clinical_paths/private_benchmark_math.py conditional_trajectories/backbones.py')
    add(SRC, 'run_phase2_twostage.py src/data/data_module_landmark.py src/data/data_module_landmark_cached.py src/evaluation/efficiency.py src/evaluation/metrics.py src/models/causal_tabdiff.py src/models/causal_tabdiff_trajectory.py src/utils/jbhi_protocol.py src/utils/training_logger.py')
    add(SRC/'external/TSLib', 'layers/Embed.py layers/SelfAttention_Family.py layers/Transformer_EncDec.py layers/__init__.py models/TimeXer.py models/iTransformer.py models/__init__.py utils/__init__.py utils/masking.py')
    for d in ('sssd/imputers','sssd/sssd_utils','tsdiff/architecture','survtraj/surv_traj'):
        paths.extend(sorted((EXP/'marked_progression/assets/upstream'/d).glob('*.py')))
    for f in range(3):
        for s in (42, 52, 62):
            key = Path(f'dev{f}') / str(s)
            add(FIX/'private/pretraining'/key, 'pretrained.pt pretrain.csv')
            for arm in 'RN':
                add(FIX/'private/validation'/key/arm, 'best.pt fit.json predictions.npz calibration.npz')
            add(OLD/'runs'/key, 'pretrain.log')
            for arm in 'ABCD':
                add(OLD/'runs'/key/arm, 'complete.json')
                if (f,s)==(0,42) or arm in 'BC':
                    add(OLD/'runs'/key/arm, 'best.pt predictions.npz')
            for name in DEEP:
                add(OLD/'baselines'/key/name, 'complete.json')
                if (f,s)==(0,42):
                    add(OLD/'baselines'/key/name, 'best.pt predictions.npz scaling.npz')
    return paths


def verify():
    manifest={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selection()}
    dest=HERE/'private/inputs.sha256.json'
    if dest.exists():
        assert manifest==json.loads(dest.read_text()),'Frozen input digest mismatch'
    else:
        dest.write_text(json.dumps(manifest,indent=2))
    print('Verified local frozen inputs:',len(manifest),flush=True)


if __name__=='__main__':
    verify()
