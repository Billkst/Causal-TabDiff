#!/usr/bin/env python -u
import json
import sys
from pathlib import Path

import numpy as np

OUTPUT_ROOT = Path('outputs/submission_v3_5seeds/param_discussion_layer2')

CONFIGS = {
    'focal_gamma': ['0.0', '1.0', '2.0', '3.0', '4.0'],
    'oversample_ratio': ['1.0', '2.5', '7.5', '10.0'],
    'disc_weight': ['0.0', '0.01', '0.05', '0.1', '0.2'],
}
SEEDS = [42, 0, 1, 2, 3]

REQUIRED_NPZ_KEYS = ['test_y_true', 'test_y_pred', 'val_y_true', 'val_y_pred', 'seed']
REQUIRED_EFF_KEYS = ['total_training_wall_clock_sec']


def verify():
    total_expected = sum(len(v) for v in CONFIGS.values()) * len(SEEDS)
    npz_ok = 0
    eff_ok = 0
    errors = []

    for param, values in CONFIGS.items():
        for val in values:
            for seed in SEEDS:
                tag = f'{param}_{val}_seed{seed}'
                npz_path = OUTPUT_ROOT / param / f'{tag}.npz'
                eff_path = OUTPUT_ROOT / param / f'{tag}_efficiency.json'

                if not npz_path.exists():
                    errors.append(f'MISSING npz: {npz_path}')
                    continue

                try:
                    d = np.load(npz_path)
                    for k in REQUIRED_NPZ_KEYS:
                        if k not in d:
                            errors.append(f'NPZ missing key "{k}": {npz_path}')
                    if len(d['test_y_true']) == 0:
                        errors.append(f'NPZ empty test_y_true: {npz_path}')
                    npz_ok += 1
                except Exception as e:
                    errors.append(f'NPZ corrupt: {npz_path} ({e})')

                if not eff_path.exists():
                    errors.append(f'MISSING eff: {eff_path}')
                    continue

                try:
                    with open(eff_path) as f:
                        eff = json.load(f)
                    for k in REQUIRED_EFF_KEYS:
                        if k not in eff:
                            errors.append(f'EFF missing key "{k}": {eff_path}')
                    eff_ok += 1
                except Exception as e:
                    errors.append(f'EFF corrupt: {eff_path} ({e})')

    print(f'{"="*60}', flush=True)
    print(f'5-SEED VERIFICATION REPORT', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'Expected: {total_expected} runs', flush=True)
    print(f'NPZ OK:   {npz_ok}/{total_expected}', flush=True)
    print(f'EFF OK:   {eff_ok}/{total_expected}', flush=True)
    print(f'Errors:   {len(errors)}', flush=True)

    if errors:
        print(f'\n--- ERRORS ---', flush=True)
        for e in errors:
            print(f'  {e}', flush=True)
        print(f'\nVERDICT: FAIL', flush=True)
        return 1
    else:
        print(f'\nVERDICT: ALL {total_expected} RUNS VERIFIED OK', flush=True)
        return 0


if __name__ == '__main__':
    sys.exit(verify())
