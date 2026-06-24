"""Aggregate per-seed causal-validation JSONs into mean +/- std for the paper.

Usage (from repo root, after running run_causal_validation.py for each seed):
  python aggregate_causal_validation.py --seeds 42 52 62 72 82
"""
import argparse
import json
import os

import numpy as np


def _ms(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 52, 62, 72, 82])
    ap.add_argument('--dir', type=str, default='outputs/causal_validation')
    args = ap.parse_args()

    runs = []
    for s in args.seeds:
        p = os.path.join(args.dir, f'seed{s}.json')
        if not os.path.exists(p):
            print(f"  [skip] missing {p}")
            continue
        with open(p) as f:
            runs.append(json.load(f))
    if not runs:
        raise SystemExit("No seed JSONs found.")
    n = len(runs)

    print('=' * 70)
    print(f"CAUSAL VALIDATION (mean +/- std over {n} seeds: "
          f"{', '.join(str(r['seed']) for r in runs)})")
    print('=' * 70)

    A = [r['A_discriminator_energy'] for r in runs]
    print("(A) Discriminator recovers smoking exposure from trajectory")
    for k, lab in [('alpha_recovery_auroc', 'AUROC'), ('pearson_r', 'Pearson r'), ('energy_mse', 'energy MSE')]:
        m, sd = _ms([a[k] for a in A])
        print(f"    {lab:<12} {m:.4f} +/- {sd:.4f}")

    B = [r['B_counterfactual_do_smoking'] for r in runs]
    print("(B) Counterfactual contrast under do(smoking)")
    for k, lab in [('cf_ate_mean_gap', 'cf ATE (mean gap)'), ('cf_sign_consistency', 'sign-consistency'),
                   ('observational_assoc', 'obs association'),
                   ('cf_mean_risk_do0', 'risk do(0)'), ('cf_mean_risk_do1', 'risk do(1)')]:
        m, sd = _ms([b[k] for b in B])
        print(f"    {lab:<20} {m:+.4f} +/- {sd:.4f}")

    print("(C) Gradient-guidance sweep (energy should fall as s rises)")
    scales = [row['guidance_scale'] for row in runs[0]['C_guidance_sweep']['sweep']]
    for i, s in enumerate(scales):
        em, esd = _ms([r['C_guidance_sweep']['sweep'][i]['mean_energy'] for r in runs])
        am, asd = _ms([r['C_guidance_sweep']['sweep'][i]['mean_abs_alpha_err'] for r in runs])
        cm, csd = _ms([r['C_guidance_sweep']['sweep'][i]['controllability_auroc'] for r in runs])
        print(f"    s={s:>4} | energy {em:.4f}+/-{esd:.4f} | "
              f"|alpha_err| {am:.3f}+/-{asd:.3f} | ctrl-AUROC {cm:.3f}+/-{csd:.3f}")

    print("(D) Conditioning-signal ablation (true alpha should win)")
    for mode in ('true', 'shuffle', 'zero', 'one'):
        pm, psd = _ms([r['D_alpha_ablation'][mode]['auprc'] for r in runs])
        rm, rsd = _ms([r['D_alpha_ablation'][mode]['auroc'] for r in runs])
        print(f"    alpha={mode:<8} AUPRC {pm:.4f}+/-{psd:.4f} | AUROC {rm:.4f}+/-{rsd:.4f}")
    print('=' * 70)


if __name__ == '__main__':
    main()
