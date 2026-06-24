"""
Causal-validation harness for Causal-TabDiff (runs on a trained phase2 checkpoint; NO retraining).
alpha = real smoking exposure (export ALPHA_TREATMENT_COLUMN=baseline_cigsmok).
  (A) discriminator f_phi recovers smoking alpha from the trajectory: AUROC / r / energy.
  (B) counterfactual do(smoking): risk(do=1)-risk(do=0) via the TRAINED alpha-conditioned risk head.
  (C) gradient-guidance (Eq.37): energy U and alpha-recovery vs guidance scale s.
  (D) conditioning ablation: test AUPRC/AUROC with alpha={true,shuffle,zero,one}.
Usage:  export ALPHA_TREATMENT_COLUMN=baseline_cigsmok
        python run_causal_validation.py --seed 42 --use_cached_dataset
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.data_module_landmark import load_and_split_data, LandmarkDataset, collate_fn
from src.data.data_module_landmark_cached import LandmarkDatasetCached
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.evaluation.metrics import compute_ranking_metrics

DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'


def _alpha_from_batch(batch):
    """True conditioning exposure for this batch. Requires ALPHA_TREATMENT_COLUMN to be set."""
    if 'alpha_target' not in batch:
        raise SystemExit(
            "FATAL: batch has no 'alpha_target'. Set ALPHA_TREATMENT_COLUMN=baseline_cigsmok "
            "before running causal validation (otherwise alpha is the meaningless landmark index)."
        )
    return batch['alpha_target'].float()


def _risk_logit_with_alpha(model, x, alpha, hl):
    """Headline 2-year risk logit under a given conditioning alpha (the do(smoking) lever)."""
    return model(x, alpha, history_length=hl)['risk_2year_logit']


# ---------------------------------------------------------------------------
# (A) discriminator / energy fidelity
# ---------------------------------------------------------------------------
def exp_discriminator(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            alpha = _alpha_from_batch(batch).to(device)
            alpha_pred = model.base_model.discriminator(x)
            preds.append(alpha_pred.cpu().numpy())
            trues.append(alpha.cpu().numpy())
    p = np.concatenate(preds).flatten()
    a = np.concatenate(trues).flatten()
    energy = float(np.mean((p - a) ** 2))
    r = float(np.corrcoef(p, a)[0, 1]) if np.std(p) > 0 and np.std(a) > 0 else float('nan')
    auroc = float(roc_auc_score(a, p)) if len(np.unique(a)) == 2 else float('nan')
    return {'energy_mse': energy, 'pearson_r': r, 'alpha_recovery_auroc': auroc,
            'n': int(len(a)), 'alpha_pos_rate': float(np.mean(a > 0.5))}


# ---------------------------------------------------------------------------
# (B) counterfactual contrast under do(smoking)
# ---------------------------------------------------------------------------
def exp_counterfactual(model, loader, device):
    model.eval()
    risk0_all, risk1_all, true_alpha, y_all = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            hl = batch['history_length'].to(device)
            a = _alpha_from_batch(batch).to(device)
            zeros = torch.zeros_like(a)
            ones = torch.ones_like(a)
            r0 = torch.sigmoid(_risk_logit_with_alpha(model, x, zeros, hl))
            r1 = torch.sigmoid(_risk_logit_with_alpha(model, x, ones, hl))
            risk0_all.append(r0.cpu().numpy())
            risk1_all.append(r1.cpu().numpy())
            true_alpha.append(a.cpu().numpy())
            y_all.append(batch['y_2year'].numpy())
    r0 = np.concatenate(risk0_all).flatten()
    r1 = np.concatenate(risk1_all).flatten()
    a = np.concatenate(true_alpha).flatten()
    y = np.concatenate(y_all).flatten()
    gap = r1 - r0
    # observational association: predicted risk under each subject's TRUE alpha
    risk_true = np.where(a > 0.5, r1, r0)
    obs_assoc = float(np.mean(risk_true[a > 0.5]) - np.mean(risk_true[a <= 0.5])) \
        if (np.any(a > 0.5) and np.any(a <= 0.5)) else float('nan')
    return {
        'cf_ate_mean_gap': float(np.mean(gap)),
        'cf_gap_std': float(np.std(gap)),
        'cf_sign_consistency': float(np.mean(gap > 0)),    # fraction where do(smoke)=1 raises risk
        'cf_mean_risk_do0': float(np.mean(r0)),
        'cf_mean_risk_do1': float(np.mean(r1)),
        'observational_assoc': obs_assoc,
        'n': int(len(gap)),
    }


# ---------------------------------------------------------------------------
# (C) gradient-guidance functionality (Eq. 37)
# ---------------------------------------------------------------------------
def exp_guidance(model, device, scales, n_gen):
    model.eval()
    base = model.base_model
    rows = []
    # generate equal halves at alpha=0 (former) and alpha=1 (current smoker)
    for s in scales:
        energies, abs_errs, preds, targets = [], [], [], []
        for target_val in (0.0, 1.0):
            alpha = torch.full((n_gen, base.cond_mlp[0].in_features), float(target_val), device=device)
            with torch.no_grad():
                x_gen = base.sample_with_guidance(n_gen, alpha, guidance_scale=float(s))
                pred = base.discriminator(x_gen)
            e = F.mse_loss(pred, alpha[:, :1] if pred.shape[1] == 1 else alpha, reduction='mean')
            energies.append(float(e.item()))
            abs_errs.append(float((pred - target_val).abs().mean().item()))
            preds.append(pred.cpu().numpy().flatten())
            targets.append(np.full(n_gen, target_val))
        p = np.concatenate(preds)
        t = np.concatenate(targets)
        ctrl_auroc = float(roc_auc_score(t, p)) if len(np.unique(t)) == 2 else float('nan')
        rows.append({
            'guidance_scale': float(s),
            'mean_energy': float(np.mean(energies)),
            'mean_abs_alpha_err': float(np.mean(abs_errs)),
            'controllability_auroc': ctrl_auroc,
        })
    return {'sweep': rows, 'n_gen_per_alpha': int(n_gen)}


# ---------------------------------------------------------------------------
# (D) conditioning-signal ablation
# ---------------------------------------------------------------------------
def _eval_alpha_mode(model, loader, device, mode, rng):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            hl = batch['history_length'].to(device)
            a = _alpha_from_batch(batch).to(device)
            if mode == 'true':
                alpha = a
            elif mode == 'zero':
                alpha = torch.zeros_like(a)
            elif mode == 'one':
                alpha = torch.ones_like(a)
            elif mode == 'shuffle':
                perm = torch.from_numpy(rng.permutation(a.shape[0])).to(device)
                alpha = a[perm]
            else:
                raise ValueError(mode)
            logit = _risk_logit_with_alpha(model, x, alpha, hl)
            y_true.append(batch['y_2year'].numpy())
            y_pred.append(torch.sigmoid(logit).cpu().numpy())
    yt = np.concatenate(y_true).flatten()
    yp = np.concatenate(y_pred).flatten()
    m = compute_ranking_metrics(yt, yp)
    return {'auprc': float(m['auprc']), 'auroc': float(m['auroc'])}


def exp_alpha_ablation(model, loader, device, seed):
    rng = np.random.RandomState(seed)
    return {mode: _eval_alpha_mode(model, loader, device, mode, rng)
            for mode in ('true', 'shuffle', 'zero', 'one')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--table_path', type=str, default=DEFAULT_TABLE_PATH)
    ap.add_argument('--batch_size', type=int, default=2048)
    ap.add_argument('--diffusion_steps', type=int, default=100)
    ap.add_argument('--use_cached_dataset', action='store_true')
    ap.add_argument('--guidance_scales', type=str, default='0,0.5,1,2,5')
    ap.add_argument('--n_gen', type=int, default=512)
    ap.add_argument('--ckpt', type=str, default=None,
                    help='phase2 checkpoint (default checkpoints/landmark/phase2_best_seed{seed}.pt)')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    if not os.environ.get('ALPHA_TREATMENT_COLUMN'):
        raise SystemExit("FATAL: export ALPHA_TREATMENT_COLUMN=baseline_cigsmok first.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt or f'checkpoints/landmark/phase2_best_seed{args.seed}.pt'
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"FATAL: checkpoint not found: {ckpt_path} (run phase2 for this seed first).")

    # same split as training (seed-matched) -> evaluate on the held-out test set
    _, _, test_df, landmark_to_idx = load_and_split_data(args.table_path, args.seed, None)
    ds_cls = LandmarkDatasetCached if args.use_cached_dataset else LandmarkDataset
    test_ds = ds_cls(test_df, landmark_to_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0, pin_memory=True)

    sample = next(iter(test_loader))
    t_steps = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    trajectory_len = sample['trajectory_target'].shape[1]
    if 'alpha_target' not in sample:
        raise SystemExit("FATAL: alpha_target absent -- is ALPHA_TREATMENT_COLUMN a baseline col?")

    model = CausalTabDiffTrajectory(t_steps, feature_dim, args.diffusion_steps, trajectory_len).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch','?')}, "
          f"feature_dim={feature_dim}, t_steps={t_steps}, traj_len={trajectory_len})")

    scales = [float(s) for s in args.guidance_scales.split(',') if s.strip() != '']

    results = {
        'seed': args.seed,
        'checkpoint': ckpt_path,
        'feature_dim': int(feature_dim),
        'A_discriminator_energy': exp_discriminator(model, test_loader, device),
        'B_counterfactual_do_smoking': exp_counterfactual(model, test_loader, device),
        'C_guidance_sweep': exp_guidance(model, device, scales, args.n_gen),
        'D_alpha_ablation': exp_alpha_ablation(model, test_loader, device, args.seed),
    }

    out_path = args.out or f'outputs/causal_validation/seed{args.seed}.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # human summary
    A, B, C, D = (results['A_discriminator_energy'], results['B_counterfactual_do_smoking'],
                  results['C_guidance_sweep'], results['D_alpha_ablation'])
    print('=' * 64)
    print(f"CAUSAL VALIDATION | seed {args.seed}")
    print(f"(A) f_phi recovers smoking: AUROC {A['alpha_recovery_auroc']:.3f} | "
          f"Pearson r {A['pearson_r']:.3f} | energy {A['energy_mse']:.4f}")
    print(f"(B) do(smoking) cf ATE {B['cf_ate_mean_gap']:+.4f} | "
          f"sign-consistency {B['cf_sign_consistency']*100:.1f}% | obs-assoc {B['observational_assoc']:+.4f}")
    print(f"    mean risk do(0)={B['cf_mean_risk_do0']:.4f}  do(1)={B['cf_mean_risk_do1']:.4f}")
    print("(C) guidance sweep (energy should fall as s rises):")
    for row in C['sweep']:
        print(f"    s={row['guidance_scale']:>4} | energy {row['mean_energy']:.4f} | "
              f"|alpha_err| {row['mean_abs_alpha_err']:.3f} | ctrl-AUROC {row['controllability_auroc']:.3f}")
    print("(D) alpha ablation (true should win):")
    for mode in ('true', 'shuffle', 'zero', 'one'):
        print(f"    alpha={mode:<8} AUPRC {D[mode]['auprc']:.4f} | AUROC {D[mode]['auroc']:.4f}")
    print(f"Saved -> {out_path}")
    print('=' * 64)


if __name__ == '__main__':
    main()
