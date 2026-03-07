import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.wrappers import STaSyWrapper
from src.data.data_module import get_dataloader


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_meta(data_dir: str):
    env_path = os.environ.get('DATASET_METADATA_PATH', '').strip()
    if env_path:
        if not os.path.isabs(env_path):
            env_path = os.path.abspath(os.path.join(ROOT, env_path))
        with open(env_path, "r", encoding="utf-8") as f:
            return json.load(f)

    meta_path = os.path.join(data_dir, "dataset_metadata.json")
    if not os.path.exists(meta_path):
        meta_path = "src/data/dataset_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconstruct_real_semantic(batch, meta, device):
    real_x_analog = batch["x"].to(device)
    cat_raw = batch["x_cat_raw"].to(device).float()
    d_orig = len(meta["columns"])
    real_x_raw_t = torch.zeros((real_x_analog.shape[0], real_x_analog.shape[1], d_orig), device=device)

    analog_offset = 0
    cat_idx = 0
    for i_col, col_meta in enumerate(meta["columns"]):
        if col_meta["type"] == "continuous":
            dim = col_meta["dim"]
            real_x_raw_t[:, :, i_col:i_col + 1] = real_x_analog[:, :, analog_offset:analog_offset + dim]
            analog_offset += dim
        else:
            dim = col_meta["dim"]
            real_x_raw_t[:, :, i_col:i_col + 1] = cat_raw[:, :, cat_idx:cat_idx + 1]
            analog_offset += dim
            cat_idx += 1

    return real_x_raw_t[:, -1, :]


def train_tstr(fake_x, fake_y, real_x, real_y, seed=42):
    clf = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=seed)
    clf.fit(fake_x, fake_y)
    prob = clf.predict_proba(real_x)[:, 1]
    pred = clf.predict(real_x)
    return roc_auc_score(real_y, prob), f1_score(real_y, pred)


def domain_auc(real_x, fake_x, seed=42):
    X = np.concatenate([real_x, fake_x], axis=0)
    y = np.concatenate([np.ones(real_x.shape[0]), np.zeros(fake_x.shape[0])], axis=0)
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=seed)
    clf.fit(x_tr, y_tr)
    prob = clf.predict_proba(x_te)[:, 1]
    return roc_auc_score(y_te, prob)


def nn_memorization_ratio(real_x, fake_x):
    eps = 1e-6
    mu = real_x.mean(axis=0, keepdims=True)
    sd = real_x.std(axis=0, keepdims=True) + eps
    real_z = (real_x - mu) / sd
    fake_z = (fake_x - mu) / sd

    nn_rr = NearestNeighbors(n_neighbors=2).fit(real_z)
    rr_dist, _ = nn_rr.kneighbors(real_z)
    real_to_real = rr_dist[:, 1]

    nn_fr = NearestNeighbors(n_neighbors=1).fit(real_z)
    fr_dist, _ = nn_fr.kneighbors(fake_z)
    fake_to_real = fr_dist[:, 0]

    med_rr = float(np.median(real_to_real))
    med_fr = float(np.median(fake_to_real))
    ratio = med_fr / (med_rr + eps)
    return ratio, med_fr, med_rr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit_epochs", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--report_path", type=str, default="logs/evaluation/stasy_leakage_report.md")
    parser.add_argument("--fail_on_risk", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_meta(args.data_dir)

    dataloader = get_dataloader(data_dir=args.data_dir, batch_size=args.batch_size, debug_mode=True)
    sample_batch = next(iter(dataloader))
    t_steps = sample_batch["x"].shape[1]
    feature_dim = sample_batch["x"].shape[2]

    wrapper = STaSyWrapper(t_steps=t_steps, feature_dim=feature_dim, diffusion_steps=100)
    wrapper.fit(dataloader, epochs=args.fit_epochs, device=device, debug_mode=True)

    real_x_all, fake_x_all, real_y_all, fake_y_all = [], [], [], []
    for i, batch in enumerate(dataloader):
        if i >= args.max_batches:
            break
        alpha = batch["alpha_target"].to(device)
        real_y = batch["y"].reshape(-1).cpu().numpy()

        real_x = reconstruct_real_semantic(batch, meta, device=device)
        fake_x, fake_y = wrapper.sample(batch_size=real_x.shape[0], alpha_target=alpha, device=device)

        real_x_all.append(real_x.detach().cpu().numpy())
        fake_x_all.append(fake_x.detach().cpu().numpy())
        real_y_all.append((real_y > 0.5).astype(int))
        fake_y_all.append((fake_y.reshape(-1).detach().cpu().numpy() > 0.5).astype(int))

    real_x = np.concatenate(real_x_all, axis=0)
    fake_x = np.concatenate(fake_x_all, axis=0)
    real_y = np.concatenate(real_y_all, axis=0)
    fake_y = np.concatenate(fake_y_all, axis=0)

    fake_pos_rate = float(fake_y.mean())
    real_pos_rate = float(real_y.mean())

    if len(np.unique(fake_y)) < 2 or len(np.unique(real_y)) < 2:
        tstr_auc, tstr_f1 = 0.5, 0.0
        tstr_auc_shuf, tstr_f1_shuf = 0.5, 0.0
    else:
        tstr_auc, tstr_f1 = train_tstr(fake_x, fake_y, real_x, real_y, seed=args.seed)
        shuffled = fake_y.copy()
        np.random.shuffle(shuffled)
        tstr_auc_shuf, tstr_f1_shuf = train_tstr(fake_x, shuffled, real_x, real_y, seed=args.seed)

    d_auc = domain_auc(real_x, fake_x, seed=args.seed)
    mem_ratio, med_fr, med_rr = nn_memorization_ratio(real_x, fake_x)

    label_gap = abs(fake_pos_rate - real_pos_rate)

    print("=== STaSy Leakage Quick Check ===")
    print(f"samples={len(real_y)}, device={device}")
    print(f"label prevalence real={real_pos_rate:.4f}, fake={fake_pos_rate:.4f}")
    print(f"TSTR auc={tstr_auc:.4f}, f1={tstr_f1:.4f}")
    print(f"Shuffle-control TSTR auc={tstr_auc_shuf:.4f}, f1={tstr_f1_shuf:.4f}")
    print(f"Domain AUC(real-vs-fake)={d_auc:.4f}")
    print(f"NN memorization ratio(fake->real / real->real)={mem_ratio:.4f} (med_fr={med_fr:.4f}, med_rr={med_rr:.4f})")

    risks = []
    if tstr_auc > 0.95 and tstr_f1 > 0.80 and tstr_auc_shuf > 0.70:
        risks.append("高风险：高TSTR且shuffle对照仍偏高，疑似标签泄漏。")
    if mem_ratio < 0.80:
        risks.append("高风险：fake样本离real样本过近，疑似记忆化。")
    if label_gap > 0.20:
        risks.append("中风险：fake/real标签比例差异过大，可能导致TSTR失真。")

    gate_fail_reasons = []
    if tstr_auc_shuf > 0.70:
        gate_fail_reasons.append(f"shuffle-control AUC过高({tstr_auc_shuf:.4f} > 0.7000)")
    if d_auc > 0.90:
        gate_fail_reasons.append(f"domain AUC过高({d_auc:.4f} > 0.9000)")
    if label_gap > 0.20:
        gate_fail_reasons.append(f"标签比例差异过大({label_gap:.4f} > 0.2000)")

    gate_pass = len(gate_fail_reasons) == 0

    if not risks:
        print("RISK: 未发现强泄漏信号（基于快速检查）。")
    else:
        print("RISK:")
        for r in risks:
            print(f"- {r}")

    print(f"GATE: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        for reason in gate_fail_reasons:
            print(f"- {reason}")

    report_lines = [
        "# STaSy Leakage Diagnostic Report",
        "",
        f"- samples: {len(real_y)}",
        f"- device: {device}",
        f"- label_prevalence_real: {real_pos_rate:.4f}",
        f"- label_prevalence_fake: {fake_pos_rate:.4f}",
        f"- label_gap: {label_gap:.4f}",
        f"- tstr_auc: {tstr_auc:.4f}",
        f"- tstr_f1: {tstr_f1:.4f}",
        f"- shuffle_tstr_auc: {tstr_auc_shuf:.4f}",
        f"- shuffle_tstr_f1: {tstr_f1_shuf:.4f}",
        f"- domain_auc: {d_auc:.4f}",
        f"- nn_mem_ratio: {mem_ratio:.4f}",
        "",
        f"## Gate Verdict: {'PASS' if gate_pass else 'FAIL'}",
    ]
    if gate_pass:
        report_lines.append("- No blocking leakage signal under current gate thresholds.")
    else:
        report_lines.append("- Blocking reasons:")
        for reason in gate_fail_reasons:
            report_lines.append(f"  - {reason}")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"REPORT: {report_path}")

    if args.fail_on_risk and not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
