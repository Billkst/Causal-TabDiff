#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验补救脚本 — 只训练 Full Model，保存 checkpoint，补做采样评估
====================================================================
已有可直接复用的结果（来自 ablation_modules_20260320_081505.json）：
  - w/o Trajectory Loss: Direct 评估有效
  - w/o Dual Attention:  Direct 评估有效

本脚本只做：
  1. 训练 Full Model (disc_w=0.5, traj_w=1.0) × 5 seeds × 30 epochs
  2. 保存每个 seed 的 best checkpoint
  3. 对同一 checkpoint 分别做:
     - Direct Forward 评估 → Full Model 结果
     - Sampled g=2.0 → Full Model (Sampled) 结果
     - Sampled g=0.0 → w/o Causal Guidance 结果
  4. 合并历史数据，输出完整消融表

预计耗时：~6 小时（训练 5h + 采样 1h）
"""
import sys
import os
import json
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, '/home/UserData/ljx/Project_2/Causal-TabDiff')
from src.data.data_module_landmark import get_dataloader
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.evaluation.metrics import compute_ranking_metrics, compute_all_metrics, find_optimal_threshold

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

DATA_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'
SEEDS = [42, 52, 62, 72, 82]
EPOCHS = 30
BS = 4096
LR = 1e-3
WD = 1e-4
DIFF_STEPS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOG_DIR = 'logs/ablation_modules'
CKPT_DIR = 'logs/ablation_modules/checkpoints'
PREV_RESULTS_PATH = 'logs/ablation_modules/ablation_modules_20260320_081505.json'


class DualLogger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.f = open(log_path, 'w', buffering=1)

    def log(self, msg):
        print(msg, flush=True)
        self.f.write(msg + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def train_one_epoch(model, train_loader, optimizer, disc_weight, traj_weight):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(train_loader, desc="  train", ncols=80, file=sys.stderr, leave=False):
        x = batch['x'].to(DEVICE, non_blocking=True)
        alpha = batch['landmark'].float().to(DEVICE, non_blocking=True)
        y = batch['y_2year'].to(DEVICE, non_blocking=True)
        traj_t = batch['trajectory_target'].to(DEVICE, non_blocking=True)
        traj_m = batch['trajectory_valid_mask'].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        out = model(x, alpha)
        loss = (out['diff_loss']
                + disc_weight * out['disc_loss']
                + traj_weight * model.compute_trajectory_loss(out['trajectory'], traj_t, traj_m)
                + F.binary_cross_entropy(out['risk_2year'], y))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_direct(model, val_loader):
    model.eval()
    all_yt, all_yp = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(DEVICE, non_blocking=True)
            alpha = batch['landmark'].float().to(DEVICE, non_blocking=True)
            out = model(x, alpha)
            all_yt.append(batch['y_2year'].cpu().numpy())
            all_yp.append(out['risk_2year'].cpu().numpy())
    return np.concatenate(all_yt).flatten(), np.concatenate(all_yp).flatten()


def evaluate_sampled(model, val_loader, guidance_scale):
    model.eval()
    all_yt, all_yp = [], []
    with torch.no_grad():
        for batch in val_loader:
            alpha = batch['landmark'].float().to(DEVICE, non_blocking=True)
            bs = alpha.shape[0]
            sampled_x = model.sample_with_guidance(bs, alpha, guidance_scale=guidance_scale)
            c_emb = model.base_model.cond_mlp(alpha)
            h = model.base_model.block1(sampled_x, c_emb)
            h_pooled = h.mean(dim=1)
            traj_probs = torch.sigmoid(model.trajectory_head(h_pooled))
            risk = model.compute_2year_risk(traj_probs)
            all_yt.append(batch['y_2year'].cpu().numpy())
            all_yp.append(risk.cpu().numpy())
    return np.concatenate(all_yt).flatten(), np.concatenate(all_yp).flatten()


def compute_full_metrics(y_true, y_pred):
    ranking = compute_ranking_metrics(y_true, y_pred)
    threshold, _ = find_optimal_threshold(y_true, y_pred, metric='f1')
    all_m = compute_all_metrics(y_true, y_pred, threshold=threshold)
    return {
        'auprc': float(ranking.get('auprc', 0.0)),
        'auroc': float(ranking.get('auroc', 0.0)),
        'f1': float(all_m.get('f1', 0.0)),
        'precision': float(all_m.get('precision', 0.0)),
        'recall': float(all_m.get('recall', 0.0)),
        'brier_score': float(all_m.get('brier_score', 0.0)),
        'threshold': float(threshold),
    }


def main():
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f'ablation_fix_{timestamp}.log')
    results_path = os.path.join(LOG_DIR, f'ablation_fix_{timestamp}.json')
    os.makedirs(CKPT_DIR, exist_ok=True)

    logger = DualLogger(log_path)
    logger.log("=" * 70)
    logger.log("消融补救: 训练 Full Model + 采样评估 (g=0 / g=2)")
    logger.log("=" * 70)
    logger.log(f"时间: {timestamp} | 设备: {DEVICE}")
    logger.log(f"EPOCHS={EPOCHS} | BS={BS} | LR={LR} | Seeds={SEEDS}")
    logger.log("")

    disc_weight = 0.5
    traj_weight = 1.0
    full_direct = []
    full_sampled_g2 = []
    wo_guidance_sampled_g0 = []

    total_start = time.time()

    for seed in SEEDS:
        logger.log(f"{'='*50}")
        logger.log(f"Seed={seed}")
        logger.log(f"{'='*50}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = get_dataloader(DATA_PATH, 'train', BS, seed)
        val_loader = get_dataloader(DATA_PATH, 'val', BS, seed)

        model = CausalTabDiffTrajectory(3, 15, DIFF_STEPS, 7).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        best_auprc = 0.0
        best_metrics = {}
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for ep in range(EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, disc_weight, traj_weight)
            ep_time = time.time() - t0

            y_true, y_pred = evaluate_direct(model, val_loader)
            metrics = compute_full_metrics(y_true, y_pred)
            val_auprc = metrics['auprc'] if not np.isnan(metrics['auprc']) else 0.0

            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_metrics = metrics.copy()
                best_metrics['epoch'] = ep + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            msg = (f"  Epoch {ep+1:>2}/{EPOCHS} | Seed {seed} | LR {LR:.1e} | "
                   f"TrainLoss {train_loss:.4f} | ValAUPRC {val_auprc:.4f} | "
                   f"BestValAUPRC {best_auprc:.4f} | Time {ep_time:.1f}s")
            logger.log(msg)

            if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
                logger.log(f"    AUROC={metrics['auroc']:.4f} | F1={metrics['f1']:.4f} | "
                           f"Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | "
                           f"Brier={metrics['brier_score']:.6f}")

        full_direct.append({'seed': seed, 'metrics': best_metrics})

        # 保存 checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f'full_model_seed{seed}.pt')
        torch.save(best_state, ckpt_path)
        logger.log(f"  Checkpoint 已保存: {ckpt_path}")

        # 加载 best 模型做采样评估
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        # Sampled g=2.0 → Full Model
        logger.log(f"  [采样评估] guidance_scale=2.0 (Full Model)")
        t0 = time.time()
        yt_s, yp_s = evaluate_sampled(model, val_loader, guidance_scale=2.0)
        m_g2 = compute_full_metrics(yt_s, yp_s)
        logger.log(f"    AUPRC={m_g2['auprc']:.4f} | AUROC={m_g2['auroc']:.4f} | "
                   f"F1={m_g2['f1']:.4f} | Time={time.time()-t0:.1f}s")
        full_sampled_g2.append({'seed': seed, 'metrics': m_g2})

        # Sampled g=0.0 → w/o Causal Guidance
        logger.log(f"  [采样评估] guidance_scale=0.0 (w/o Causal Guidance)")
        t0 = time.time()
        yt_s0, yp_s0 = evaluate_sampled(model, val_loader, guidance_scale=0.0)
        m_g0 = compute_full_metrics(yt_s0, yp_s0)
        logger.log(f"    AUPRC={m_g0['auprc']:.4f} | AUROC={m_g0['auroc']:.4f} | "
                   f"F1={m_g0['f1']:.4f} | Time={time.time()-t0:.1f}s")
        wo_guidance_sampled_g0.append({'seed': seed, 'metrics': m_g0})

        logger.log("")

    # 加载历史结果
    logger.log("=" * 70)
    logger.log("合并历史消融结果")
    logger.log("=" * 70)

    with open(PREV_RESULTS_PATH) as f:
        prev = json.load(f)

    wo_traj_direct = [
        {'seed': r['seed'], 'metrics': r['direct']}
        for r in prev['w/o Trajectory Loss']
    ]
    wo_dual_direct = [
        {'seed': r['seed'], 'metrics': r['direct']}
        for r in prev['w/o Dual Attention']
    ]

    # 汇总表
    def summarize(name, results, eval_type, logger):
        auprcs = [r['metrics']['auprc'] for r in results]
        aurocs = [r['metrics']['auroc'] for r in results]
        f1s = [r['metrics']['f1'] for r in results]
        precs = [r['metrics']['precision'] for r in results]
        recs = [r['metrics']['recall'] for r in results]
        briers = [r['metrics']['brier_score'] for r in results]
        logger.log(f"  {name:<25} ({eval_type})")
        logger.log(f"    AUPRC:  {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}  {[round(a,4) for a in auprcs]}")
        logger.log(f"    AUROC:  {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        logger.log(f"    F1:     {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        logger.log(f"    Prec:   {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        logger.log(f"    Recall: {np.mean(recs):.4f} ± {np.std(recs):.4f}")
        logger.log(f"    Brier:  {np.mean(briers):.6f} ± {np.std(briers):.6f}")
        return {
            'auprc_mean': float(np.mean(auprcs)), 'auprc_std': float(np.std(auprcs)),
            'auroc_mean': float(np.mean(aurocs)), 'auroc_std': float(np.std(aurocs)),
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'precision_mean': float(np.mean(precs)), 'precision_std': float(np.std(precs)),
            'recall_mean': float(np.mean(recs)), 'recall_std': float(np.std(recs)),
            'brier_mean': float(np.mean(briers)), 'brier_std': float(np.std(briers)),
            'per_seed': results,
        }

    logger.log("")
    logger.log("=" * 70)
    logger.log("最终消融对比表")
    logger.log("=" * 70)
    logger.log("")

    final = {}
    final['Full Model (Direct)'] = summarize('Full Model', full_direct, 'Direct', logger)
    logger.log("")
    final['Full Model (Sampled g=2)'] = summarize('Full Model', full_sampled_g2, 'Sampled g=2.0', logger)
    logger.log("")
    final['w/o Causal Guidance'] = summarize('w/o Causal Guidance', wo_guidance_sampled_g0, 'Sampled g=0.0', logger)
    logger.log("")
    final['w/o Trajectory Loss'] = summarize('w/o Trajectory Loss', wo_traj_direct, 'Direct', logger)
    logger.log("")
    final['w/o Dual Attention'] = summarize('w/o Dual Attention', wo_dual_direct, 'Direct', logger)
    logger.log("")

    # 紧凑对比表
    logger.log("=" * 70)
    logger.log("紧凑对比 (论文用)")
    logger.log("=" * 70)
    logger.log(f"{'变体':<30} {'AUPRC':>16} {'AUROC':>16} {'F1':>16}")
    logger.log("-" * 80)
    for name, data in final.items():
        logger.log(f"{name:<30} "
                   f"{data['auprc_mean']:.4f}±{data['auprc_std']:.4f} "
                   f"{data['auroc_mean']:.4f}±{data['auroc_std']:.4f} "
                   f"{data['f1_mean']:.4f}±{data['f1_std']:.4f}")

    total_elapsed = time.time() - total_start
    logger.log(f"\n总耗时: {total_elapsed/3600:.1f} 小时")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=float)

    logger.log(f"结果已保存: {results_path}")
    logger.log(f"日志已保存: {log_path}")
    logger.close()


if __name__ == '__main__':
    main()
