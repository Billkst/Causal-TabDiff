#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块消融实验 (Ablation Study) — 正式版
=============================================
对比 Full Model vs w/o 各核心模块，5-seeds × 30 epochs。

消融变体:
  1. Full Model           — 完整模型 (disc_weight=0.5, traj_weight=1.0, guidance_scale=2.0, dual_attn=True)
  2. w/o Discriminator    — 移除因果判别器 (disc_weight=0)
  3. w/o Trajectory Loss  — 移除轨迹损失 (traj_weight=0)
  4. w/o Causal Guidance   — 采样时不做因果梯度引导 (guidance_scale=0)
  5. w/o Dual Attention    — 移除正交双重注意力 (use_time_attn=False, use_feat_attn=False)

评估方式:
  - Direct Forward: trajectory_head 直接预测
  - Sampled (仅在 guidance_scale > 0 时): sample_with_guidance → readout
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

# ============================================================
# 全局配置
# ============================================================
DATA_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'
SEEDS = [42, 52, 62, 72, 82]
EPOCHS = 30
BS = 4096
LR = 1e-3
WD = 1e-4
DIFF_STEPS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 消融变体定义
ABLATION_CONFIGS = {
    'Full Model': {
        'disc_weight': 0.5,
        'traj_weight': 1.0,
        'guidance_scale': 2.0,
        'use_time_attn': True,
        'use_feat_attn': True,
    },
    'w/o Discriminator': {
        'disc_weight': 0.0,
        'traj_weight': 1.0,
        'guidance_scale': 0.0,   # 无判别器则梯度引导也无法工作
        'use_time_attn': True,
        'use_feat_attn': True,
    },
    'w/o Trajectory Loss': {
        'disc_weight': 0.5,
        'traj_weight': 0.0,
        'guidance_scale': 2.0,
        'use_time_attn': True,
        'use_feat_attn': True,
    },
    'w/o Causal Guidance': {
        'disc_weight': 0.5,       # 判别器仍训练（作为辅助损失）
        'traj_weight': 1.0,
        'guidance_scale': 0.0,    # 采样时不用梯度引导
        'use_time_attn': True,
        'use_feat_attn': True,
    },
    'w/o Dual Attention': {
        'disc_weight': 0.5,
        'traj_weight': 1.0,
        'guidance_scale': 2.0,
        'use_time_attn': False,
        'use_feat_attn': False,
    },
}

LOG_DIR = 'logs/ablation_modules'


# ============================================================
# 日志工具
# ============================================================
class DualLogger:
    """同时写终端和日志文件，行缓冲模式。"""

    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.f = open(log_path, 'w', buffering=1)

    def log(self, msg):
        print(msg, flush=True)
        self.f.write(msg + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


# ============================================================
# 训练 & 评估
# ============================================================
def build_model(config):
    """根据消融配置构建模型。"""
    model = CausalTabDiffTrajectory(
        t_steps=3,
        feature_dim=15,
        diffusion_steps=DIFF_STEPS,
        trajectory_len=7,
        use_time_attn=config['use_time_attn'],
        use_feat_attn=config['use_feat_attn'],
    ).to(DEVICE)
    return model


def train_one_epoch(model, train_loader, optimizer, config):
    """训练一个epoch，返回平均损失。"""
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
                + config['disc_weight'] * out['disc_loss']
                + config['traj_weight'] * model.compute_trajectory_loss(out['trajectory'], traj_t, traj_m)
                + F.binary_cross_entropy(out['risk_2year'], y))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_direct(model, val_loader):
    """Direct forward 评估，返回 (y_true, y_pred) 用于全指标计算。"""
    model.eval()
    all_yt, all_yp = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(DEVICE, non_blocking=True)
            alpha = batch['landmark'].float().to(DEVICE, non_blocking=True)
            out = model(x, alpha)
            all_yt.append(batch['y_2year'].cpu().numpy())
            all_yp.append(out['risk_2year'].cpu().numpy())

    y_true = np.concatenate(all_yt).flatten()
    y_pred = np.concatenate(all_yp).flatten()
    return y_true, y_pred


def evaluate_sampled(model, val_loader, guidance_scale):
    """采样评估: sample_with_guidance → block1 → trajectory_head → risk_2year。"""
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

    y_true = np.concatenate(all_yt).flatten()
    y_pred = np.concatenate(all_yp).flatten()
    return y_true, y_pred


def compute_full_metrics(y_true, y_pred, val_y_true=None, val_y_pred=None):
    """计算全部指标: AUPRC, AUROC, F1, Precision, Recall, Brier 等。"""
    ranking = compute_ranking_metrics(y_true, y_pred)

    # 用验证集自身找最优阈值（如果提供了val数据）
    ref_yt = val_y_true if val_y_true is not None else y_true
    ref_yp = val_y_pred if val_y_pred is not None else y_pred
    threshold, _ = find_optimal_threshold(ref_yt, ref_yp, metric='f1')
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


# ============================================================
# 单次实验运行
# ============================================================
def run_single(variant_name, config, seed, logger):
    """对一个变体 + 一个seed运行完整训练与评估。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = get_dataloader(DATA_PATH, 'train', BS, seed)
    val_loader = get_dataloader(DATA_PATH, 'val', BS, seed)

    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_auprc = 0.0
    best_metrics = {}
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    logger.log(f"  [开始训练] {variant_name} | Seed={seed} | Epochs={EPOCHS}")

    for ep in range(EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, config)
        ep_time = time.time() - t0

        # Direct forward 评估
        y_true, y_pred = evaluate_direct(model, val_loader)
        metrics = compute_full_metrics(y_true, y_pred)
        val_auprc = metrics['auprc'] if not np.isnan(metrics['auprc']) else 0.0

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_metrics = metrics.copy()
            best_metrics['epoch'] = ep + 1
            # 保存最佳模型状态用于后续采样评估
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 每个 epoch 输出基础信息
        msg = (f"  Epoch {ep+1:>2}/{EPOCHS} | Seed {seed} | LR {LR:.1e} | "
               f"TrainLoss {train_loss:.4f} | ValAUPRC {val_auprc:.4f} | "
               f"BestValAUPRC {best_auprc:.4f} | Time {ep_time:.1f}s")
        logger.log(msg)

        # 每10个epoch或最后一个epoch输出完整指标
        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            logger.log(f"    AUROC={metrics['auroc']:.4f} | F1={metrics['f1']:.4f} | "
                       f"Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | "
                       f"Brier={metrics['brier_score']:.6f}")

    # 采样评估（仅在 guidance_scale > 0 时）
    sampled_metrics = {}
    if config['guidance_scale'] > 0.0:
        logger.log(f"  [采样评估] guidance_scale={config['guidance_scale']}")
        # 加载最佳模型
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        t0 = time.time()
        y_true_s, y_pred_s = evaluate_sampled(model, val_loader, config['guidance_scale'])
        sampled_metrics = compute_full_metrics(y_true_s, y_pred_s)
        elapsed = time.time() - t0
        logger.log(f"    Sampled AUPRC={sampled_metrics['auprc']:.4f} | "
                   f"AUROC={sampled_metrics['auroc']:.4f} | "
                   f"F1={sampled_metrics['f1']:.4f} | Time={elapsed:.1f}s")

    result = {
        'variant': variant_name,
        'seed': seed,
        'direct': best_metrics,
        'sampled': sampled_metrics,
        'config': config,
    }

    logger.log(f"  [完成] {variant_name} Seed={seed} | "
               f"Best Direct AUPRC={best_auprc:.4f} (Epoch {best_metrics.get('epoch', '?')})")
    logger.log("")

    return result


# ============================================================
# 主流程
# ============================================================
def main():
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f'ablation_modules_{timestamp}.log')
    results_path = os.path.join(LOG_DIR, f'ablation_modules_{timestamp}.json')
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = DualLogger(log_path)

    logger.log("=" * 80)
    logger.log("Causal-TabDiff 模块消融实验 (Ablation Study)")
    logger.log("=" * 80)
    logger.log(f"时间: {timestamp}")
    logger.log(f"设备: {DEVICE}")
    logger.log(f"配置: EPOCHS={EPOCHS} | BS={BS} | LR={LR} | WD={WD} | DIFF_STEPS={DIFF_STEPS}")
    logger.log(f"Seeds: {SEEDS}")
    logger.log(f"消融变体: {list(ABLATION_CONFIGS.keys())}")
    logger.log("")

    # 打印消融配置表
    logger.log("消融配置详情:")
    logger.log(f"{'变体':<25} {'disc_w':>7} {'traj_w':>7} {'guid_s':>7} {'time_attn':>10} {'feat_attn':>10}")
    logger.log("-" * 75)
    for name, cfg in ABLATION_CONFIGS.items():
        logger.log(f"{name:<25} {cfg['disc_weight']:>7.1f} {cfg['traj_weight']:>7.1f} "
                   f"{cfg['guidance_scale']:>7.1f} {str(cfg['use_time_attn']):>10} {str(cfg['use_feat_attn']):>10}")
    logger.log("")

    all_results = {}
    total_start = time.time()

    for variant_name, config in ABLATION_CONFIGS.items():
        logger.log("=" * 60)
        logger.log(f"消融变体: {variant_name}")
        logger.log("=" * 60)

        variant_results = []
        for seed in SEEDS:
            result = run_single(variant_name, config, seed, logger)
            variant_results.append(result)

        # 汇总该变体的跨seed统计
        direct_auprcs = [r['direct']['auprc'] for r in variant_results]
        direct_aurocs = [r['direct']['auroc'] for r in variant_results]
        direct_f1s = [r['direct']['f1'] for r in variant_results]

        logger.log(f"--- {variant_name} 汇总 (Direct) ---")
        logger.log(f"  AUPRC: {np.mean(direct_auprcs):.4f} ± {np.std(direct_auprcs):.4f}  "
                   f"[{', '.join(f'{v:.4f}' for v in direct_auprcs)}]")
        logger.log(f"  AUROC: {np.mean(direct_aurocs):.4f} ± {np.std(direct_aurocs):.4f}  "
                   f"[{', '.join(f'{v:.4f}' for v in direct_aurocs)}]")
        logger.log(f"  F1:    {np.mean(direct_f1s):.4f} ± {np.std(direct_f1s):.4f}  "
                   f"[{', '.join(f'{v:.4f}' for v in direct_f1s)}]")

        if config['guidance_scale'] > 0.0:
            sampled_auprcs = [r['sampled']['auprc'] for r in variant_results]
            sampled_aurocs = [r['sampled']['auroc'] for r in variant_results]
            logger.log(f"--- {variant_name} 汇总 (Sampled, g={config['guidance_scale']}) ---")
            logger.log(f"  AUPRC: {np.mean(sampled_auprcs):.4f} ± {np.std(sampled_auprcs):.4f}  "
                       f"[{', '.join(f'{v:.4f}' for v in sampled_auprcs)}]")
            logger.log(f"  AUROC: {np.mean(sampled_aurocs):.4f} ± {np.std(sampled_aurocs):.4f}  "
                       f"[{', '.join(f'{v:.4f}' for v in sampled_aurocs)}]")

        logger.log("")
        all_results[variant_name] = variant_results

    # 最终汇总表
    total_elapsed = time.time() - total_start
    logger.log("=" * 80)
    logger.log("最终消融对比表 (Direct Forward)")
    logger.log("=" * 80)
    logger.log(f"{'变体':<25} {'AUPRC':>16} {'AUROC':>16} {'F1':>16} {'Precision':>12} {'Recall':>12} {'Brier':>12}")
    logger.log("-" * 105)

    for variant_name in ABLATION_CONFIGS:
        vr = all_results[variant_name]
        auprcs = [r['direct']['auprc'] for r in vr]
        aurocs = [r['direct']['auroc'] for r in vr]
        f1s = [r['direct']['f1'] for r in vr]
        precs = [r['direct']['precision'] for r in vr]
        recs = [r['direct']['recall'] for r in vr]
        briers = [r['direct']['brier_score'] for r in vr]

        logger.log(f"{variant_name:<25} "
                   f"{np.mean(auprcs):.4f}±{np.std(auprcs):.4f} "
                   f"{np.mean(aurocs):.4f}±{np.std(aurocs):.4f} "
                   f"{np.mean(f1s):.4f}±{np.std(f1s):.4f} "
                   f"{np.mean(precs):.4f}±{np.std(precs):.4f}"
                   f"{np.mean(recs):.4f}±{np.std(recs):.4f}"
                   f"{np.mean(briers):.6f}±{np.std(briers):.6f}")

    logger.log("")
    logger.log(f"总耗时: {total_elapsed/3600:.1f} 小时")

    # 保存JSON
    # 把 numpy 类型转为 Python 原生类型
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = {}
    for vn, vr_list in all_results.items():
        serializable_results[vn] = []
        for r in vr_list:
            sr = {
                'variant': r['variant'],
                'seed': r['seed'],
                'direct': {k: to_serializable(v) for k, v in r['direct'].items()},
                'sampled': {k: to_serializable(v) for k, v in r['sampled'].items()},
                'config': r['config'],
            }
            serializable_results[vn].append(sr)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.log(f"结果已保存: {results_path}")
    logger.log(f"日志已保存: {log_path}")
    logger.close()


if __name__ == '__main__':
    main()
