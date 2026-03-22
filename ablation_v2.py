#!/usr/bin/env python
"""
消融实验 v2 — 修正版
- guidance_scale: 纯推理期消融，每seed只训练1次，扫5个guidance_scale采样评估
- diffusion_steps: 不同扩散步数训练，direct forward评估
- traj_weight: 不同轨迹权重训练，direct forward评估

时间预算：3个消融总计 ≤ 8小时
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, '/home/UserData/ljx/Project_2/Causal-TabDiff')
from src.data.data_module_landmark import get_dataloader
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.evaluation.metrics import compute_ranking_metrics

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

DATA_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'
SEEDS = [42, 52, 62, 72, 82]
EPOCHS = 10
BS = 4096
LR = 1e-3
DEVICE = torch.device('cuda')


def train_model(seed, diff_steps, disc_w, traj_w, log):
    """训练模型并返回模型和最佳direct-forward AUPRC"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = get_dataloader(DATA_PATH, 'train', BS, seed)
    val_loader = get_dataloader(DATA_PATH, 'val', BS, seed)

    model = CausalTabDiffTrajectory(3, 15, diff_steps, 7).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_auprc = 0.0
    for ep in tqdm(range(EPOCHS), desc=f"seed={seed}", file=sys.stderr):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            x = batch['x'].to(DEVICE, non_blocking=True)
            alpha = batch['landmark'].float().to(DEVICE, non_blocking=True)
            y = batch['y_2year'].to(DEVICE, non_blocking=True)
            traj_t = batch['trajectory_target'].to(DEVICE, non_blocking=True)
            traj_m = batch['trajectory_valid_mask'].to(DEVICE, non_blocking=True)

            opt.zero_grad()
            out = model(x, alpha)
            loss = (out['diff_loss']
                    + disc_w * out['disc_loss']
                    + traj_w * model.compute_trajectory_loss(out['trajectory'], traj_t, traj_m)
                    + F.binary_cross_entropy(out['risk_2year'], y))
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        # 每5个epoch或最后一个epoch验证
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            auprc = evaluate_direct(model, val_loader)
            if not np.isnan(auprc) and auprc > best_auprc:
                best_auprc = auprc
            msg = f"  Ep{ep+1}/{EPOCHS} | Loss {ep_loss/len(train_loader):.4f} | AUPRC {auprc:.4f} | Best {best_auprc:.4f}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

    return model, best_auprc


def evaluate_direct(model, val_loader):
    """Direct forward评估：不经过采样，直接用trajectory_head预测"""
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(DEVICE, non_blocking=True)
            alpha = batch['landmark'].float().to(DEVICE, non_blocking=True)
            out = model(x, alpha)
            yt.append(batch['y_2year'].cpu().numpy())
            yp.append(out['risk_2year'].cpu().numpy())
    y_true = np.concatenate(yt).flatten()
    y_pred = np.concatenate(yp).flatten()
    return compute_ranking_metrics(y_true, y_pred)['auprc']


def evaluate_sampled(model, val_loader, guidance_scale):
    """采样评估：sample_with_guidance生成样本 → block1 → trajectory_head → risk"""
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in val_loader:
            alpha = batch['landmark'].float().to(DEVICE, non_blocking=True)
            bs = alpha.shape[0]

            # 采样生成
            sampled_x = model.sample_with_guidance(bs, alpha, guidance_scale=guidance_scale)

            # 用已训练的readout路径评估：block1 → mean pool → trajectory_head → risk
            c_emb = model.base_model.cond_mlp(alpha)
            h = model.base_model.block1(sampled_x, c_emb)
            h_pooled = h.mean(dim=1)
            traj_probs = torch.sigmoid(model.trajectory_head(h_pooled))
            risk = model.compute_2year_risk(traj_probs)

            yt.append(batch['y_2year'].cpu().numpy())
            yp.append(risk.cpu().numpy())

    y_true = np.concatenate(yt).flatten()
    y_pred = np.concatenate(yp).flatten()
    return compute_ranking_metrics(y_true, y_pred)['auprc']


# ============================================================
# 消融1: guidance_scale (纯推理期消融)
# ============================================================
def run_guidance_scale(log):
    guidance_values = [0.0, 0.5, 1.0, 2.0, 4.0]
    results = {str(g): [] for g in guidance_values}

    log.write("=== 消融1: guidance_scale (因果梯度引导强度) ===\n")
    log.write(f"测试值: {guidance_values}\n")
    log.write("训练配置: disc_weight=1.0, traj_weight=1.0, diffusion_steps=100\n\n")

    for seed in SEEDS:
        msg = f"训练 seed={seed} (disc_weight=1.0固定)"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

        # 每个seed只训练1次
        model, direct_auprc = train_model(seed, diff_steps=100, disc_w=1.0, traj_w=1.0, log=log)

        # 扫不同guidance_scale采样评估
        val_loader = get_dataloader(DATA_PATH, 'val', BS, seed)
        for g in guidance_values:
            t0 = time.time()
            auprc = evaluate_sampled(model, val_loader, guidance_scale=g)
            elapsed = time.time() - t0
            results[str(g)].append(auprc)
            msg = f"  guidance_scale={g} | SampledAUPRC {auprc:.4f} | DirectAUPRC {direct_auprc:.4f} | {elapsed:.1f}s"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

        log.write("\n")

    return results


# ============================================================
# 消融2: diffusion_steps
# ============================================================
def run_diffusion_steps(log):
    values = [25, 50, 100, 150, 200]
    results = {str(v): [] for v in values}

    log.write("=== 消融2: diffusion_steps (扩散步数) ===\n")
    log.write(f"测试值: {values}\n")
    log.write("训练配置: disc_weight=1.0, traj_weight=1.0\n\n")

    for val in values:
        for seed in SEEDS:
            msg = f"diffusion_steps={val} seed={seed}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

            model, best_auprc = train_model(seed, diff_steps=val, disc_w=1.0, traj_w=1.0, log=log)
            results[str(val)].append(best_auprc)
            log.write(f"  Best={best_auprc:.4f}\n\n")
            log.flush()

    return results


# ============================================================
# 消融3: traj_weight
# ============================================================
def run_traj_weight(log):
    values = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = {str(v): [] for v in values}

    log.write("=== 消融3: traj_weight (轨迹损失权重) ===\n")
    log.write(f"测试值: {values}\n")
    log.write("训练配置: disc_weight=1.0, diffusion_steps=100\n\n")

    for val in values:
        for seed in SEEDS:
            msg = f"traj_weight={val} seed={seed}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

            model, best_auprc = train_model(seed, diff_steps=100, disc_w=1.0, traj_w=val, log=log)
            results[str(val)].append(best_auprc)
            log.write(f"  Best={best_auprc:.4f}\n\n")
            log.flush()

    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    ablation_type = sys.argv[1]
    os.makedirs('logs/ablations', exist_ok=True)

    log_path = f'logs/ablations/{ablation_type}.log'
    log = open(log_path, 'w', buffering=1)

    start_time = time.time()

    if ablation_type == 'guidance_scale':
        results = run_guidance_scale(log)
        values = [0.0, 0.5, 1.0, 2.0, 4.0]
    elif ablation_type == 'diffusion_steps':
        results = run_diffusion_steps(log)
        values = [25, 50, 100, 150, 200]
    elif ablation_type == 'traj_weight':
        results = run_traj_weight(log)
        values = [0.0, 0.5, 1.0, 1.5, 2.0]
    else:
        print(f"未知消融类型: {ablation_type}", flush=True)
        sys.exit(1)

    # 汇总
    log.write("\n=== Summary ===\n")
    for val in values:
        auprcs = results[str(val)]
        mean, std = np.mean(auprcs), np.std(auprcs)
        msg = f"{ablation_type}={val} | Mean={mean:.4f}±{std:.4f}"
        print(msg, flush=True)
        log.write(msg + "\n")

    elapsed = time.time() - start_time
    msg = f"\n总耗时: {elapsed/3600:.1f}小时"
    print(msg, flush=True)
    log.write(msg + "\n")
    log.close()

    with open(f'logs/ablations/{ablation_type}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
