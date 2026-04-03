#!/usr/bin/env python3
"""
TSDiff 正式训练脚本 — submission_v3 协议
- 生成式扩散模型 (DDPM) 用于 Layer1 2-year risk baseline
- 训练：在 train set 上训练 TSDiff 生成模型
- 预测：采样 n 个样本，取生成的 Y 作为 "预测概率"
- 评估：val 上找 F1 最优阈值 → 固定应用到 test
- 输出：统一 npz 格式，与 iTransformer 完全一致
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data.data_module_landmark import LandmarkDataset, collate_fn, load_and_split_data
from evaluation.metrics import compute_all_metrics, find_optimal_threshold


# ── Logging ──────────────────────────────────────────────────────────────────

class TrainingLogger:
    """双通道日志：终端 + 文件，行缓冲，实时可见。"""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.log_path, 'w', buffering=1, encoding='utf-8')

    def log(self, msg):
        print(msg, flush=True)
        self.fp.write(msg + '\n')
        self.fp.flush()

    def close(self):
        self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Cached Dataset ───────────────────────────────────────────────────────────

class CachedLandmarkDataset(torch.utils.data.Dataset):
    """将 LandmarkDataset 全部样本预加载到内存列表。"""

    def __init__(self, df, landmark_to_idx, logger, stage_name, progress_every=5000):
        source = LandmarkDataset(df, landmark_to_idx)
        total = len(source)
        self.cached_samples = []
        t0 = time.time()
        logger.log(f"[cache] {stage_name} start | samples={total}")
        for i in range(total):
            self.cached_samples.append(source[i])
            done = i + 1
            if done == 1 or done % progress_every == 0 or done == total:
                logger.log(f"[cache] {stage_name} {done}/{total} | elapsed={time.time() - t0:.1f}s")
        logger.log(f"[cache] {stage_name} complete | samples={total} | time={time.time() - t0:.1f}s")

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx):
        return self.cached_samples[idx]


# ── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── TSDiff Training ─────────────────────────────────────────────────────────

def train_tsdiff(model, train_loader, epochs, device, logger, heartbeat_batches, seed):
    """训练 TSDiff DDPM 生成模型，返回 (model, best_epoch, best_train_loss)。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_batches = len(train_loader)
    total_start = time.time()

    best_train_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    logger.log(f"[train] start | epochs={epochs} | train_batches={train_batches}")

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(enumerate(train_loader, 1), total=train_batches,
                     desc=f"Epoch {epoch+1}/{epochs} [train]", ncols=100, file=sys.stderr)
        for batch_idx, batch in pbar:
            x = batch['x'].to(device)
            y = batch['y_2year'].to(device)

            # 扁平化: (B, seq_len, feat) + (B, 1) → (B, 1, seq_len*feat+1)
            x_flat = x.reshape(x.shape[0], -1)
            xy = torch.cat([x_flat, y], dim=1).unsqueeze(1)

            optimizer.zero_grad()
            loss = model.train_step(xy)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx == 1 or batch_idx % heartbeat_batches == 0 or batch_idx == train_batches:
                logger.log(
                    f"[train] epoch {epoch+1}/{epochs} | batch {batch_idx}/{train_batches} | "
                    f"loss={loss.item():.6f} | elapsed={time.time() - epoch_start:.1f}s"
                )

        train_loss_avg = train_loss_sum / train_batches
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start

        # TSDiff 是生成模型，没有 val metric；用 train loss 选最佳 epoch
        if train_loss_avg < best_train_loss:
            best_train_loss = train_loss_avg
            best_epoch = epoch + 1
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.log(f"[train] ★ new_best | epoch {epoch+1}/{epochs} | best_train_loss={best_train_loss:.6f}")

        logger.log(
            f"Epoch {epoch+1}/{epochs} | Seed {seed} | LR 1.0e-3 | "
            f"TrainLoss {train_loss_avg:.6f} | BestTrainLoss {best_train_loss:.6f} | "
            f"Time {epoch_time:.1f}s | Total {total_elapsed:.1f}s"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    logger.log(f"[train] complete | total_time={time.time() - total_start:.1f}s | best_train_loss={best_train_loss:.6f} | best_epoch={best_epoch}")
    return model, best_epoch, best_train_loss


# ── Predict (生成式采样) ─────────────────────────────────────────────────────

def predict_by_sampling(model, n_samples, seq_len, feature_dim, device, split_name, logger, batch_size=2048):
    """
    TSDiff 预测：采样 n_samples 个样本，取生成的 Y 概率作为 "预测"。
    分批采样避免 OOM。
    """
    logger.log(f"[predict] {split_name} start | n_samples={n_samples}")
    t0 = time.time()
    model.eval()

    total_dim = seq_len * feature_dim + 1
    all_probs = []

    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        with torch.no_grad():
            samples = model.sample(bs, seq_len=1, features=total_dim, device=device)
        samples = samples.squeeze(1)  # (bs, total_dim)
        y_raw = samples[:, -1]  # 最后一维是 Y
        probs = torch.sigmoid(y_raw).cpu().numpy()
        all_probs.append(probs)
        remaining -= bs

    y_pred = np.concatenate(all_probs, axis=0)
    logger.log(f"[predict] {split_name} complete | time={time.time() - t0:.1f}s")
    return y_pred


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='TSDiff Official Training (submission_v3)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--heartbeat_batches', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='outputs/submission_v3/tsdiff')
    args = parser.parse_args()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = f"logs/train_tsdiff_seed{args.seed}_{timestamp}.log"

    with TrainingLogger(log_path) as logger:
        logger.log("=" * 80)
        logger.log("TSDiff Official Training (submission_v3)")
        logger.log("=" * 80)
        logger.log(f"config | seed={args.seed} epochs={args.epochs} batch_size={args.batch_size} "
                    f"num_workers={args.num_workers} heartbeat={args.heartbeat_batches}")
        logger.log(f"config | output_dir={args.output_dir} log_path={log_path}")

        # ── device ──
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.log(f"device | {device} | cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.log(f"device | gpu={torch.cuda.get_device_name(0)}")

        # ── seed ──
        set_seed(args.seed)
        logger.log(f"seed | {args.seed} set (deterministic=True)")

        # ── data ──
        logger.log("[data] loading splits...")
        t0 = time.time()
        table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
        train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=args.seed)
        logger.log(f"[data] splits loaded | time={time.time() - t0:.1f}s")
        logger.log(f"[data] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
                    f"pos_rate_train={train_df['y_2year'].mean():.4f}")

        # ── cached datasets ──
        logger.log("[cache] building cached datasets...")
        cache_start = time.time()
        train_ds = CachedLandmarkDataset(train_df, landmark_to_idx, logger, 'train')
        val_ds = CachedLandmarkDataset(val_df, landmark_to_idx, logger, 'val')
        test_ds = CachedLandmarkDataset(test_df, landmark_to_idx, logger, 'test')
        cache_time = time.time() - cache_start
        logger.log(f"[cache] all complete | total_time={cache_time:.1f}s")

        # ── dataloaders ──
        loader_kwargs = dict(
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
        test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, **loader_kwargs)

        # ── 获取数据维度 ──
        sample_batch = next(iter(train_loader))
        seq_len = sample_batch['x'].shape[1]   # 3
        feature_dim = sample_batch['x'].shape[2]  # 15
        total_dim = seq_len * feature_dim + 1  # 46

        # ── model ──
        from baselines.tsdiff_core.model import TSDiffDDPM
        model = TSDiffDDPM(input_dim=total_dim, timesteps=100).to(device)
        logger.log(f"[model] TSDiffDDPM(input_dim={total_dim}, timesteps=100)")

        # ── train ──
        model, best_epoch, best_train_loss = train_tsdiff(
            model, train_loader, args.epochs, device, logger, args.heartbeat_batches, args.seed
        )

        # ── predict (生成式采样) ──
        n_val = len(val_ds)
        n_test = len(test_ds)
        val_y_pred = predict_by_sampling(model, n_val, seq_len, feature_dim, device, 'val', logger)
        test_y_pred = predict_by_sampling(model, n_test, seq_len, feature_dim, device, 'test', logger)

        # ── 真实标签 ──
        val_y_true = np.array([val_ds[i]['y_2year'].item() for i in range(n_val)], dtype=np.float32)
        test_y_true = np.array([test_ds[i]['y_2year'].item() for i in range(n_test)], dtype=np.float32)

        # ── threshold & metrics (submission_v3 协议) ──
        optimal_threshold, val_f1 = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
        logger.log(f"[eval] optimal_threshold={optimal_threshold:.6f} | val_f1={val_f1:.6f}")

        metrics = compute_all_metrics(test_y_true, test_y_pred, threshold=optimal_threshold)
        logger.log(f"[eval] test_AUPRC={metrics['auprc']:.6f} | test_AUROC={metrics['auroc']:.6f} | "
                    f"test_F1={metrics['f1']:.6f} | test_MCC={metrics['mcc']:.6f}")
        logger.log(f"[eval] test_precision={metrics['precision']:.6f} | test_recall={metrics['recall']:.6f} | "
                    f"test_specificity={metrics['specificity']:.6f} | test_balanced_acc={metrics['balanced_accuracy']:.6f}")

        # ── save npz ──
        os.makedirs(args.output_dir, exist_ok=True)
        npz_path = f"{args.output_dir}/TSDiff_seed{args.seed}_official.npz"
        np.savez(
            npz_path,
            val_y_true=val_y_true,
            val_y_pred=val_y_pred,
            test_y_true=test_y_true,
            test_y_pred=test_y_pred,
            optimal_threshold=optimal_threshold,
            test_f1=metrics['f1'],
            test_precision=metrics['precision'],
            test_recall=metrics['recall'],
            test_specificity=metrics['specificity'],
            test_balanced_accuracy=metrics['balanced_accuracy'],
            test_mcc=metrics['mcc'],
            seed=args.seed,
            best_epoch=best_epoch,
            best_val_auprc=0.0,  # TSDiff 是生成模型，无 val AUPRC；占位保持格式一致
            best_train_loss=best_train_loss,
        )
        logger.log(f"[save] npz saved: {npz_path}")

        # ── summary ──
        total_time = time.time() - t0
        logger.log("=" * 80)
        logger.log("RUN COMPLETE")
        logger.log(f"seed={args.seed} | epochs={args.epochs} | total_time={total_time:.1f}s")
        logger.log(f"cache_time={cache_time:.1f}s | best_train_loss={best_train_loss:.6f} | best_epoch={best_epoch}")
        logger.log(f"test_AUPRC={metrics['auprc']:.6f} | test_AUROC={metrics['auroc']:.6f} | test_F1={metrics['f1']:.6f}")
        logger.log(f"threshold={optimal_threshold:.6f} | npz={npz_path}")
        logger.log("=" * 80)


if __name__ == '__main__':
    main()
