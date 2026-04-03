#!/usr/bin/env python3
"""
iTransformer 正式训练脚本 (fast 版本)
- 基于 train_itransformer_official.py，保持 submission_v3 协议完全不变
- 唯一改动：用 CachedLandmarkDataset 替换原始 DataLoader，消除重复数据处理开销
- 已通过 benchmark_cached_original.py 验证：26x 训练加速，指标差异 < 0.005
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

from baselines.tslib_wrappers import iTransformerWrapper
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
    """将 LandmarkDataset 全部样本预加载到内存列表，消除 __getitem__ 重复计算。"""

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


# ── Training ─────────────────────────────────────────────────────────────────

def train_itransformer(model, train_loader, val_loader, epochs, device, train_df, logger, heartbeat_batches, seed):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_pos = int(train_df['y_2year'].sum())
    n_total = len(train_df)
    pos_weight = torch.tensor([(n_total - n_pos) / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auprc = -float('inf')
    best_model_state = None
    best_epoch = 0
    train_batches = len(train_loader)
    total_start = time.time()

    logger.log(f"[train] start | epochs={epochs} | train_batches={train_batches} | val_batches={len(val_loader)}")

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── train ──
        model.train()
        train_loss_sum = 0.0
        pbar = tqdm(enumerate(train_loader, 1), total=train_batches,
                     desc=f"Epoch {epoch+1}/{epochs} [train]", ncols=100, file=sys.stderr)
        for batch_idx, batch in pbar:
            x = batch['x'].to(device)
            y = batch['y_2year'].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits[:, 1:2], y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # heartbeat 日志（纯文本，写入日志文件）
            if batch_idx == 1 or batch_idx % heartbeat_batches == 0 or batch_idx == train_batches:
                logger.log(
                    f"[train] epoch {epoch+1}/{epochs} | batch {batch_idx}/{train_batches} | "
                    f"loss={loss.item():.6f} | elapsed={time.time() - epoch_start:.1f}s"
                )

        # ── validate ──
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", ncols=100, file=sys.stderr):
                x = batch['x'].to(device)
                y = batch['y_2year'].cpu().numpy()
                logits = model(x)
                probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
                val_probs.append(probs)
                val_labels.append(y.flatten())

        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)

        from sklearn.metrics import average_precision_score
        val_auprc = average_precision_score(val_labels, val_probs)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch + 1
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.log(f"[train] ★ new_best | epoch {epoch+1}/{epochs} | best_val_auprc={best_val_auprc:.6f}")

        epoch_time = time.time() - epoch_start
        train_loss_avg = train_loss_sum / train_batches
        total_elapsed = time.time() - total_start

        # epoch 汇总行（AGENTS.md 规范）
        logger.log(
            f"Epoch {epoch+1}/{epochs} | Seed {seed} | LR 1.0e-3 | "
            f"TrainLoss {train_loss_avg:.4f} | ValAUPRC {val_auprc:.6f} | "
            f"BestValAUPRC {best_val_auprc:.6f} | Time {epoch_time:.1f}s | Total {total_elapsed:.1f}s"
        )

    model.load_state_dict(best_model_state)
    logger.log(f"[train] complete | total_time={time.time() - total_start:.1f}s | best_val_auprc={best_val_auprc:.6f} | best_epoch={best_epoch}")
    return model, best_val_auprc, best_epoch


# ── Predict ──────────────────────────────────────────────────────────────────

def predict(model, loader, device, split_name, logger):
    model.eval()
    preds, labels = [], []
    t0 = time.time()
    logger.log(f"[predict] {split_name} start | batches={len(loader)}")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predict [{split_name}]", ncols=100, file=sys.stderr):
            x = batch['x'].to(device)
            y = batch['y_2year'].cpu().numpy()
            logits = model(x)
            probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
            preds.append(probs)
            labels.append(y.flatten())
    logger.log(f"[predict] {split_name} complete | time={time.time() - t0:.1f}s")
    return np.concatenate(preds), np.concatenate(labels)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='iTransformer Official Fast Training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--heartbeat_batches', type=int, default=50)
    parser.add_argument('--cache_progress_every', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, default='outputs/submission_v3/itransformer')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()

    # ── 日志 ──
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_name = f"train_itransformer_fast_seed{args.seed}_{timestamp}"
    log_path = Path(args.log_dir) / f"{log_name}.log"

    with TrainingLogger(log_path) as logger:
        run_start = time.time()

        # ── config ──
        logger.log("=" * 80)
        logger.log("iTransformer Official Fast Training")
        logger.log("=" * 80)
        logger.log(f"config | seed={args.seed} epochs={args.epochs} batch_size={args.batch_size} "
                    f"num_workers={args.num_workers} heartbeat={args.heartbeat_batches}")
        logger.log(f"config | output_dir={args.output_dir} log_path={log_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.log(f"device | {device} | cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.log(f"device | gpu={torch.cuda.get_device_name(0)}")

        # ── seed ──
        set_seed(args.seed)
        logger.log(f"seed | {args.seed} set (deterministic=True)")

        # ── data ──
        logger.log("[data] loading splits...")
        data_start = time.time()
        train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
            'data/landmark_tables/unified_person_landmark_table.pkl', seed=args.seed
        )
        logger.log(f"[data] splits loaded | time={time.time() - data_start:.1f}s")
        logger.log(f"[data] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
                    f"pos_rate_train={train_df['y_2year'].mean():.4f}")

        # ── cache ──
        cache_start = time.time()
        logger.log("[cache] building cached datasets...")
        train_ds = CachedLandmarkDataset(train_df, landmark_to_idx, logger, 'train', args.cache_progress_every)
        val_ds = CachedLandmarkDataset(val_df, landmark_to_idx, logger, 'val', args.cache_progress_every)
        test_ds = CachedLandmarkDataset(test_df, landmark_to_idx, logger, 'test', args.cache_progress_every)
        cache_time = time.time() - cache_start
        logger.log(f"[cache] all complete | total_time={cache_time:.1f}s")

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
        )

        # ── model ──
        model = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
        logger.log(f"[model] iTransformerWrapper(seq_len=3, enc_in=15, task=classification, num_class=2)")

        # ── train ──
        model, best_val_auprc, best_epoch = train_itransformer(
            model, train_loader, val_loader, args.epochs, device, train_df, logger, args.heartbeat_batches, args.seed
        )

        # ── predict ──
        val_y_pred, val_y_true = predict(model, val_loader, device, 'val', logger)
        test_y_pred, test_y_true = predict(model, test_loader, device, 'test', logger)

        # ── threshold & metrics (submission_v3 协议) ──
        optimal_threshold, val_f1 = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
        logger.log(f"[eval] optimal_threshold={optimal_threshold:.6f} | val_f1={val_f1:.6f}")

        metrics = compute_all_metrics(test_y_true, test_y_pred, threshold=optimal_threshold)
        logger.log(f"[eval] test_AUPRC={metrics['auprc']:.6f} | test_AUROC={metrics['auroc']:.6f} | "
                    f"test_F1={metrics['f1']:.6f} | test_MCC={metrics['mcc']:.6f}")
        logger.log(f"[eval] test_precision={metrics['precision']:.6f} | test_recall={metrics['recall']:.6f} | "
                    f"test_specificity={metrics['specificity']:.6f} | test_balanced_acc={metrics['balanced_accuracy']:.6f}")
        logger.log(f"[eval] brier_score={metrics['brier_score']:.6f} | eo_ratio={metrics['eo_ratio']:.6f}")

        # ── save npz (与 official 完全一致的格式) ──
        os.makedirs(args.output_dir, exist_ok=True)
        npz_path = f"{args.output_dir}/iTransformer_seed{args.seed}_official.npz"
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
            best_val_auprc=best_val_auprc,
        )
        logger.log(f"[save] npz saved: {npz_path}")

        # ── summary ──
        total_time = time.time() - run_start
        logger.log("=" * 80)
        logger.log("RUN COMPLETE")
        logger.log(f"seed={args.seed} | epochs={args.epochs} | total_time={total_time:.1f}s")
        logger.log(f"cache_time={cache_time:.1f}s | best_val_auprc={best_val_auprc:.6f}")
        logger.log(f"test_AUPRC={metrics['auprc']:.6f} | test_AUROC={metrics['auroc']:.6f} | test_F1={metrics['f1']:.6f}")
        logger.log(f"threshold={optimal_threshold:.6f} | npz={npz_path}")
        logger.log("=" * 80)


if __name__ == '__main__':
    main()
