#!/usr/bin/env python3
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from baselines.tslib_wrappers import iTransformerWrapper
from data.data_module_landmark import LandmarkDataset, collate_fn, create_dataloaders, load_and_split_data
from evaluation.metrics import compute_all_metrics, find_optimal_threshold


class TrainingLogger:
    def __init__(self, log_dir, script_name):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f'{script_name}.log'
        self.log_fp = open(self.log_file, 'w', buffering=1, encoding='utf-8')

    def log(self, message):
        print(message, flush=True)
        self.log_fp.write(message + '\n')
        self.log_fp.flush()

    def close(self):
        self.log_fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CachedLandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, df, landmark_to_idx, logger, stage_name, progress_every):
        source_dataset = LandmarkDataset(df, landmark_to_idx)
        total_samples = len(source_dataset)
        self.cached_samples = []
        cache_start = time.time()
        logger.log(f"[{stage_name}] cache start | samples {total_samples}")
        for index in range(total_samples):
            self.cached_samples.append(source_dataset[index])
            current = index + 1
            if current == 1 or current % progress_every == 0 or current == total_samples:
                elapsed = time.time() - cache_start
                logger.log(
                    f"[{stage_name}] cache progress {current}/{total_samples} | elapsed {elapsed:.1f}s"
                )
        logger.log(
            f"[{stage_name}] cache complete | samples {total_samples} | total_time {time.time() - cache_start:.1f}s"
        )

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx):
        return self.cached_samples[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--heartbeat_batches', type=int, default=50)
    parser.add_argument('--cache_progress_every', type=int, default=5000)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_name', type=str, default='benchmark_cached_original_internal')
    parser.add_argument(
        '--output_json',
        type=str,
        default='outputs/submission_v3/itransformer/benchmark_cached_original_summary.json',
    )
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gpu_memory():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            check=False,
        )
        return int(result.stdout.strip().split('\n')[0])
    except Exception:
        return -1


def create_cached_dataloaders(
    train_df,
    val_df,
    test_df,
    landmark_to_idx,
    batch_size,
    num_workers,
    logger,
    cache_progress_every,
):
    train_dataset = CachedLandmarkDataset(
        train_df,
        landmark_to_idx,
        logger,
        'cached-train',
        cache_progress_every,
    )
    val_dataset = CachedLandmarkDataset(
        val_df,
        landmark_to_idx,
        logger,
        'cached-val',
        cache_progress_every,
    )
    test_dataset = CachedLandmarkDataset(
        test_df,
        landmark_to_idx,
        logger,
        'cached-test',
        cache_progress_every,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


def train_itransformer(model, train_loader, val_loader, epochs, device, train_df, version_name, logger, heartbeat_batches):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_pos = int(train_df['y_2year'].sum())
    n_total = len(train_df)
    pos_weight = torch.tensor([(n_total - n_pos) / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val_auprc = -float('inf')
    best_model_state = None
    epoch_times = []
    train_batches = len(train_loader)
    total_start = time.time()
    logger.log(
        f"[{version_name}] training start | epochs {epochs} | train_batches {train_batches} | val_batches {len(val_loader)}"
    )
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch['x'].to(device)
            y = batch['y_2year'].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits[:, 1:2], y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            if batch_idx == 1 or batch_idx % heartbeat_batches == 0 or batch_idx == train_batches:
                logger.log(
                    f"[{version_name}] epoch {epoch + 1}/{epochs} | batch {batch_idx}/{train_batches} | "
                    f"loss {loss.item():.6f} | epoch_elapsed {time.time() - epoch_start:.1f}s"
                )
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
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
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.log(
                f"[{version_name}] new_best | epoch {epoch + 1}/{epochs} | best_val_auprc {best_val_auprc:.6f}"
            )
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        train_loss_avg = train_loss_sum / train_batches
        logger.log(
            f"[{version_name}] epoch_done {epoch + 1}/{epochs} | train_loss {train_loss_avg:.6f} | "
            f"val_auprc {val_auprc:.6f} | best_val_auprc {best_val_auprc:.6f} | "
            f"epoch_time {epoch_time:.2f}s | total_elapsed {time.time() - total_start:.2f}s"
        )
    model.load_state_dict(best_model_state)
    logger.log(
        f"[{version_name}] training complete | total_elapsed {time.time() - total_start:.2f}s | best_val_auprc {best_val_auprc:.6f}"
    )
    return model, epoch_times, best_val_auprc


def predict(model, loader, device, version_name, split_name, logger):
    model.eval()
    preds = []
    labels = []
    start_time = time.time()
    logger.log(f"[{version_name}] predict start | split {split_name} | batches {len(loader)}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch['x'].to(device)
            y = batch['y_2year'].cpu().numpy()
            logits = model(x)
            probs = torch.sigmoid(logits[:, 1:2]).cpu().numpy().flatten()
            preds.append(probs)
            labels.append(y.flatten())
            if batch_idx == 1 or batch_idx == len(loader):
                logger.log(
                    f"[{version_name}] predict progress | split {split_name} | batch {batch_idx}/{len(loader)}"
                )
    logger.log(
        f"[{version_name}] predict complete | split {split_name} | total_time {time.time() - start_time:.2f}s"
    )
    return np.concatenate(preds), np.concatenate(labels)


def serialize_metrics(metrics):
    serialized = {}
    for key, value in metrics.items():
        if np.isscalar(value):
            serialized[key] = float(value)
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            try:
                serialized[key] = float(value)
            except (TypeError, ValueError):
                serialized[key] = str(value)
    return serialized


def run_branch(version_name, train_loader, val_loader, test_loader, train_df, device, args, logger):
    set_seed(args.seed)
    model = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
    logger.log(f"[{version_name}] model initialized")
    branch_start = time.time()
    model, epoch_times, best_val_auprc = train_itransformer(
        model,
        train_loader,
        val_loader,
        args.epochs,
        device,
        train_df,
        version_name,
        logger,
        args.heartbeat_batches,
    )
    train_wall_clock = time.time() - branch_start
    val_y_pred, val_y_true = predict(model, val_loader, device, version_name, 'val', logger)
    test_y_pred, test_y_true = predict(model, test_loader, device, version_name, 'test', logger)
    optimal_threshold, val_f1 = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
    metrics = compute_all_metrics(test_y_true, test_y_pred, threshold=optimal_threshold)
    logger.log(
        f"[{version_name}] metrics | test_auprc {metrics['auprc']:.6f} | test_auroc {metrics['auroc']:.6f} | "
        f"test_f1 {metrics['f1']:.6f} | optimal_threshold {optimal_threshold:.6f} | val_f1 {val_f1:.6f}"
    )
    return {
        'train_wall_clock': train_wall_clock,
        'epoch_times': epoch_times,
        'best_val_auprc': best_val_auprc,
        'optimal_threshold': float(optimal_threshold),
        'val_f1': float(val_f1),
        'metrics': serialize_metrics(metrics),
        'val_samples': int(len(val_y_true)),
        'test_samples': int(len(test_y_true)),
    }


def save_summary(summary, output_json, logger):
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2)
    logger.log(f"summary saved | path {output_path}")


def main():
    args = parse_args()
    total_start = time.time()
    with TrainingLogger(log_dir=args.log_dir, script_name=args.log_name) as logger:
        logger.log(f"internal log file | path {logger.log_file}")
        logger.log(
            f"run config | seed {args.seed} | epochs {args.epochs} | batch_size {args.batch_size} | "
            f"num_workers {args.num_workers} | heartbeat_batches {args.heartbeat_batches} | "
            f"cache_progress_every {args.cache_progress_every}"
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.log(f"device | {device}")
        logger.log('stage start | load_and_split_data')
        train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
            'data/landmark_tables/unified_person_landmark_table.pkl',
            seed=args.seed,
        )
        logger.log(
            f"stage done | load_and_split_data | train {len(train_df)} | val {len(val_df)} | test {len(test_df)}"
        )
        mem_before = get_gpu_memory()
        logger.log(f"gpu memory | before {mem_before} MB")

        set_seed(args.seed)
        logger.log('stage start | original dataloaders')
        train_loader_orig, val_loader_orig, test_loader_orig = create_dataloaders(
            train_df,
            val_df,
            test_df,
            landmark_to_idx,
            batch_size=args.batch_size,
            num_workers=0,
        )
        logger.log('stage done | original dataloaders')
        original_result = run_branch(
            'Original',
            train_loader_orig,
            val_loader_orig,
            test_loader_orig,
            train_df,
            device,
            args,
            logger,
        )
        mem_after_orig = get_gpu_memory()
        logger.log(f"gpu memory | after_original {mem_after_orig} MB")

        del train_loader_orig, val_loader_orig, test_loader_orig
        torch.cuda.empty_cache()

        set_seed(args.seed)
        logger.log('stage start | cached dataloaders')
        cache_start = time.time()
        train_loader_cached, val_loader_cached, test_loader_cached = create_cached_dataloaders(
            train_df,
            val_df,
            test_df,
            landmark_to_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            logger=logger,
            cache_progress_every=args.cache_progress_every,
        )
        cache_time = time.time() - cache_start
        logger.log(f"stage done | cached dataloaders | cache_time {cache_time:.2f}s")
        mem_after_cache = get_gpu_memory()
        logger.log(f"gpu memory | after_cache_build {mem_after_cache} MB")
        cached_result = run_branch(
            'Cached',
            train_loader_cached,
            val_loader_cached,
            test_loader_cached,
            train_df,
            device,
            args,
            logger,
        )
        mem_after_cached = get_gpu_memory()
        logger.log(f"gpu memory | after_cached {mem_after_cached} MB")

        max_metric_diff = max(
            abs(original_result['metrics']['auprc'] - cached_result['metrics']['auprc']),
            abs(original_result['metrics']['auroc'] - cached_result['metrics']['auroc']),
            abs(original_result['metrics']['f1'] - cached_result['metrics']['f1']),
        )
        threshold_diff = abs(original_result['optimal_threshold'] - cached_result['optimal_threshold'])
        summary = {
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'heartbeat_batches': args.heartbeat_batches,
            'cache_progress_every': args.cache_progress_every,
            'wall_clock_total': time.time() - total_start,
            'gpu_memory_mb': {
                'before': mem_before,
                'after_original': mem_after_orig,
                'after_cache_build': mem_after_cache,
                'after_cached': mem_after_cached,
            },
            'cache_time': cache_time,
            'original': original_result,
            'cached': cached_result,
            'max_metric_diff': float(max_metric_diff),
            'threshold_diff': float(threshold_diff),
            'acceptable_under_0_01': bool(max_metric_diff < 0.01),
        }
        logger.log(
            f"comparison | original_wall_clock {original_result['train_wall_clock']:.2f}s | "
            f"cached_wall_clock {cached_result['train_wall_clock']:.2f}s | cache_time {cache_time:.2f}s | "
            f"max_metric_diff {max_metric_diff:.6f} | threshold_diff {threshold_diff:.6f} | "
            f"acceptable_under_0_01 {max_metric_diff < 0.01}"
        )
        save_summary(summary, args.output_json, logger)
        logger.log(f"run complete | total_wall_clock {time.time() - total_start:.2f}s")


if __name__ == '__main__':
    main()
