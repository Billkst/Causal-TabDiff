import argparse
import datetime
import json
import os
import sys
import time
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_module_landmark import load_and_split_data, LandmarkDataset, collate_fn
from src.data.data_module_landmark_cached import LandmarkDatasetCached
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.utils.training_logger import TrainingLogger
from src.evaluation.metrics import compute_ranking_metrics, find_optimal_threshold, compute_threshold_metrics
from src.evaluation.efficiency import EfficiencyTracker

DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'
DEFAULT_OUTPUT_ROOT = 'outputs/param_discussion_layer2'
DEFAULT_LOG_ROOT = 'logs/param_discussion_layer2'
DEFAULT_CHECKPOINT_ROOT = 'checkpoints/param_discussion_layer2'

# Module-level mutable roots (overridden by CLI args in main)
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
LOG_ROOT = DEFAULT_LOG_ROOT
CHECKPOINT_ROOT = DEFAULT_CHECKPOINT_ROOT

PARAM_MATRIX = {
    'focal_gamma': [0.0, 1.0, 2.0, 3.0, 5.0],
    'oversample_ratio': [1.0, 3.0, 5.0, 10.0, 20.0],
    'disc_weight': [0.0, 0.25, 0.5, 1.0, 2.0],
    'use_uw': [True, False],
}

OFFICIAL_DEFAULTS = {
    'focal_gamma': 2.0,
    'oversample_ratio': 5.0,
    'disc_weight': 0.5,
    'use_uw': True,
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_alpha_target(batch, device):
    if 'alpha_target' in batch:
        return batch['alpha_target'].float().to(device)
    if 'landmark' in batch:
        return batch['landmark'].float().to(device)
    raise KeyError("Batch must contain either 'alpha_target' or 'landmark'.")


def evaluate_model(model, dataloader, device):
    model.eval()
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device)
            hl = batch['history_length'].to(device)
            outputs = model(x, alpha, history_length=hl)
            all_y_true.append(batch['y_2year'].numpy())
            all_y_pred.append(torch.sigmoid(outputs['risk_2year_logit']).cpu().numpy())
    return np.concatenate(all_y_true).flatten(), np.concatenate(all_y_pred).flatten()


def focal_loss_with_logits(logits, targets, gamma=2.0, pos_weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = ((1 - pt) ** gamma) * bce_loss
    if pos_weight is not None:
        weight = torch.where(targets > 0.5, pos_weight.reshape(-1, 1), torch.ones_like(targets))
        focal_loss = focal_loss * weight
    return focal_loss.mean()


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigma[i])
            total = total + precision * loss + self.log_sigma[i]
        return total

    def get_weights(self):
        return torch.exp(-self.log_sigma).detach().cpu().tolist()


def parse_seeds(args):
    if args.seeds:
        return [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    if args.seed is not None:
        return [args.seed]
    return [42]


def normalize_value_for_name(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return str(value)
    return str(value)


def normalize_value_for_summary(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def parse_values(param_name, values_str):
    raw_values = [v.strip() for v in values_str.split(',') if v.strip()]
    parsed = []
    for value in raw_values:
        if param_name == 'use_uw':
            lowered = value.lower()
            if lowered not in {'true', 'false'}:
                raise ValueError('--values for use_uw must be true,false')
            parsed.append(lowered == 'true')
        else:
            parsed.append(float(value))
    return parsed


def get_selected_params(args):
    if args.param:
        if args.param not in PARAM_MATRIX:
            raise ValueError(f'Unknown parameter: {args.param}')
        return [args.param]
    return list(PARAM_MATRIX.keys())


def get_values_for_param(param_name, args):
    values = list(PARAM_MATRIX[param_name])
    if args.values:
        values = parse_values(param_name, args.values)
    if args.skip_default:
        default_value = OFFICIAL_DEFAULTS[param_name]
        values = [v for v in values if v != default_value]
    return values


def make_run_config(param_name, param_value, seed):
    config = dict(OFFICIAL_DEFAULTS)
    config[param_name] = param_value
    config['param_name'] = param_name
    config['param_value'] = param_value
    config['seed'] = seed
    return config


def make_paths(param_name, param_value, seed):
    value_str = normalize_value_for_name(param_value)
    base_name = f'{param_name}_{value_str}_seed{seed}'
    output_dir = os.path.join(OUTPUT_ROOT, param_name)
    log_dir = os.path.join(LOG_ROOT, param_name)
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, param_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return {
        'base_name': base_name,
        'pred_path': os.path.join(output_dir, f'{base_name}.npz'),
        'efficiency_path': os.path.join(output_dir, f'{base_name}_efficiency.json'),
        'checkpoint_path': os.path.join(checkpoint_dir, f'{base_name}_best.pt'),
        'log_dir': log_dir,
        'output_dir': output_dir,
        'value_str': value_str,
    }


def run_single_experiment(args, run_config):
    seed = run_config['seed']
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    param_name = run_config['param_name']
    param_value = run_config['param_value']
    paths = make_paths(param_name, param_value, seed)

    with TrainingLogger(paths['log_dir'], paths['base_name']) as logger:
        logger.log(f'Device: {device}')
        logger.log('Phase 2: Two-stage training (pretrain diffusion → finetune classification)')

        train_df, val_df, test_df, landmark_to_idx = load_and_split_data(args.table_path, seed)
        train_df = cast(pd.DataFrame, train_df)
        val_df = cast(pd.DataFrame, val_df)
        test_df = cast(pd.DataFrame, test_df)
        dataset_cls = LandmarkDatasetCached if args.use_cached_dataset else LandmarkDataset
        logger.log(f'Using dataset: {dataset_cls.__name__}')

        train_dataset = dataset_cls(train_df, landmark_to_idx)
        val_dataset = dataset_cls(val_df, landmark_to_idx)
        test_dataset = dataset_cls(test_df, landmark_to_idx)

        train_labels = np.asarray(train_df['y_2year'], dtype=int)
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0

        nw = args.num_workers
        train_loader_unsup = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
        )

        if n_pos > 0:
            sample_weights = np.where(
                train_labels == 1,
                run_config['oversample_ratio'] * n_neg / n_pos,
                1.0,
            )
            sampler = WeightedRandomSampler(
                sample_weights.tolist(),
                len(train_labels),
                replacement=True,
            )
        else:
            sampler = None

        train_loader_sup = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
        )

        sample = next(iter(train_loader_unsup))
        t_steps = sample['x'].shape[1]
        feature_dim = sample['x'].shape[2]
        trajectory_len = sample['trajectory_target'].shape[1]

        model = CausalTabDiffTrajectory(t_steps, feature_dim, 100, trajectory_len).to(device)

        tracker = EfficiencyTracker()
        tracker.set_model_size(model)
        tracker.set_gpu_info()
        tracker.set_metadata(
            model_name='CausalTabDiff',
            layer='layer2',
            seed=seed,
            experiment_type='param_discussion_official',
            command=' '.join(sys.argv),
        )
        tracker.metrics['param_name'] = param_name
        tracker.metrics['param_value'] = normalize_value_for_summary(param_value)
        tracker.metrics['focal_gamma'] = float(run_config['focal_gamma'])
        tracker.metrics['oversample_ratio'] = float(run_config['oversample_ratio'])
        tracker.metrics['disc_weight'] = float(run_config['disc_weight'])
        tracker.metrics['use_uw'] = bool(run_config['use_uw'])
        tracker.set_output_paths(output_file=paths['pred_path'], log_file=str(logger.log_file))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_pos_ratio = (n_pos / len(train_df) * 100.0) if len(train_df) > 0 else 0.0

        logger.log('=' * 64)
        logger.log('实验: Causal-TabDiff Layer2 Parameter Discussion (Official Protocol)')
        logger.log(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ({device})")
        logger.log(f'Seed: {seed}')
        logger.log(
            f"配置: pretrain_epochs={args.pretrain_epochs}, finetune_epochs={args.finetune_epochs}, "
            f"batch_size={args.batch_size}"
        )
        logger.log(
            f"      focal_gamma={run_config['focal_gamma']}, oversample_ratio={run_config['oversample_ratio']}, "
            f"disc_weight={run_config['disc_weight']}, use_uw={run_config['use_uw']}"
        )
        logger.log(f'讨论参数: {param_name}={param_value}')
        logger.log(f'数据: {args.table_path}')
        logger.log(
            f'      Train: {len(train_df)} samples, Val: {len(val_df)} samples, '
            f'Test: {len(test_df)} samples'
        )
        logger.log(f'      Pos ratio: train={train_pos_ratio:.1f}%')
        logger.log(
            f'Model: CausalTabDiffTrajectory (total_params={total_params}, '
            f'trainable={trainable_params})'
        )
        logger.log(f'输出: {paths["pred_path"]}')
        logger.log(f'效率: {paths["efficiency_path"]}')
        logger.log(f'日志: {logger.log_file}')
        logger.log(f"命令: {' '.join(sys.argv)}")
        logger.log('=' * 64)

        total_start_time = time.time()
        pretrain_epoch_times = []
        finetune_epoch_times = []
        finetune_epochs_completed = 0

        with tracker.track_training():
            logger.log(f'\n=== Phase A: Pretrain diffusion backbone ({args.pretrain_epochs} epochs) ===')
            for p in model.risk_head.parameters():
                p.requires_grad = False
            for p in model.trajectory_head.parameters():
                p.requires_grad = False

            pretrain_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3,
                weight_decay=1e-4,
            )

            with tracker.track_phase_time('pretrain'):
                for epoch in range(args.pretrain_epochs):
                    epoch_start = time.time()
                    model.train()
                    total_loss = 0.0
                    pbar = tqdm(
                        train_loader_unsup,
                        desc=f'Pretrain {epoch + 1}/{args.pretrain_epochs}',
                        ncols=80,
                        file=sys.stderr,
                    )
                    last_heartbeat = time.time()
                    for i, batch in enumerate(pbar):
                        x = batch['x'].to(device)
                        alpha = build_alpha_target(batch, device)
                        hl = batch['history_length'].to(device)

                        pretrain_opt.zero_grad()
                        outputs = model(x, alpha, history_length=hl)
                        loss = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
                        loss.backward()
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.log(
                                f'⚠️ FATAL: Loss is {loss.item()} at pretrain epoch {epoch + 1}. Terminating.'
                            )
                            sys.exit(1)
                        pretrain_opt.step()
                        total_loss += loss.item()
                        pbar.set_postfix(loss=f'{loss.item():.4f}')

                        if time.time() - last_heartbeat > 60:
                            logger.log(
                                f'  [Heartbeat] Pretrain Epoch {epoch + 1} | Batch {i + 1}/{len(train_loader_unsup)} | '
                                f'Loss {loss.item():.4f} | Elapsed {time.time() - epoch_start:.1f}s'
                            )
                            last_heartbeat = time.time()

                    avg_loss = total_loss / len(train_loader_unsup)
                    epoch_time = time.time() - epoch_start
                    pretrain_epoch_times.append(epoch_time)
                    logger.log(
                        f'  Pretrain {epoch + 1}/{args.pretrain_epochs} | '
                        f'Loss {avg_loss:.4f} | Time {epoch_time:.1f}s'
                    )

            for p in model.parameters():
                p.requires_grad = True

            logger.log(f'\n=== Phase B: Finetune with {"UW loss" if run_config["use_uw"] else "simple sum"} ({args.finetune_epochs} epochs) ===')

            if run_config['use_uw']:
                uw_loss = UncertaintyWeightedLoss(n_tasks=3).to(device)
                uw_loss.log_sigma.data[2] = -1.0
                finetune_opt = torch.optim.Adam(
                    [
                        {'params': model.risk_head.parameters(), 'lr': 1e-3},
                        {'params': model.trajectory_head.parameters(), 'lr': 5e-4},
                        {'params': model.base_model.parameters(), 'lr': 1e-4},
                        {'params': uw_loss.parameters(), 'lr': 1e-2},
                    ],
                    weight_decay=1e-4,
                )
            else:
                uw_loss = None
                finetune_opt = torch.optim.Adam(
                    [
                        {'params': model.risk_head.parameters(), 'lr': 1e-3},
                        {'params': model.trajectory_head.parameters(), 'lr': 5e-4},
                        {'params': model.base_model.parameters(), 'lr': 1e-4},
                    ],
                    weight_decay=1e-4,
                )

            pos_weight = torch.tensor([pos_weight_val], device=device)
            best_val_auprc = -1.0
            patience = 30
            patience_counter = 0
            checkpoint_path = paths['checkpoint_path']
            best_val_y_true = np.array([])
            best_val_y_pred = np.array([])

            with tracker.track_phase_time('finetune'):
                for epoch in range(args.finetune_epochs):
                    finetune_epochs_completed = epoch + 1
                    epoch_start = time.time()
                    model.train()
                    total_loss = 0.0

                    pbar = tqdm(
                        train_loader_sup,
                        desc=f'Finetune {epoch + 1}/{args.finetune_epochs}',
                        ncols=80,
                        file=sys.stderr,
                    )
                    last_heartbeat = time.time()
                    for i, batch in enumerate(pbar):
                        x = batch['x'].to(device)
                        alpha = build_alpha_target(batch, device)
                        y_2year = batch['y_2year'].to(device)
                        traj_target = batch['trajectory_target'].to(device)
                        traj_mask = batch['trajectory_valid_mask'].to(device)
                        hl = batch['history_length'].to(device)

                        finetune_opt.zero_grad()
                        outputs = model(x, alpha, history_length=hl)

                        loss_gen = outputs['diff_loss'] + run_config['disc_weight'] * outputs['disc_loss']
                        loss_traj = model.compute_trajectory_loss(
                            outputs['trajectory_logits'],
                            traj_target,
                            traj_mask,
                        )
                        loss_cls = focal_loss_with_logits(
                            outputs['risk_2year_logit'],
                            y_2year,
                            gamma=run_config['focal_gamma'],
                            pos_weight=pos_weight,
                        )

                        if run_config['use_uw']:
                            assert uw_loss is not None
                            loss = uw_loss([loss_gen, loss_traj, loss_cls])
                        else:
                            loss = loss_gen + loss_traj + loss_cls

                        loss.backward()
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.log(
                                f'⚠️ FATAL: Loss is {loss.item()} at finetune epoch {epoch + 1}. Terminating.'
                            )
                            sys.exit(1)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        finetune_opt.step()
                        total_loss += loss.item()
                        pbar.set_postfix(loss=f'{loss.item():.2f}')

                        if time.time() - last_heartbeat > 60:
                            logger.log(
                                f'  [Heartbeat] Finetune Epoch {epoch + 1} | Batch {i + 1}/{len(train_loader_sup)} | '
                                f'Loss {loss.item():.4f} | Elapsed {time.time() - epoch_start:.1f}s'
                            )
                            last_heartbeat = time.time()

                    train_loss = total_loss / len(train_loader_sup)
                    val_y_true, val_y_pred = evaluate_model(model, val_loader, device)
                    metrics = compute_ranking_metrics(val_y_true, val_y_pred)
                    val_auprc = metrics['auprc'] if not np.isnan(metrics['auprc']) else 0.0
                    val_auroc = metrics['auroc'] if not np.isnan(metrics['auroc']) else 0.5

                    if val_auprc > best_val_auprc:
                        old_best = best_val_auprc
                        best_val_auprc = val_auprc
                        patience_counter = 0
                        best_val_y_true = val_y_true
                        best_val_y_pred = val_y_pred
                        torch.save(
                            {
                                'epoch': epoch + 1,
                                'model_state_dict': model.state_dict(),
                                'best_val_auprc': best_val_auprc,
                                'val_auroc': val_auroc,
                                'seed': seed,
                            },
                            checkpoint_path,
                        )
                        logger.log(
                            f'  ★ New Best ValAUPRC: {old_best:.4f} → {best_val_auprc:.4f} '
                            f'(epoch {epoch + 1}) | Checkpoint saved'
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.log(f'  Early stopping at epoch {epoch + 1}')
                            break

                    epoch_time = time.time() - epoch_start
                    finetune_epoch_times.append(epoch_time)
                    logger.epoch_log(
                        epoch=epoch + 1,
                        total_epochs=args.finetune_epochs,
                        seed=seed,
                        lr=finetune_opt.param_groups[0]['lr'],
                        train_loss=train_loss,
                        val_loss=0.0,
                        best_val_metric=best_val_auprc,
                        epoch_time=epoch_time,
                        ValAUPRC=val_auprc,
                        ValAUROC=val_auroc,
                    )

                    if (epoch + 1) % 10 == 0:
                        gpu_mem = (
                            torch.cuda.max_memory_allocated() / 1024 / 1024
                            if torch.cuda.is_available() else 0
                        )
                        total_elapsed = time.time() - total_start_time
                        logger.log(f'  Epoch {epoch + 1}/{args.finetune_epochs} | FULL METRICS')
                        logger.log(f'    ValAUPRC {val_auprc:.4f} | ValAUROC {val_auroc:.4f}')
                        logger.log(
                            f'    BestValAUPRC {best_val_auprc:.4f} | '
                            f'EarlyStopping patience {patience_counter}/{patience}'
                        )
                        if run_config['use_uw']:
                            assert uw_loss is not None
                            w = uw_loss.get_weights()
                            logger.log(
                                f'    UW weights: gen={w[0]:.2f} traj={w[1]:.2f} cls={w[2]:.2f}'
                            )
                        else:
                            logger.log('    UW weights: disabled (simple sum)')
                        if len(np.unique(val_y_true)) >= 2:
                            thr, _ = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
                            y_bin = (val_y_pred >= thr).astype(int)
                            tm = compute_threshold_metrics(val_y_true, y_bin)
                            logger.log(
                                f"    F1 {tm['f1']:.4f} | Prec {tm['precision']:.4f} | "
                                f"Rec {tm['recall']:.4f} | Spec {tm['specificity']:.4f} | "
                                f"BalAcc {tm['balanced_accuracy']:.4f} | MCC {tm['mcc']:.4f}"
                            )
                        logger.log(
                            f'    GPU Memory: {gpu_mem:.0f} MB peak | Epoch Time: {epoch_time:.1f}s | '
                            f'Total Elapsed: {total_elapsed:.1f}s'
                        )

        with tracker.track_phase_time('evaluation'):
            logger.log('\nEvaluating on test set ...')
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                logger.log(f"Loaded best model from epoch {ckpt['epoch']}")
            else:
                ckpt = {'epoch': args.finetune_epochs, 'best_val_auprc': best_val_auprc}

            if len(np.unique(best_val_y_true)) >= 2:
                optimal_threshold, _ = find_optimal_threshold(best_val_y_true, best_val_y_pred, metric='f1')
            else:
                optimal_threshold = 0.5
            logger.log(f'Optimal threshold (val F1): {optimal_threshold:.4f}')

            test_y_true, test_y_pred = evaluate_model(model, test_loader, device)
            test_m = compute_ranking_metrics(test_y_true, test_y_pred)
            logger.log(f"Test | AUPRC {test_m['auprc']:.4f} | AUROC {test_m['auroc']:.4f}")

            test_y_bin = (test_y_pred >= optimal_threshold).astype(int)
            test_thr_m = compute_threshold_metrics(test_y_true, test_y_bin)
            logger.log(
                f"Test (thr={optimal_threshold:.4f}) | F1 {test_thr_m['f1']:.4f} | "
                f"Prec {test_thr_m['precision']:.4f} | Rec {test_thr_m['recall']:.4f} | "
                f"Spec {test_thr_m['specificity']:.4f} | BalAcc {test_thr_m['balanced_accuracy']:.4f} | "
                f"MCC {test_thr_m['mcc']:.4f}"
            )

        logger.log('Measuring inference latency...')
        sample_batch = next(iter(test_loader))
        sample_x = sample_batch['x'].to(device)
        sample_alpha = build_alpha_target(sample_batch, device)
        sample_hl = sample_batch['history_length'].to(device)

        model.eval()

        def forward_fn():
            model(sample_x, sample_alpha, history_length=sample_hl)

        tracker.measure_inference_latency(
            forward_fn=forward_fn,
            batch_size=sample_x.shape[0],
            n_iterations=1000,
            warmup_steps=20,
        )
        logger.log(
            f"Inference latency: {tracker.metrics['inference_latency_ms_per_sample']:.4f} ms/sample | "
            f"Throughput: {tracker.metrics['inference_throughput_samples_per_sec']:.1f} samples/sec"
        )

        total_wall_clock = time.time() - total_start_time
        tracker.metrics['total_wall_clock_sec'] = total_wall_clock
        tracker.metrics['training_time_sec'] = total_wall_clock
        if not torch.cuda.is_available():
            tracker.metrics['gpu_memory_mb_peak'] = 0.0
            tracker.metrics['peak_gpu_memory_mb'] = 0.0
        actual_epochs = args.pretrain_epochs + min(finetune_epochs_completed, args.finetune_epochs)
        avg_epoch_time = total_wall_clock / actual_epochs if actual_epochs > 0 else 0.0
        tracker.set_epochs(num_epochs_actual=actual_epochs, average_epoch_time_sec=avg_epoch_time)
        tracker.metrics['pretrain_epoch_times_sec'] = pretrain_epoch_times
        tracker.metrics['finetune_epoch_times_sec'] = finetune_epoch_times

        np.savez(
            paths['pred_path'],
            test_y_true=test_y_true,
            test_y_pred=test_y_pred,
            val_y_true=best_val_y_true,
            val_y_pred=best_val_y_pred,
            optimal_threshold=optimal_threshold,
            seed=seed,
            best_epoch=ckpt['epoch'],
            best_val_auprc=float(ckpt['best_val_auprc']),
            test_f1=test_thr_m['f1'],
            test_precision=test_thr_m['precision'],
            test_recall=test_thr_m['recall'],
            test_specificity=test_thr_m['specificity'],
            test_balanced_accuracy=test_thr_m['balanced_accuracy'],
            test_mcc=test_thr_m['mcc'],
            focal_gamma=run_config['focal_gamma'],
            oversample_ratio=run_config['oversample_ratio'],
        )
        logger.log(f"Saved to {paths['pred_path']}")

        is_valid, missing = tracker.validate_schema()
        tracker.save_json(paths['efficiency_path'])
        logger.log(f"Efficiency saved to {paths['efficiency_path']}")
        if not is_valid:
            logger.log(f"⚠️ Schema validation failed, missing: {json.dumps(missing, ensure_ascii=False)}")

        logger.log('=' * 64)
        logger.log('实验完成: Causal-TabDiff Layer2 Parameter Discussion (Official Protocol)')
        logger.log(f'总耗时: {total_wall_clock:.1f}s ({total_wall_clock / 60:.1f} min)')
        logger.log(f"最佳 Epoch: {ckpt['epoch']} | BestValAUPRC: {best_val_auprc:.4f}")
        logger.log(f"Test AUPRC: {test_m['auprc']:.4f} | Test AUROC: {test_m['auroc']:.4f}")
        logger.log(f"Optimal Threshold: {optimal_threshold:.4f} | Test F1: {test_thr_m['f1']:.4f}")
        logger.log(
            f"参数量: {tracker.metrics.get('num_parameters_total', 'N/A')} (total) / "
            f"{tracker.metrics.get('num_parameters_trainable', 'N/A')} (trainable)"
        )
        gpu_peak = tracker.metrics.get('gpu_memory_mb_peak', 0)
        logger.log(f'GPU 显存峰值: {gpu_peak:.0f} MB')
        logger.log(f'输出文件: {paths["pred_path"]}')
        logger.log(f'效率文件: {paths["efficiency_path"]}')
        logger.log('=' * 64)

    return {
        'param_name': param_name,
        'param_value': param_value,
        'seed': seed,
        'npz_path': paths['pred_path'],
        'efficiency_path': paths['efficiency_path'],
        'log_path': os.path.join(paths['log_dir'], f'{paths["base_name"]}.log'),
        'best_epoch': int(ckpt['epoch']),
        'best_val_auprc': float(ckpt['best_val_auprc']),
        'test_auprc': float(test_m['auprc']) if not np.isnan(test_m['auprc']) else 0.0,
        'test_auroc': float(test_m['auroc']) if not np.isnan(test_m['auroc']) else 0.5,
        'test_f1': float(test_thr_m['f1']),
        'test_precision': float(test_thr_m['precision']),
        'test_recall': float(test_thr_m['recall']),
        'test_specificity': float(test_thr_m['specificity']),
        'test_balanced_accuracy': float(test_thr_m['balanced_accuracy']),
        'test_mcc': float(test_thr_m['mcc']),
        'training_time_sec': float(total_wall_clock),
    }


def summarize_results(results, seeds):
    summary = {}
    for item in results:
        param_name = item['param_name']
        value_key = normalize_value_for_summary(item['param_value'])
        seed_key = f'seed{item["seed"]}'
        metrics = {
            'test_auprc': item['test_auprc'],
            'test_auroc': item['test_auroc'],
            'test_f1': item['test_f1'],
            'test_mcc': item['test_mcc'],
            'training_time_sec': item['training_time_sec'],
            'best_epoch': item['best_epoch'],
            'best_val_auprc': item['best_val_auprc'],
            'npz_path': item['npz_path'],
            'efficiency_path': item['efficiency_path'],
        }
        summary.setdefault(param_name, {})
        if len(seeds) == 1:
            summary[param_name][value_key] = metrics
        else:
            summary[param_name].setdefault(value_key, {})
            summary[param_name][value_key][seed_key] = metrics

    if len(seeds) > 1:
        for param_name, value_dict in summary.items():
            for value_key, seed_dict in value_dict.items():
                seed_metrics = [seed_dict[f'seed{seed}'] for seed in seeds if f'seed{seed}' in seed_dict]
                if seed_metrics:
                    seed_dict['mean'] = {
                        'test_auprc': float(np.mean([m['test_auprc'] for m in seed_metrics])),
                        'test_auroc': float(np.mean([m['test_auroc'] for m in seed_metrics])),
                        'test_f1': float(np.mean([m['test_f1'] for m in seed_metrics])),
                        'test_mcc': float(np.mean([m['test_mcc'] for m in seed_metrics])),
                        'training_time_sec': float(np.mean([m['training_time_sec'] for m in seed_metrics])),
                    }
    return summary


def main():
    global OUTPUT_ROOT, LOG_ROOT, CHECKPOINT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument('--table_path', type=str, default=DEFAULT_TABLE_PATH)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--param', type=str, default=None)
    parser.add_argument('--values', type=str, default=None)
    parser.add_argument('--skip_default', action='store_true')
    parser.add_argument('--use_cached_dataset', action='store_true', default=True)
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--log_root', type=str, default=DEFAULT_LOG_ROOT)
    parser.add_argument('--checkpoint_root', type=str, default=DEFAULT_CHECKPOINT_ROOT)
    args = parser.parse_args()

    OUTPUT_ROOT = args.output_root
    LOG_ROOT = args.log_root
    CHECKPOINT_ROOT = args.checkpoint_root

    seeds = parse_seeds(args)
    selected_params = get_selected_params(args)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_results = []
    for param_name in selected_params:
        param_values = get_values_for_param(param_name, args)
        if not param_values:
            print(f'Skip {param_name}: no values to run after filtering.', flush=True)
            continue
        for seed in seeds:
            for param_value in param_values:
                run_config = make_run_config(param_name, param_value, seed)
                result = run_single_experiment(args, run_config)
                all_results.append(result)

    summary = summarize_results(all_results, seeds)
    summary_path = os.path.join(OUTPUT_ROOT, 'param_discussion_layer2_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'Summary saved to {summary_path}', flush=True)


if __name__ == '__main__':
    main()
