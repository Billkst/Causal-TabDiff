import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.data_module_landmark import load_and_split_data, LandmarkDataset, collate_fn
from src.data.data_module_landmark_cached import LandmarkDatasetCached
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.utils.training_logger import TrainingLogger
from src.evaluation.metrics import (
    compute_ranking_metrics,
    find_optimal_threshold,
    compute_threshold_metrics,
)
from src.evaluation.efficiency import EfficiencyTracker


DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'
DEFAULT_OUTPUT_ROOT = 'outputs/ablation_layer2'
DEFAULT_LOG_DIR = 'logs/training'

ABLATION_CONFIGS = {
    'Full_Model': {
        'disc_weight': 0.5,
        'focal_gamma': 2.0,
        'oversample_ratio': 5.0,
        'use_oversampling': True,
        'use_uw': True,
        'use_time_attn': True,
        'use_feat_attn': True,
        'traj_weight_enabled': True,
    },
    'wo_Discriminator': {
        'disc_weight': 0.0,
        'focal_gamma': 2.0,
        'oversample_ratio': 5.0,
        'use_oversampling': True,
        'use_uw': True,
        'use_time_attn': True,
        'use_feat_attn': True,
        'traj_weight_enabled': True,
    },
    'wo_Trajectory_Loss': {
        'disc_weight': 0.5,
        'focal_gamma': 2.0,
        'oversample_ratio': 5.0,
        'use_oversampling': True,
        'use_uw': True,
        'use_time_attn': True,
        'use_feat_attn': True,
        'traj_weight_enabled': False,
    },
    'wo_Dual_Attention': {
        'disc_weight': 0.5,
        'focal_gamma': 2.0,
        'oversample_ratio': 5.0,
        'use_oversampling': True,
        'use_uw': True,
        'use_time_attn': False,
        'use_feat_attn': False,
        'traj_weight_enabled': True,
    },
    'wo_Focal_Loss': {
        'disc_weight': 0.5,
        'focal_gamma': 0.0,
        'oversample_ratio': 5.0,
        'use_oversampling': True,
        'use_uw': True,
        'use_time_attn': True,
        'use_feat_attn': True,
        'traj_weight_enabled': True,
    },
    'wo_Oversampling': {
        'disc_weight': 0.5,
        'focal_gamma': 2.0,
        'oversample_ratio': 1.0,
        'use_oversampling': False,
        'use_uw': True,
        'use_time_attn': True,
        'use_feat_attn': True,
        'traj_weight_enabled': True,
    },
    'wo_UW_Loss': {
        'disc_weight': 0.5,
        'focal_gamma': 2.0,
        'oversample_ratio': 5.0,
        'use_oversampling': True,
        'use_uw': False,
        'use_time_attn': True,
        'use_feat_attn': True,
        'traj_weight_enabled': True,
    },
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
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
            history_length = batch['history_length'].to(device)
            outputs = model(x, alpha, history_length=history_length)
            all_y_true.append(batch['y_2year'].cpu().numpy())
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


T = TypeVar('T')


def parse_csv_arg(text: str, cast_fn: Callable[[str], T] = str):
    return [cast_fn(item.strip()) for item in text.split(',') if item.strip()]


def resolve_seeds(args):
    if args.seed is not None and args.seeds is not None:
        raise ValueError('--seed 与 --seeds 不能同时使用。')
    if args.seeds is not None:
        return parse_csv_arg(args.seeds, int)
    if args.seed is not None:
        return [args.seed]
    return [42]


def resolve_variants(args):
    if args.variant and args.variants:
        raise ValueError('--variant 与 --variants 不能同时使用。')

    variant_names = list(ABLATION_CONFIGS.keys())
    if args.skip_full_model:
        variant_names = [name for name in variant_names if name != 'Full_Model']

    if args.variant:
        variant_names = [args.variant]
    elif args.variants:
        variant_names = parse_csv_arg(args.variants, str)

    invalid = [name for name in variant_names if name not in ABLATION_CONFIGS]
    if invalid:
        raise ValueError(f'未知 variant: {invalid}. 可选: {list(ABLATION_CONFIGS.keys())}')
    return variant_names


def create_output_paths(output_root, variant_name, seed):
    variant_dir = Path(output_root) / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    pred_path = variant_dir / f'{variant_name}_seed{seed}.npz'
    efficiency_path = variant_dir / f'{variant_name}_seed{seed}_efficiency.json'
    checkpoint_path = variant_dir / f'{variant_name}_seed{seed}_best.pt'
    return variant_dir, pred_path, efficiency_path, checkpoint_path


def build_supervised_loader(train_dataset, train_labels, batch_size, num_workers, config):
    n_pos = int(train_labels.sum())
    n_neg = int(len(train_labels) - n_pos)

    if config['use_oversampling'] and n_pos > 0:
        sample_weights = np.where(
            train_labels == 1,
            config['oversample_ratio'] * n_neg / n_pos,
            1.0,
        )
        sampler = WeightedRandomSampler(
            sample_weights.tolist(),
            len(train_labels),
            replacement=True,
        )
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def format_uw_weights(config, uw_loss):
    if not config['use_uw']:
        return 'Simple sum (no UW)'
    weights = uw_loss.get_weights()
    if config['traj_weight_enabled']:
        return f'UW weights: gen={weights[0]:.2f} traj={weights[1]:.2f} cls={weights[2]:.2f}'
    return f'UW weights: gen={weights[0]:.2f} cls={weights[1]:.2f}'


def run_single_variant(args, variant_name, seed):
    set_seed(seed)

    config = dict(ABLATION_CONFIGS[variant_name])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    debug_n_persons = 100 if args.debug_mode else None
    variant_dir, pred_path, efficiency_path, checkpoint_path = create_output_paths(
        args.output_root,
        variant_name,
        seed,
    )

    log_name = f'ablation_layer2_{variant_name}_seed{seed}'
    with TrainingLogger(args.log_dir, log_name) as logger:
        logger.log('=' * 72)
        logger.log('实验: Causal-TabDiff Layer2 Official-Compatible Ablation')
        logger.log(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f'Variant: {variant_name}')
        logger.log(f'Seed: {seed}')
        logger.log(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ({device})")
        logger.log(
            f'配置: pretrain_epochs={args.pretrain_epochs}, finetune_epochs={args.finetune_epochs}, '
            f'batch_size={args.batch_size}, num_workers={args.num_workers}, use_cached_dataset={args.use_cached_dataset}'
        )
        logger.log(
            '变体设置: '
            f"disc_weight={config['disc_weight']}, focal_gamma={config['focal_gamma']}, "
            f"oversample_ratio={config['oversample_ratio']}, use_oversampling={config['use_oversampling']}, "
            f"use_uw={config['use_uw']}, use_time_attn={config['use_time_attn']}, "
            f"use_feat_attn={config['use_feat_attn']}, traj_weight_enabled={config['traj_weight_enabled']}"
        )
        logger.log(f'数据: {args.table_path}')
        logger.log(f'输出: {pred_path}')
        logger.log(f'效率: {efficiency_path}')
        logger.log(f'日志: {logger.log_file}')
        logger.log(f"命令: {' '.join(sys.argv)}")
        logger.log('=' * 72)

        train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
            args.table_path,
            seed,
            debug_n_persons,
        )
        dataset_cls = LandmarkDatasetCached if args.use_cached_dataset else LandmarkDataset
        logger.log(f'Using dataset: {dataset_cls.__name__}')

        train_dataset = dataset_cls(train_df, landmark_to_idx)
        val_dataset = dataset_cls(val_df, landmark_to_idx)
        test_dataset = dataset_cls(test_df, landmark_to_idx)

        train_labels = np.asarray(train_df['y_2year']).astype(int)
        n_pos = int(train_labels.sum())
        n_neg = int(len(train_labels) - n_pos)
        pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
        train_pos_ratio = (n_pos / len(train_df) * 100.0) if len(train_df) > 0 else 0.0
        logger.log(
            f'数据拆分: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} | '
            f'TrainPosRatio={train_pos_ratio:.1f}%'
        )

        train_loader_unsup = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        train_loader_sup = build_supervised_loader(
            train_dataset,
            train_labels,
            args.batch_size,
            args.num_workers,
            config,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        sample = next(iter(train_loader_unsup))
        t_steps = sample['x'].shape[1]
        feature_dim = sample['x'].shape[2]
        trajectory_len = sample['trajectory_target'].shape[1]

        model = CausalTabDiffTrajectory(
            t_steps=t_steps,
            feature_dim=feature_dim,
            diffusion_steps=100 if not args.debug_mode else 10,
            trajectory_len=trajectory_len,
            use_time_attn=bool(config['use_time_attn']),
            use_feat_attn=bool(config['use_feat_attn']),
        ).to(device)

        tracker = EfficiencyTracker()
        tracker.set_model_size(model)
        tracker.set_gpu_info()
        tracker.set_metadata(
            model_name='CausalTabDiff',
            layer='layer2',
            seed=seed,
            experiment_type=f'ablation_layer2_official::{variant_name}',
            command=' '.join(sys.argv),
        )
        tracker.set_output_paths(output_file=str(pred_path), log_file=str(logger.log_file))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log(
            f'模型: CausalTabDiffTrajectory (total_params={total_params}, trainable={trainable_params}, '
            f't_steps={t_steps}, feature_dim={feature_dim}, trajectory_len={trajectory_len})'
        )

        total_start_time = time.time()
        finetune_epochs_completed = 0
        best_val_auprc = -1.0
        best_val_y_true = np.array([])
        best_val_y_pred = np.array([])

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
                        history_length = batch['history_length'].to(device)

                        pretrain_opt.zero_grad()
                        outputs = model(x, alpha, history_length=history_length)
                        loss = outputs['diff_loss'] + config['disc_weight'] * outputs['disc_loss']
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
                    logger.epoch_log(
                        epoch=epoch + 1,
                        total_epochs=args.pretrain_epochs,
                        seed=seed,
                        lr=pretrain_opt.param_groups[0]['lr'],
                        train_loss=avg_loss,
                        epoch_time=epoch_time,
                    )

            for p in model.parameters():
                p.requires_grad = True

            logger.log(f'\n=== Phase B: Finetune with official protocol ({args.finetune_epochs} epochs) ===')

            if config['use_uw']:
                n_tasks = 3 if config['traj_weight_enabled'] else 2
                uw_loss = UncertaintyWeightedLoss(n_tasks=n_tasks).to(device)
                if config['traj_weight_enabled']:
                    uw_loss.log_sigma.data[2] = -1.0
                else:
                    uw_loss.log_sigma.data[1] = -1.0
                optimizer_groups = [
                    {'params': model.risk_head.parameters(), 'lr': 1e-3},
                    {'params': model.trajectory_head.parameters(), 'lr': 5e-4},
                    {'params': model.base_model.parameters(), 'lr': 1e-4},
                    {'params': uw_loss.parameters(), 'lr': 1e-2},
                ]
            else:
                uw_loss = None
                optimizer_groups = [
                    {'params': model.risk_head.parameters(), 'lr': 1e-3},
                    {'params': model.trajectory_head.parameters(), 'lr': 5e-4},
                    {'params': model.base_model.parameters(), 'lr': 1e-4},
                ]

            finetune_opt = torch.optim.Adam(optimizer_groups, weight_decay=1e-4)
            pos_weight = torch.tensor([pos_weight_val], device=device)
            patience = 30
            patience_counter = 0

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
                        history_length = batch['history_length'].to(device)

                        finetune_opt.zero_grad()
                        outputs = model(x, alpha, history_length=history_length)

                        loss_gen = outputs['diff_loss'] + config['disc_weight'] * outputs['disc_loss']
                        loss_cls = focal_loss_with_logits(
                            outputs['risk_2year_logit'],
                            y_2year,
                            gamma=config['focal_gamma'],
                            pos_weight=pos_weight,
                        )

                        loss_traj = None
                        if config['traj_weight_enabled']:
                            loss_traj = model.compute_trajectory_loss(
                                outputs['trajectory_logits'],
                                traj_target,
                                traj_mask,
                            )

                        if config['use_uw']:
                            assert uw_loss is not None
                            if config['traj_weight_enabled']:
                                assert loss_traj is not None
                                loss = uw_loss([loss_gen, loss_traj, loss_cls])
                            else:
                                loss = uw_loss([loss_gen, loss_cls])
                        else:
                            if config['traj_weight_enabled']:
                                assert loss_traj is not None
                                loss = loss_gen + loss_traj + loss_cls
                            else:
                                loss = loss_gen + loss_cls

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
                                'variant': variant_name,
                                'config': config,
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
                        logger.log(f'    {format_uw_weights(config, uw_loss)}')
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
            if checkpoint_path.exists():
                ckpt = torch.load(checkpoint_path, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                logger.log(f"Loaded best model from epoch {ckpt['epoch']}")
            else:
                ckpt = {'epoch': args.finetune_epochs, 'best_val_auprc': best_val_auprc}

            if len(np.unique(best_val_y_true)) >= 2:
                optimal_threshold, _ = find_optimal_threshold(
                    best_val_y_true,
                    best_val_y_pred,
                    metric='f1',
                )
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
        sample_history_length = sample_batch['history_length'].to(device)

        model.eval()

        def forward_fn():
            model(sample_x, sample_alpha, history_length=sample_history_length)

        n_iters = 1000 if not args.debug_mode else 50
        tracker.measure_inference_latency(
            forward_fn=forward_fn,
            batch_size=sample_x.shape[0],
            n_iterations=n_iters,
            warmup_steps=20 if not args.debug_mode else 5,
        )
        logger.log(
            f"Inference latency: {tracker.metrics['inference_latency_ms_per_sample']:.4f} ms/sample | "
            f"Throughput: {tracker.metrics['inference_throughput_samples_per_sec']:.1f} samples/sec"
        )

        total_wall_clock = time.time() - total_start_time
        tracker.metrics['total_wall_clock_sec'] = total_wall_clock
        if not torch.cuda.is_available():
            tracker.metrics['gpu_memory_mb_peak'] = 0.0
            tracker.metrics['peak_gpu_memory_mb'] = 0.0
        actual_epochs = args.pretrain_epochs + min(finetune_epochs_completed, args.finetune_epochs)
        avg_epoch_time = total_wall_clock / actual_epochs if actual_epochs > 0 else 0.0
        tracker.set_epochs(num_epochs_actual=actual_epochs, average_epoch_time_sec=avg_epoch_time)

        np.savez(
            pred_path,
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
            focal_gamma=config['focal_gamma'],
            oversample_ratio=config['oversample_ratio'],
        )
        logger.log(f'Saved to {pred_path}')

        is_valid, missing = tracker.validate_schema()
        tracker.save_json(str(efficiency_path))
        logger.log(f'Efficiency saved to {efficiency_path}')
        if not is_valid:
            logger.log(f"⚠️ Schema validation failed, missing: {json.dumps(missing, ensure_ascii=False)}")

        logger.log('=' * 72)
        logger.log('实验完成: Causal-TabDiff Layer2 Official-Compatible Ablation')
        logger.log(f'Variant: {variant_name} | Seed: {seed}')
        logger.log(f'总耗时: {total_wall_clock:.1f}s ({total_wall_clock / 60:.1f} min)')
        logger.log(f"最佳 Epoch: {ckpt['epoch']} | BestValAUPRC: {best_val_auprc:.4f}")
        logger.log(f"Test AUPRC: {test_m['auprc']:.4f} | Test AUROC: {test_m['auroc']:.4f}")
        logger.log(
            f"Optimal Threshold: {optimal_threshold:.4f} | Test F1: {test_thr_m['f1']:.4f} | "
            f"Precision: {test_thr_m['precision']:.4f} | Recall: {test_thr_m['recall']:.4f}"
        )
        logger.log(
            f"参数量: {tracker.metrics.get('num_parameters_total', 'N/A')} (total) / "
            f"{tracker.metrics.get('num_parameters_trainable', 'N/A')} (trainable)"
        )
        logger.log(f"GPU 显存峰值: {tracker.metrics.get('gpu_memory_mb_peak', 0):.0f} MB")
        logger.log(f'输出文件: {pred_path}')
        logger.log(f'效率文件: {efficiency_path}')
        logger.log('=' * 72)

    return {
        'seed': seed,
        'variant': variant_name,
        'output_path': str(pred_path),
        'efficiency_path': str(efficiency_path),
        'best_epoch': int(ckpt['epoch']),
        'best_val_auprc': float(ckpt['best_val_auprc']),
        'optimal_threshold': float(optimal_threshold),
        'test_auprc': float(test_m['auprc']),
        'test_auroc': float(test_m['auroc']),
        'test_f1': float(test_thr_m['f1']),
        'test_precision': float(test_thr_m['precision']),
        'test_recall': float(test_thr_m['recall']),
        'test_specificity': float(test_thr_m['specificity']),
        'test_balanced_accuracy': float(test_thr_m['balanced_accuracy']),
        'test_mcc': float(test_thr_m['mcc']),
        'focal_gamma': float(config['focal_gamma']),
        'oversample_ratio': float(config['oversample_ratio']),
    }


def save_summary(summary, output_root):
    summary_path = Path(output_root) / 'ablation_layer2_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', buffering=1, encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
        fp.flush()
    print(f'Summary saved to {summary_path}', flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--table_path', type=str, default=DEFAULT_TABLE_PATH)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--variant', type=str, default=None)
    parser.add_argument('--variants', type=str, default=None)
    parser.add_argument('--skip_full_model', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--log_dir', type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument('--use_cached_dataset', dest='use_cached_dataset', action='store_true')
    parser.add_argument('--no_cached_dataset', dest='use_cached_dataset', action='store_false')
    parser.set_defaults(use_cached_dataset=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.debug_mode:
        args.pretrain_epochs = 2
        args.finetune_epochs = 3
        args.batch_size = 16

    seeds = resolve_seeds(args)
    variants = resolve_variants(args)

    print('开始运行 official-compatible layer2 ablation。', flush=True)
    print(f'Variants: {variants}', flush=True)
    print(f'Seeds: {seeds}', flush=True)

    summary = {}
    for variant_name in variants:
        summary.setdefault(variant_name, {})
        for seed in seeds:
            result = run_single_variant(args, variant_name, seed)
            summary[variant_name][f'seed{seed}'] = result
            save_summary(summary, args.output_root)

    print('全部 ablation 运行完成。', flush=True)


if __name__ == '__main__':
    main()
