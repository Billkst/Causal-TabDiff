import argparse
import datetime
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

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
from src.utils.jbhi_protocol import (
    EXPECTED_FEATURE_DIM,
    JBHI_PROTOCOL,
    LEGACY_PROTOCOL,
    SMOKING_TREATMENT_COLUMN,
    activate_jbhi_smoking_condition,
    critical_source_fingerprint,
    jbhi_run_paths,
    prepare_jbhi_run_directory,
    sha256_file,
    source_revision,
    write_manifest,
)
from src.evaluation.metrics import (
    compute_ranking_metrics,
    find_optimal_threshold,
    compute_threshold_metrics,
)
from src.evaluation.efficiency import EfficiencyTracker

DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'


def seed_everything(seed):
    """尽可能固定训练、采样和 DataLoader 的随机性，并记录可复现边界。"""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False


def build_alpha_target(batch, device, require_real_condition=False):
    if 'alpha_target' in batch:
        return batch['alpha_target'].float().to(device)
    if require_real_condition:
        raise KeyError(
            'JBHI 吸烟条件协议要求 batch 含 alpha_target；'
            '拒绝回退到 landmark。'
        )
    if 'landmark' in batch:
        return batch['landmark'].float().to(device)
    raise KeyError("Batch must contain either 'alpha_target' or 'landmark'.")


def validate_jbhi_smoking_batch(batch):
    """在训练前阻断错误条件列、15D 输入和非二元 alpha。"""
    if 'alpha_target' not in batch:
        raise RuntimeError('JBHI 协议缺少 alpha_target，不能继续训练。')
    feature_dim = int(batch['x'].shape[-1])
    if feature_dim != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            f'JBHI 协议要求 {EXPECTED_FEATURE_DIM} 维输入，实际得到 {feature_dim} 维。'
        )
    alpha = batch['alpha_target'].detach().reshape(-1)
    if not torch.isfinite(alpha).all():
        raise RuntimeError('alpha_target 含非有限值，不能继续训练。')
    allowed = torch.logical_or(torch.isclose(alpha, torch.zeros_like(alpha)), torch.isclose(alpha, torch.ones_like(alpha)))
    if not bool(allowed.all()):
        values = sorted({float(v) for v in alpha.detach().cpu().tolist()})
        raise RuntimeError(f'JBHI 协议要求二元 alpha=0/1，实际取值为 {values[:10]}。')
    return sorted({float(v) for v in alpha.detach().cpu().tolist()})


def summarize_binary_condition(dataframe, split_name):
    """全表检查吸烟条件，阻止单个随机 batch 掩盖数据问题。"""
    if SMOKING_TREATMENT_COLUMN not in dataframe.columns:
        raise RuntimeError(f'{split_name} split 缺少 {SMOKING_TREATMENT_COLUMN}。')
    condition = dataframe[SMOKING_TREATMENT_COLUMN]
    if condition.isna().any():
        raise RuntimeError(f'{split_name} split 的 {SMOKING_TREATMENT_COLUMN} 含缺失值。')
    values = condition.astype(float)
    invalid = values[~values.isin((0.0, 1.0))]
    if not invalid.empty:
        unique_invalid = sorted({float(value) for value in invalid.unique()})
        raise RuntimeError(
            f'{split_name} split 的 {SMOKING_TREATMENT_COLUMN} 不是二元 0/1：'
            f'{unique_invalid[:10]}。'
        )
    counts = values.value_counts().to_dict()
    if 0.0 not in counts or 1.0 not in counts:
        raise RuntimeError(
            f'{split_name} split 未同时包含 alpha=0 和 alpha=1，拒绝正式条件化运行。'
        )
    return {
        'n': int(len(values)),
        'counts': {'0': int(counts[0.0]), '1': int(counts[1.0])},
        'positive_rate': float(values.mean()),
    }


def runtime_metadata(device):
    """写入能够解释服务器运行差异的最小运行时信息。"""
    return {
        'python_executable': sys.executable,
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cuda_available': bool(torch.cuda.is_available()),
        'device': str(device),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'determinism': {
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
            'deterministic_algorithms_warn_only': getattr(
                torch, 'is_deterministic_algorithms_warn_only_enabled', lambda: None
            )(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cuda_matmul_allow_tf32': (
                torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends, 'cuda') else None
            ),
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        },
    }


def input_feature_schema(dataframe, landmark_to_idx):
    """记录数据集实际喂给每个时间点的列名，避免仅凭 14D 数字猜测。"""
    dataset = LandmarkDataset(dataframe.iloc[:1], landmark_to_idx)
    return {
        'baseline_columns': list(dataset.baseline_cols),
        'temporal_columns_by_timepoint': dataset.temporal_cols,
        'feature_columns_by_timepoint': {
            timepoint: list(dataset.baseline_cols) + list(columns)
            for timepoint, columns in dataset.temporal_cols.items()
        },
    }


def atomic_torch_save(payload, path):
    """原子保存最佳 checkpoint，避免进程中断留下半个权重文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_save_npz(path, **arrays):
    """原子保存预测，保证 manifest 不会引用半写入的 npz。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def evaluate_model(model, dataloader, device, require_real_condition=False):
    model.eval()
    all_y_true, all_y_pred, all_pids = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            if require_real_condition and 'pid' not in batch:
                raise RuntimeError('JBHI 协议评估必须保留 PID 顺序，以核验配对基线。')
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device, require_real_condition)
            hl = batch['history_length'].to(device)
            outputs = model(x, alpha, history_length=hl)
            all_y_true.append(batch['y_2year'].numpy())
            all_y_pred.append(torch.sigmoid(outputs['risk_2year_logit']).cpu().numpy())
            if 'pid' in batch:
                all_pids.extend(batch['pid'])
    return (
        np.concatenate(all_y_true).flatten(),
        np.concatenate(all_y_pred).flatten(),
        np.asarray(all_pids, dtype=np.int64),
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument(
        '--preflight_only',
        action='store_true',
        help='仅执行 JBHI 数据/环境/14D 预检并写入独立预检 manifest，不训练。',
    )
    parser.add_argument('--table_path', type=str, default=DEFAULT_TABLE_PATH)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--oversample_ratio', type=float, default=5.0)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--use_cached_dataset', action='store_true')
    parser.add_argument(
        '--protocol',
        choices=(LEGACY_PROTOCOL, JBHI_PROTOCOL),
        default=LEGACY_PROTOCOL,
        help='legacy_landmark 保留旧实验；jbhi_smoking 强制 baseline_cigsmok 为外置条件。',
    )
    parser.add_argument(
        '--run_dir',
        type=str,
        default=None,
        help='仅 jbhi_smoking 使用。默认 outputs/jbhi_smoking/seed<seed>。',
    )
    args = parser.parse_args()

    if args.debug_mode:
        args.pretrain_epochs = 2
        args.finetune_epochs = 3
        args.batch_size = 16

    is_jbhi_smoking = args.protocol == JBHI_PROTOCOL
    if args.preflight_only and not is_jbhi_smoking:
        raise SystemExit('--preflight_only 只适用于 --protocol jbhi_smoking。')
    if is_jbhi_smoking:
        if args.use_cached_dataset:
            raise SystemExit(
                'JBHI 正式协议不支持 --use_cached_dataset；该缓存 collate 不保留固定三时点和 PID 顺序。'
            )
        if args.debug_mode:
            raise SystemExit(
                'JBHI 正式协议拒绝 --debug_mode；调试请使用 legacy 路线或独立实验目录。'
            )
        activate_jbhi_smoking_condition()
        if args.preflight_only and args.run_dir is None:
            raise SystemExit(
                '--preflight_only 必须提供独立的 --run_dir，避免占用正式 seed 目录。'
            )
        artifact_paths = jbhi_run_paths(args.seed, args.run_dir)
        prepare_jbhi_run_directory(artifact_paths)
    else:
        artifact_paths = None

    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger_dir = artifact_paths['log_dir'] if artifact_paths else Path('logs/training')
    logger_name = 'phase2' if artifact_paths else f'phase2_seed{args.seed}'
    with TrainingLogger(logger_dir, logger_name) as logger:
        logger.log(f'Device: {device}')
        logger.log('Phase 2: Two-stage training (pretrain diffusion → finetune classification)')
        logger.log(f'Protocol: {args.protocol}')
        if is_jbhi_smoking:
            logger.log(
                'Smoking condition: '
                f'{SMOKING_TREATMENT_COLUMN} (held out of X; expected feature_dim={EXPECTED_FEATURE_DIM})'
            )

        debug_n_persons = 100 if args.debug_mode else None
        train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
            args.table_path, args.seed, debug_n_persons
        )
        condition_by_split = None
        feature_schema = None
        if is_jbhi_smoking:
            condition_by_split = {
                'train': summarize_binary_condition(train_df, 'train'),
                'validation': summarize_binary_condition(val_df, 'validation'),
                'test': summarize_binary_condition(test_df, 'test'),
            }
            feature_schema = input_feature_schema(train_df, landmark_to_idx)
        
        dataset_cls = LandmarkDatasetCached if args.use_cached_dataset else LandmarkDataset
        logger.log(f"Using dataset: {dataset_cls.__name__}")
        
        train_dataset = dataset_cls(train_df, landmark_to_idx)
        val_dataset = dataset_cls(val_df, landmark_to_idx)
        test_dataset = dataset_cls(test_df, landmark_to_idx)

        train_labels = train_df['y_2year'].values.astype(int)
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0

        nw = args.num_workers
        unsup_loader_generator = torch.Generator().manual_seed(args.seed)
        sup_sampler_generator = torch.Generator().manual_seed(args.seed + 1)
        sup_loader_generator = torch.Generator().manual_seed(args.seed + 2)
        train_loader_unsup = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0,
            generator=unsup_loader_generator,
        )

        if n_pos > 0:
            sample_weights = np.where(train_labels == 1, args.oversample_ratio * n_neg / n_pos, 1.0)
            sampler = WeightedRandomSampler(
                torch.DoubleTensor(sample_weights), len(train_labels), replacement=True,
                generator=sup_sampler_generator,
            )
        else:
            sampler = None
        train_loader_sup = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            collate_fn=collate_fn, num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0,
            generator=sup_loader_generator,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0,
        )

        sample = next(iter(train_loader_unsup))
        t_steps = sample['x'].shape[1]
        feature_dim = sample['x'].shape[2]
        trajectory_len = sample['trajectory_target'].shape[1]
        alpha_values = None
        if is_jbhi_smoking:
            alpha_values = validate_jbhi_smoking_batch(sample)
            logger.log(f'JBHI preflight: feature_dim={feature_dim} | alpha_values={alpha_values}')
            logger.log(
                'JBHI condition distribution: '
                f'{json.dumps(condition_by_split, ensure_ascii=False, sort_keys=True)}'
            )
            logger.log(
                'JBHI feature schema: '
                f'{json.dumps(feature_schema, ensure_ascii=False, sort_keys=True)}'
            )

            manifest = {
                'status': 'started',
                'protocol': JBHI_PROTOCOL,
                'treatment_column': SMOKING_TREATMENT_COLUMN,
                'expected_feature_dim': EXPECTED_FEATURE_DIM,
                'actual_feature_dim': int(feature_dim),
                'alpha_values_in_preflight_batch': alpha_values,
                'condition_by_split': condition_by_split,
                'input_feature_schema': feature_schema,
                'seed': int(args.seed),
                'table_path': str(Path(args.table_path).resolve()),
                'table_sha256': sha256_file(args.table_path),
                'working_directory': str(Path.cwd().resolve()),
                'command': [sys.executable, *sys.argv],
                'source': source_revision(Path(__file__).resolve().parent),
                'critical_source_fingerprint': critical_source_fingerprint(
                    Path(__file__).resolve().parent
                ),
                'runtime': runtime_metadata(device),
                'training_configuration': {
                    'pretrain_epochs': int(args.pretrain_epochs),
                    'finetune_epochs': int(args.finetune_epochs),
                    'batch_size': int(args.batch_size),
                    'num_workers': int(args.num_workers),
                    'oversample_ratio': float(args.oversample_ratio),
                    'focal_gamma': float(args.focal_gamma),
                    'use_cached_dataset': bool(args.use_cached_dataset),
                    'diffusion_steps': 100,
                    'optimizer': {
                        'pretrain': {'name': 'Adam', 'lr': 1e-3, 'weight_decay': 1e-4},
                        'finetune': {
                            'name': 'Adam',
                            'risk_head_lr': 1e-3,
                            'trajectory_head_lr': 5e-4,
                            'base_model_lr': 1e-4,
                            'uncertainty_weight_lr': 1e-2,
                            'weight_decay': 1e-4,
                        },
                    },
                    'early_stopping_patience': 30,
                },
                'artifacts': {key: str(value) for key, value in artifact_paths.items()},
            }
            write_manifest(artifact_paths['manifest'], manifest)
            if args.preflight_only:
                manifest.update({
                    'status': 'preflight_passed',
                    'preflight_completed_at': datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    'log_file': str(logger.log_file),
                })
                write_manifest(artifact_paths['manifest'], manifest)
                logger.log('JBHI preflight passed; no model, checkpoint, or prediction was created.')
                return

        model = CausalTabDiffTrajectory(
            t_steps, feature_dim,
            100 if not args.debug_mode else 10,
            trajectory_len,
        ).to(device)

        tracker = EfficiencyTracker()
        tracker.set_model_size(model)
        tracker.set_gpu_info()
        tracker.set_metadata(
            model_name='CausalTabDiff',
            layer='layer2',
            seed=args.seed,
            experiment_type=args.protocol,
            command=' '.join(sys.argv)
        )
        efficiency_path = (
            str(artifact_paths['efficiency']) if artifact_paths
            else f'outputs/efficiency/CausalTabDiff_layer2_seed{args.seed}_efficiency.json'
        )
        pred_path = (
            str(artifact_paths['prediction']) if artifact_paths
            else f'predictions/landmark/phase2_seed{args.seed}.npz'
        )
        tracker.set_output_paths(output_file=pred_path, log_file=str(logger.log_file))
        if is_jbhi_smoking:
            tracker.metrics.update({
                'protocol': JBHI_PROTOCOL,
                'treatment_column': SMOKING_TREATMENT_COLUMN,
                'feature_dim': int(feature_dim),
                'table_sha256': manifest['table_sha256'],
            })

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_pos_ratio = (n_pos / len(train_df) * 100.0) if len(train_df) > 0 else 0.0
        logger.log('=' * 64)
        logger.log('实验: Causal-TabDiff Phase 2 Two-Stage Training')
        logger.log(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ({device})")
        logger.log(f'Seed: {args.seed}')
        logger.log(
            f'配置: pretrain_epochs={args.pretrain_epochs}, finetune_epochs={args.finetune_epochs}, '
            f'batch_size={args.batch_size}'
        )
        logger.log(f'      oversample_ratio={args.oversample_ratio}, focal_gamma={args.focal_gamma}')
        logger.log(f'数据: {args.table_path}')
        logger.log(
            f'      Train: {len(train_df)} samples, Val: {len(val_df)} samples, '
            f'Test: {len(test_df)} samples'
        )
        logger.log(f'      Pos ratio: train={train_pos_ratio:.1f}%')
        logger.log(
            f'模型: CausalTabDiffTrajectory (total_params={total_params}, '
            f'trainable={trainable_params})'
        )
        logger.log(f'输出: {pred_path}')
        logger.log(f'效率: {efficiency_path}')
        logger.log(f'日志: {logger.log_file}')
        logger.log(f"命令: {' '.join(sys.argv)}")
        logger.log('=' * 64)

        checkpoint_path = (
            str(artifact_paths['checkpoint']) if artifact_paths
            else f'checkpoints/landmark/phase2_best_seed{args.seed}.pt'
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)

        total_start_time = time.time()
        pretrain_epoch_times = []
        finetune_epoch_times = []
        finetune_epochs_completed = 0

        with tracker.track_training():
            # ====== Phase A: Pretrain diffusion backbone ======
            logger.log(f'\n=== Phase A: Pretrain diffusion backbone ({args.pretrain_epochs} epochs) ===')
            for p in model.risk_head.parameters():
                p.requires_grad = False
            for p in model.trajectory_head.parameters():
                p.requires_grad = False

            pretrain_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=1e-4,
            )

            with tracker.track_phase_time('pretrain'):
                for epoch in range(args.pretrain_epochs):
                    epoch_start = time.time()
                    model.train()
                    total_loss = 0.0
                    pbar = tqdm(train_loader_unsup, desc=f'Pretrain {epoch+1}/{args.pretrain_epochs}', ncols=80, file=sys.stderr)
                    last_heartbeat = time.time()
                    for i, batch in enumerate(pbar):
                        x = batch['x'].to(device)
                        alpha = build_alpha_target(batch, device, is_jbhi_smoking)
                        hl = batch['history_length'].to(device)
                        pretrain_opt.zero_grad()
                        outputs = model(x, alpha, history_length=hl)
                        loss = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
                        loss.backward()
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.log(f"⚠️ FATAL: Loss is {loss.item()} at pretrain epoch {epoch+1}. Terminating.")
                            sys.exit(1)
                        pretrain_opt.step()
                        total_loss += loss.item()
                        pbar.set_postfix(loss=f'{loss.item():.4f}')
                        if time.time() - last_heartbeat > 60:
                            logger.log(
                                f"  [Heartbeat] Pretrain Epoch {epoch+1} | Batch {i+1}/{len(train_loader_unsup)} | "
                                f"Loss {loss.item():.4f} | Elapsed {time.time()-epoch_start:.1f}s"
                            )
                            last_heartbeat = time.time()

                    avg_loss = total_loss / len(train_loader_unsup)
                    et = time.time() - epoch_start
                    pretrain_epoch_times.append(et)
                    logger.log(
                        f'Pretrain Epoch {epoch+1}/{args.pretrain_epochs} | Seed {args.seed} | '
                        f'LR {pretrain_opt.param_groups[0]["lr"]:.1e} | TrainLoss {avg_loss:.4f} | '
                        f'ValLoss N/A | BestValAUPRC N/A | Time {et:.1f}s'
                    )

            for p in model.parameters():
                p.requires_grad = True

            # ====== Phase B: Finetune with uncertainty-weighted multi-task loss ======
            logger.log(f'\n=== Phase B: Finetune with UW loss ({args.finetune_epochs} epochs) ===')

            uw_loss = UncertaintyWeightedLoss(n_tasks=3).to(device)
            uw_loss.log_sigma.data[2] = -1.0

            finetune_opt = torch.optim.Adam([
                {'params': model.risk_head.parameters(), 'lr': 1e-3},
                {'params': model.trajectory_head.parameters(), 'lr': 5e-4},
                {'params': model.base_model.parameters(), 'lr': 1e-4},
                {'params': uw_loss.parameters(), 'lr': 1e-2},
            ], weight_decay=1e-4)

            pos_weight = torch.tensor([pos_weight_val], device=device)
            best_val_auprc = -1.0
            patience = 30
            patience_counter = 0
            best_val_y_true = np.array([])
            best_val_y_pred = np.array([])
            best_val_pids = np.array([], dtype=np.int64)

            with tracker.track_phase_time('finetune'):
                for epoch in range(args.finetune_epochs):
                    finetune_epochs_completed = epoch + 1
                    epoch_start = time.time()
                    model.train()
                    total_loss = 0.0

                    pbar = tqdm(train_loader_sup, desc=f'Finetune {epoch+1}/{args.finetune_epochs}', ncols=80, file=sys.stderr)
                    last_heartbeat = time.time()
                    for i, batch in enumerate(pbar):
                        x = batch['x'].to(device)
                        alpha = build_alpha_target(batch, device, is_jbhi_smoking)
                        y_2year = batch['y_2year'].to(device)
                        traj_target = batch['trajectory_target'].to(device)
                        traj_mask = batch['trajectory_valid_mask'].to(device)
                        hl = batch['history_length'].to(device)

                        finetune_opt.zero_grad()
                        outputs = model(x, alpha, history_length=hl)

                        loss_gen = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
                        loss_traj = model.compute_trajectory_loss(outputs['trajectory_logits'], traj_target, traj_mask)
                        loss_cls = focal_loss_with_logits(
                            outputs['risk_2year_logit'], y_2year,
                            gamma=args.focal_gamma, pos_weight=pos_weight
                        )

                        loss = uw_loss([loss_gen, loss_traj, loss_cls])
                        loss.backward()
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.log(f"⚠️ FATAL: Loss is {loss.item()} at finetune epoch {epoch+1}. Terminating.")
                            sys.exit(1)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        finetune_opt.step()
                        total_loss += loss.item()
                        pbar.set_postfix(loss=f'{loss.item():.2f}')
                        if time.time() - last_heartbeat > 60:
                            logger.log(
                                f"  [Heartbeat] Finetune Epoch {epoch+1} | Batch {i+1}/{len(train_loader_sup)} | "
                                f"Loss {loss.item():.4f} | Elapsed {time.time()-epoch_start:.1f}s"
                            )
                            last_heartbeat = time.time()

                    train_loss = total_loss / len(train_loader_sup)
                    val_y_true, val_y_pred, val_pids = evaluate_model(
                        model, val_loader, device, require_real_condition=is_jbhi_smoking
                    )

                    metrics = compute_ranking_metrics(val_y_true, val_y_pred)
                    val_auprc = metrics['auprc'] if not np.isnan(metrics['auprc']) else 0.0
                    val_auroc = metrics['auroc'] if not np.isnan(metrics['auroc']) else 0.5

                    if val_auprc > best_val_auprc:
                        old_best = best_val_auprc
                        best_val_auprc = val_auprc
                        patience_counter = 0
                        best_val_y_true = val_y_true
                        best_val_y_pred = val_y_pred
                        best_val_pids = val_pids
                        checkpoint_payload = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'best_val_auprc': best_val_auprc,
                            'val_auroc': val_auroc,
                            'seed': args.seed,
                            'protocol': args.protocol,
                            'treatment_column': (
                                SMOKING_TREATMENT_COLUMN if is_jbhi_smoking else None
                            ),
                            'feature_dim': int(feature_dim),
                            'table_sha256': manifest['table_sha256'] if is_jbhi_smoking else None,
                            'diffusion_steps': 100 if not args.debug_mode else 10,
                            'command': [sys.executable, *sys.argv],
                            'training_configuration': (
                                manifest['training_configuration'] if is_jbhi_smoking else None
                            ),
                            'runtime': manifest['runtime'] if is_jbhi_smoking else None,
                            'critical_source_fingerprint': (
                                manifest['critical_source_fingerprint'] if is_jbhi_smoking else None
                            ),
                        }
                        atomic_torch_save(checkpoint_payload, checkpoint_path)
                        logger.log(
                            f"  ★ New Best ValAUPRC: {old_best:.4f} → {best_val_auprc:.4f} "
                            f"(epoch {epoch+1}) | Checkpoint saved"
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.log(f"  Early stopping at epoch {epoch + 1}")
                            break

                    epoch_time = time.time() - epoch_start
                    finetune_epoch_times.append(epoch_time)
                    w = uw_loss.get_weights()
                    logger.epoch_log(
                        epoch=epoch + 1, total_epochs=args.finetune_epochs,
                        seed=args.seed, lr=finetune_opt.param_groups[0]['lr'],
                        train_loss=train_loss,
                        best_val_metric=best_val_auprc, epoch_time=epoch_time,
                        ValAUPRC=val_auprc, ValAUROC=val_auroc,
                    )

                    if (epoch + 1) % 10 == 0:
                        gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                        total_elapsed = time.time() - total_start_time
                        logger.log(f"  Epoch {epoch+1}/{args.finetune_epochs} | FULL METRICS")
                        logger.log(f"    ValAUPRC {val_auprc:.4f} | ValAUROC {val_auroc:.4f}")
                        logger.log(
                            f"    BestValAUPRC {best_val_auprc:.4f} | "
                            f"EarlyStopping patience {patience_counter}/{patience}"
                        )
                        logger.log(f"    UW weights: gen={w[0]:.2f} traj={w[1]:.2f} cls={w[2]:.2f}")
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
                            f"    GPU Memory: {gpu_mem:.0f} MB peak | Epoch Time: {epoch_time:.1f}s | "
                            f"Total Elapsed: {total_elapsed:.1f}s"
                        )

        # ====== Test evaluation ======
        with tracker.track_phase_time('evaluation'):
            logger.log('\nEvaluating on test set ...')
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                logger.log(f"Loaded best model from epoch {ckpt['epoch']}")
            else:
                ckpt = {'epoch': args.finetune_epochs, 'best_val_auprc': best_val_auprc}

            # Compute optimal threshold on validation set
            if len(np.unique(best_val_y_true)) >= 2:
                optimal_threshold, _ = find_optimal_threshold(
                    best_val_y_true, best_val_y_pred, metric='f1'
                )
            else:
                optimal_threshold = 0.5
            logger.log(f"Optimal threshold (val F1): {optimal_threshold:.4f}")

            test_y_true, test_y_pred, test_pids = evaluate_model(
                model, test_loader, device, require_real_condition=is_jbhi_smoking
            )
            test_m = compute_ranking_metrics(test_y_true, test_y_pred)
            logger.log(f"Test | AUPRC {test_m['auprc']:.4f} | AUROC {test_m['auroc']:.4f}")

            # Apply fixed threshold to test set
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
        sample_alpha = build_alpha_target(sample_batch, device, is_jbhi_smoking)
        sample_hl = sample_batch['history_length'].to(device)

        model.eval()

        def forward_fn():
            model(sample_x, sample_alpha, history_length=sample_hl)

        n_iters = 1000 if not args.debug_mode else 50
        tracker.measure_inference_latency(
            forward_fn=forward_fn,
            batch_size=sample_x.shape[0],
            n_iterations=n_iters,
            warmup_steps=20 if not args.debug_mode else 5
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
        avg_epoch_time = total_wall_clock / actual_epochs if actual_epochs > 0 else 0
        tracker.set_epochs(num_epochs_actual=actual_epochs, average_epoch_time_sec=avg_epoch_time)

        atomic_save_npz(
            pred_path,
            val_y_true=best_val_y_true, val_y_pred=best_val_y_pred,
            val_pids=best_val_pids,
            test_y_true=test_y_true, test_y_pred=test_y_pred,
            test_pids=test_pids,
            optimal_threshold=optimal_threshold,
            test_f1=test_thr_m['f1'],
            test_precision=test_thr_m['precision'],
            test_recall=test_thr_m['recall'],
            test_specificity=test_thr_m['specificity'],
            test_balanced_accuracy=test_thr_m['balanced_accuracy'],
            test_mcc=test_thr_m['mcc'],
            seed=args.seed, best_epoch=ckpt['epoch'],
            best_val_auprc=float(ckpt['best_val_auprc']),
            oversample_ratio=args.oversample_ratio,
            focal_gamma=args.focal_gamma,
            protocol=args.protocol,
            treatment_column=(SMOKING_TREATMENT_COLUMN if is_jbhi_smoking else ''),
            feature_dim=int(feature_dim),
            table_sha256=(manifest['table_sha256'] if is_jbhi_smoking else ''),
        )
        logger.log(f"Saved to {pred_path}")

        os.makedirs(os.path.dirname(efficiency_path), exist_ok=True)
        is_valid, missing = tracker.validate_schema()
        tracker.save_json(efficiency_path)
        logger.log(f"Efficiency saved to {efficiency_path}")
        if not is_valid:
            logger.log(f"⚠️ Schema validation failed, missing: {json.dumps(missing, ensure_ascii=False)}")

        if is_jbhi_smoking:
            manifest.update({
                'status': 'completed',
                'checkpoint': checkpoint_path,
                'checkpoint_sha256': sha256_file(checkpoint_path),
                'prediction': pred_path,
                'prediction_sha256': sha256_file(pred_path),
                'efficiency': efficiency_path,
                'efficiency_sha256': sha256_file(efficiency_path),
                'log_file': str(logger.log_file),
                'best_epoch': int(ckpt['epoch']),
                'best_val_auprc': float(ckpt['best_val_auprc']),
                'test_metrics': {
                    'auprc': float(test_m['auprc']),
                    'auroc': float(test_m['auroc']),
                    'f1': float(test_thr_m['f1']),
                    'mcc': float(test_thr_m['mcc']),
                    'threshold': float(optimal_threshold),
                },
            })
            write_manifest(artifact_paths['manifest'], manifest)

        logger.log('=' * 64)
        logger.log('实验完成: Causal-TabDiff Phase 2 Two-Stage Training')
        logger.log(f'总耗时: {total_wall_clock:.1f}s ({total_wall_clock/60:.1f} min)')
        logger.log(f"最佳 Epoch: {ckpt['epoch']} | BestValAUPRC: {best_val_auprc:.4f}")
        logger.log(f"Test AUPRC: {test_m['auprc']:.4f} | Test AUROC: {test_m['auroc']:.4f}")
        logger.log(f"Optimal Threshold: {optimal_threshold:.4f} | Test F1: {test_thr_m['f1']:.4f}")
        logger.log(
            f"参数量: {tracker.metrics.get('num_parameters_total', 'N/A')} (total) / "
            f"{tracker.metrics.get('num_parameters_trainable', 'N/A')} (trainable)"
        )
        gpu_peak = tracker.metrics.get('gpu_memory_mb_peak', 0)
        logger.log(f'GPU 显存峰值: {gpu_peak:.0f} MB')
        logger.log(f'输出文件: {pred_path}')
        logger.log(f'效率文件: {efficiency_path}')
        logger.log('=' * 64)


if __name__ == '__main__':
    main()
