import argparse
import os
import sys
import time

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

DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--table_path', type=str, default=DEFAULT_TABLE_PATH)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--oversample_ratio', type=float, default=5.0)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--use_cached_dataset', action='store_true')
    args = parser.parse_args()

    if args.debug_mode:
        args.pretrain_epochs = 2
        args.finetune_epochs = 3
        args.batch_size = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with TrainingLogger('logs/training', f'phase2_seed{args.seed}') as logger:
        logger.log(f'Device: {device}')
        logger.log('Phase 2: Two-stage training (pretrain diffusion → finetune classification)')

        debug_n_persons = 100 if args.debug_mode else None
        train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
            args.table_path, args.seed, debug_n_persons
        )
        
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
        train_loader_unsup = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0,
        )

        if n_pos > 0:
            sample_weights = np.where(train_labels == 1, args.oversample_ratio * n_neg / n_pos, 1.0)
            sampler = WeightedRandomSampler(
                torch.DoubleTensor(sample_weights), len(train_labels), replacement=True,
            )
        else:
            sampler = None
        train_loader_sup = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            collate_fn=collate_fn, num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0,
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

        model = CausalTabDiffTrajectory(
            t_steps, feature_dim,
            100 if not args.debug_mode else 10,
            trajectory_len,
        ).to(device)

        os.makedirs('checkpoints/landmark', exist_ok=True)
        os.makedirs('predictions/landmark', exist_ok=True)

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

        for epoch in range(args.pretrain_epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader_unsup, desc=f'Pretrain {epoch+1}/{args.pretrain_epochs}', ncols=80, file=sys.stderr)
            for batch in pbar:
                x = batch['x'].to(device)
                alpha = build_alpha_target(batch, device)
                hl = batch['history_length'].to(device)
                pretrain_opt.zero_grad()
                outputs = model(x, alpha, history_length=hl)
                loss = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
                loss.backward()
                pretrain_opt.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}')

            avg_loss = total_loss / len(train_loader_unsup)
            et = time.time() - epoch_start
            logger.log(f'  Pretrain {epoch+1}/{args.pretrain_epochs} | Loss {avg_loss:.4f} | Time {et:.1f}s')

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
        checkpoint_path = f'checkpoints/landmark/phase2_best_seed{args.seed}.pt'
        best_val_y_true = np.array([])
        best_val_y_pred = np.array([])

        for epoch in range(args.finetune_epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0

            pbar = tqdm(train_loader_sup, desc=f'Finetune {epoch+1}/{args.finetune_epochs}', ncols=80, file=sys.stderr)
            for batch in pbar:
                x = batch['x'].to(device)
                alpha = build_alpha_target(batch, device)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                finetune_opt.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.2f}')

            train_loss = total_loss / len(train_loader_sup)
            val_y_true, val_y_pred = evaluate_model(model, val_loader, device)

            metrics = compute_ranking_metrics(val_y_true, val_y_pred)
            val_auprc = metrics['auprc'] if not np.isnan(metrics['auprc']) else 0.0
            val_auroc = metrics['auroc'] if not np.isnan(metrics['auroc']) else 0.5

            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                patience_counter = 0
                best_val_y_true = val_y_true
                best_val_y_pred = val_y_pred
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_auprc': best_val_auprc,
                    'val_auroc': val_auroc,
                    'seed': args.seed,
                }, checkpoint_path)
                logger.log(f"  ✓ Best | AUPRC {best_val_auprc:.4f} | AUROC {val_auroc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.log(f"  Early stopping at epoch {epoch + 1}")
                    break

            epoch_time = time.time() - epoch_start
            w = uw_loss.get_weights()
            logger.epoch_log(
                epoch=epoch + 1, total_epochs=args.finetune_epochs,
                seed=args.seed, lr=finetune_opt.param_groups[0]['lr'],
                train_loss=train_loss, val_loss=0.0,
                best_val_metric=best_val_auprc, epoch_time=epoch_time,
                ValAUPRC=val_auprc, ValAUROC=val_auroc,
            )

            if (epoch + 1) % 10 == 0:
                logger.log(f"  └─ UW weights: gen={w[0]:.2f} traj={w[1]:.2f} cls={w[2]:.2f}")
                if len(np.unique(val_y_true)) >= 2:
                    thr, _ = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
                    y_bin = (val_y_pred >= thr).astype(int)
                    tm = compute_threshold_metrics(val_y_true, y_bin)
                    logger.log(
                        f"  └─ F1 {tm['f1']:.4f} | Prec {tm['precision']:.4f} | "
                        f"Rec {tm['recall']:.4f} | Thr {thr:.3f}"
                    )

        # ====== Test evaluation ======
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

        test_y_true, test_y_pred = evaluate_model(model, test_loader, device)
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

        pred_path = f'predictions/landmark/phase2_seed{args.seed}.npz'
        np.savez(
            pred_path,
            val_y_true=best_val_y_true, val_y_pred=best_val_y_pred,
            test_y_true=test_y_true, test_y_pred=test_y_pred,
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
        )
        logger.log(f"Saved to {pred_path}")
        logger.log('Phase 2 complete')


if __name__ == '__main__':
    main()
