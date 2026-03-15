import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from src.data.data_module_landmark import get_dataloader, load_and_split_data, LandmarkDataset, collate_fn
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
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device)
            y_2year = batch['y_2year'].to(device)
            history_length = batch['history_length'].to(device)

            outputs = model(x, alpha, history_length=history_length)
            logits = outputs['risk_2year_logit']

            all_y_true.append(y_2year.cpu().numpy())
            all_y_pred.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(all_y_true).flatten()
    y_pred = np.concatenate(all_y_pred).flatten()
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--table_path', type=str, default=DEFAULT_TABLE_PATH)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.debug_mode:
        args.epochs = 3
        args.batch_size = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with TrainingLogger('logs/training', f'phase1_seed{args.seed}') as logger:
        logger.log(f'Device: {device}')
        logger.log(f'Phase 1: Direct risk head (no diffusion loss)')

        debug_n_persons = 100 if args.debug_mode else None
        train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
            args.table_path, args.seed, debug_n_persons
        )

        train_dataset = LandmarkDataset(train_df, landmark_to_idx)
        val_dataset = LandmarkDataset(val_df, landmark_to_idx)
        test_dataset = LandmarkDataset(test_df, landmark_to_idx)

        train_labels = train_df['y_2year'].values.astype(int)
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos

        if n_pos > 0:
            pos_weight_val = n_neg / n_pos
            oversample_ratio = 5.0
            sample_weights = np.where(
                train_labels == 1,
                oversample_ratio * n_neg / n_pos,
                1.0,
            )
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(train_labels),
                replacement=True,
            )
        else:
            pos_weight_val = 1.0
            sampler = None

        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )

        sample = next(iter(train_loader))
        t_steps = sample['x'].shape[1]
        feature_dim = sample['x'].shape[2]
        trajectory_len = sample['trajectory_target'].shape[1]

        model = CausalTabDiffTrajectory(
            t_steps, feature_dim,
            100 if not args.debug_mode else 10,
            trajectory_len,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        pos_weight = torch.tensor([pos_weight_val], device=device)
        logger.log(f'pos_weight={pos_weight_val:.1f} | n_pos={n_pos} | n_neg={n_neg}')

        os.makedirs('checkpoints/landmark', exist_ok=True)
        os.makedirs('predictions/landmark', exist_ok=True)
        best_val_auprc = 0.0
        patience = 30
        patience_counter = 0
        checkpoint_path = f'checkpoints/landmark/phase1_best_seed{args.seed}.pt'

        logger.log(
            f'Training | seed={args.seed} | epochs={args.epochs} | '
            f'batch_size={args.batch_size} | batches/epoch={len(train_loader)}'
        )

        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                x = batch['x'].to(device)
                alpha = build_alpha_target(batch, device)
                y_2year = batch['y_2year'].to(device)
                history_length = batch['history_length'].to(device)

                optimizer.zero_grad()
                outputs = model(x, alpha, history_length=history_length)

                logits = outputs['risk_2year_logit']
                loss = F.binary_cross_entropy_with_logits(logits, y_2year, pos_weight=pos_weight)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
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
            logger.epoch_log(
                epoch=epoch + 1, total_epochs=args.epochs,
                seed=args.seed, lr=optimizer.param_groups[0]['lr'],
                train_loss=train_loss, val_loss=0.0,
                best_val_metric=best_val_auprc, epoch_time=epoch_time,
                ValAUPRC=val_auprc, ValAUROC=val_auroc,
            )

            if (epoch + 1) % 10 == 0:
                thr, _ = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
                y_bin = (val_y_pred >= thr).astype(int)
                tm = compute_threshold_metrics(val_y_true, y_bin)
                logger.log(
                    f"  └─ F1 {tm['f1']:.4f} | Prec {tm['precision']:.4f} | "
                    f"Rec {tm['recall']:.4f} | Thr {thr:.3f}"
                )

        logger.log('\nEvaluating on test set ...')
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            logger.log(f"Loaded best model from epoch {ckpt['epoch']}")
        else:
            ckpt = {'epoch': args.epochs, 'best_val_auprc': best_val_auprc}
            best_val_y_true = val_y_true
            best_val_y_pred = val_y_pred

        test_y_true, test_y_pred = evaluate_model(model, test_loader, device)
        test_m = compute_ranking_metrics(test_y_true, test_y_pred)

        logger.log(
            f"Test | AUPRC {test_m['auprc']:.4f} | AUROC {test_m['auroc']:.4f}"
        )

        pred_path = f'predictions/landmark/phase1_seed{args.seed}.npz'
        np.savez(
            pred_path,
            val_y_true=best_val_y_true, val_y_pred=best_val_y_pred,
            test_y_true=test_y_true, test_y_pred=test_y_pred,
            seed=args.seed, best_epoch=ckpt['epoch'],
            best_val_auprc=float(ckpt['best_val_auprc']),
        )
        logger.log(f"Saved to {pred_path}")
        logger.log('Phase 1 complete')


if __name__ == '__main__':
    main()
