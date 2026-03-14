import argparse
import logging
import os

import torch
import torch.nn.functional as F

from src.data.data_module_landmark import get_dataloader
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory


DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'


os.makedirs('logs/training', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training/run_landmark.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_alpha_target(batch, device):
    if 'alpha_target' in batch:
        return batch['alpha_target'].float().to(device)
    if 'landmark' in batch:
        return batch['landmark'].float().to(device)
    raise KeyError("Batch must contain either 'alpha_target' or 'landmark'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--table_path', type=str, default=DEFAULT_TABLE_PATH)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.debug_mode:
        args.epochs = 2
        args.batch_size = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    debug_n_persons = 100 if args.debug_mode else None
    train_loader = get_dataloader(
        args.table_path,
        'train',
        batch_size=args.batch_size,
        seed=args.seed,
        debug_n_persons=debug_n_persons,
    )
    val_loader = get_dataloader(
        args.table_path,
        'val',
        batch_size=args.batch_size,
        seed=args.seed,
        debug_n_persons=debug_n_persons,
    )

    sample = next(iter(train_loader))
    t_steps = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    trajectory_len = sample['trajectory_target'].shape[1]

    model = CausalTabDiffTrajectory(
        t_steps,
        feature_dim,
        100 if not args.debug_mode else 10,
        trajectory_len,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info(
        'Training started | '
        f'train_batches={len(train_loader)} | val_batches={len(val_loader)} | '
        f'table_path={args.table_path}'
    )
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device)
            y_2year = batch['y_2year'].to(device)
            traj_target = batch['trajectory_target'].to(device)
            traj_mask = batch['trajectory_valid_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(x, alpha)

            loss_diff = outputs['diff_loss']
            loss_disc = outputs['disc_loss']
            loss_traj = model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask)
            loss_2year = F.binary_cross_entropy(outputs['risk_2year'], y_2year)

            loss = loss_diff + 0.5 * loss_disc + loss_traj + loss_2year
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f'Epoch {epoch + 1}/{args.epochs} | Loss: {total_loss / len(train_loader):.4f}')

    logger.info('Training complete')


if __name__ == '__main__':
    main()
