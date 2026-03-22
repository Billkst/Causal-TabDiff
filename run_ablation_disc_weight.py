import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from src.data.data_module_landmark import get_dataloader
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.utils.training_logger import TrainingLogger
from src.evaluation.metrics import compute_ranking_metrics

DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'
SEEDS = [42, 52, 62]
DISC_WEIGHTS = [0.0, 0.25, 0.5, 1.0]
EPOCHS = 50


def build_alpha_target(batch, device):
    if 'alpha_target' in batch:
        return batch['alpha_target'].float().to(device)
    if 'landmark' in batch:
        return batch['landmark'].float().to(device)
    raise KeyError("Batch must contain either 'alpha_target' or 'landmark'.")


def evaluate_model(model, dataloader, device, disc_weight):
    model.eval()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device)
            y_2year = batch['y_2year'].to(device)
            traj_target = batch['trajectory_target'].to(device)
            traj_mask = batch['trajectory_valid_mask'].to(device)

            outputs = model(x, alpha)
            loss_diff = outputs['diff_loss']
            loss_disc = outputs['disc_loss']
            loss_traj = model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask)
            loss_2year = F.binary_cross_entropy(outputs['risk_2year'], y_2year)
            loss = loss_diff + disc_weight * loss_disc + loss_traj + loss_2year

            total_loss += loss.item()
            all_y_true.append(y_2year.cpu().numpy())
            all_y_pred.append(outputs['risk_2year'].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    y_true = np.concatenate(all_y_true).flatten()
    y_pred = np.concatenate(all_y_pred).flatten()
    return avg_loss, y_true, y_pred


def train_single_config(seed, disc_weight, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = get_dataloader(DEFAULT_TABLE_PATH, 'train', batch_size=512, seed=seed)
    val_loader = get_dataloader(DEFAULT_TABLE_PATH, 'val', batch_size=512, seed=seed)

    sample = next(iter(train_loader))
    t_steps = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    trajectory_len = sample['trajectory_target'].shape[1]

    model = CausalTabDiffTrajectory(t_steps, feature_dim, 100, trajectory_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    best_val_auprc = 0.0

    for epoch in range(EPOCHS):
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

            loss = loss_diff + disc_weight * loss_disc + loss_traj + loss_2year
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_y_true, val_y_pred = evaluate_model(model, val_loader, device, disc_weight)

        metrics = compute_ranking_metrics(val_y_true, val_y_pred)
        val_auprc = metrics['auprc']

        if np.isnan(val_auprc):
            val_auprc = 0.0
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | TrainLoss {train_loss:.4f} | ValAUPRC {val_auprc:.4f} | Best {best_val_auprc:.4f}", flush=True)

    return best_val_auprc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('logs', exist_ok=True)

    log_file = open('logs/ablation_disc_weight.log', 'w', buffering=1)
    log_file.write(f"=== Discriminator Loss Weight Ablation ===\n")
    log_file.write(f"Seeds: {SEEDS}\n")
    log_file.write(f"Disc Weights: {DISC_WEIGHTS}\n")
    log_file.write(f"Epochs: {EPOCHS}\n")
    log_file.write(f"Device: {device}\n\n")
    log_file.flush()

    results = {}
    for disc_weight in DISC_WEIGHTS:
        results[disc_weight] = []
        for seed in SEEDS:
            msg = f"Training | DiscWeight={disc_weight} | Seed={seed}"
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

            best_auprc = train_single_config(seed, disc_weight, device)
            results[disc_weight].append(best_auprc)

            msg = f"  Result | DiscWeight={disc_weight} | Seed={seed} | BestValAUPRC={best_auprc:.4f}"
            print(msg, flush=True)
            log_file.write(msg + "\n\n")
            log_file.flush()

    log_file.write("\n=== Summary ===\n")
    for disc_weight in DISC_WEIGHTS:
        auprcs = results[disc_weight]
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)
        msg = f"DiscWeight={disc_weight} | Mean AUPRC={mean_auprc:.4f} ± {std_auprc:.4f} | Values={auprcs}"
        print(msg, flush=True)
        log_file.write(msg + "\n")

    log_file.close()
    print("\nAblation complete. Results saved to logs/ablation_disc_weight.log", flush=True)


if __name__ == '__main__':
    main()
