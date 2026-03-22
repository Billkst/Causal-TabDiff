import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.data_module_landmark import LandmarkDataset, collate_fn, load_and_split_data
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory


DEFAULT_TABLE_PATH = "data/landmark_tables/unified_person_landmark_table.pkl"
DEFAULT_CKPT_DIR = "checkpoints/landmark"
DEFAULT_OUT_DIR = "outputs/fulldata_baselines/formal_runs/layer2"


def build_alpha_target(batch, device):
    if "alpha_target" in batch:
        return batch["alpha_target"].float().to(device)
    if "landmark" in batch:
        return batch["landmark"].float().to(device)
    raise KeyError("Batch must contain either 'alpha_target' or 'landmark'.")


def run_split_inference(model, loader, device, split_name):
    y_pred_list, y_true_list, y_mask_list = [], [], []
    total_batches = len(loader)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            alpha = build_alpha_target(batch, device)
            history_length = batch["history_length"].to(device)

            outputs = model(x, alpha, history_length=history_length)
            trajectory_probs = outputs["trajectory"]

            y_pred_list.append(trajectory_probs.cpu().numpy())
            y_true_list.append(batch["trajectory_target"].cpu().numpy())
            y_mask_list.append(batch["trajectory_valid_mask"].cpu().numpy())

            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                print(
                    f"[{split_name}] batch {batch_idx}/{total_batches} done",
                    flush=True,
                )

    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    y_mask = np.concatenate(y_mask_list, axis=0)
    return y_pred, y_true, y_mask


def main():
    parser = argparse.ArgumentParser(description="Infer CausalTabDiff Layer2 trajectory predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table_path", type=str, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Seed: {args.seed}", flush=True)

    ckpt_path = os.path.join(args.ckpt_dir, f"phase2_best_seed{args.seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Checkpoint: {ckpt_path}", flush=True)

    print("Loading and splitting data...", flush=True)
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(args.table_path, args.seed, debug_n_persons=None)

    train_dataset = LandmarkDataset(train_df, landmark_to_idx)
    val_dataset = LandmarkDataset(val_df, landmark_to_idx)
    test_dataset = LandmarkDataset(test_df, landmark_to_idx)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    sample = next(iter(train_loader))
    t_steps = sample["x"].shape[1]
    feature_dim = sample["x"].shape[2]

    print(
        f"Initializing model with t_steps={t_steps}, feature_dim={feature_dim}, diffusion_steps=100, trajectory_len=7",
        flush=True,
    )
    model = CausalTabDiffTrajectory(
        t_steps=t_steps,
        feature_dim=feature_dim,
        diffusion_steps=100,
        trajectory_len=7,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded checkpoint epoch={checkpoint.get('epoch', 'NA')} best_val_auprc={checkpoint.get('best_val_auprc', 'NA')}",
        flush=True,
    )

    print("Running validation inference...", flush=True)
    val_y_pred, val_y_true, val_y_mask = run_split_inference(model, val_loader, device, "val")
    print("Running test inference...", flush=True)
    test_y_pred, test_y_true, test_y_mask = run_split_inference(model, test_loader, device, "test")

    print(
        f"val shapes: pred={val_y_pred.shape}, true={val_y_true.shape}, mask={val_y_mask.shape}",
        flush=True,
    )
    print(
        f"test shapes: pred={test_y_pred.shape}, true={test_y_true.shape}, mask={test_y_mask.shape}",
        flush=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"CausalTabDiff_seed{args.seed}_layer2.npz")
    np.savez(
        out_path,
        val_y_pred=val_y_pred,
        val_y_true=val_y_true,
        val_y_mask=val_y_mask,
        test_y_pred=test_y_pred,
        test_y_true=test_y_true,
        test_y_mask=test_y_mask,
    )
    print(f"Saved Layer2 predictions to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
