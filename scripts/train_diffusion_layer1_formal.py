import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from evaluation.efficiency import EfficiencyTracker


def predict_scores(model, n_samples, device):
    sample_out = model.sample(n_samples, device)
    if isinstance(sample_out, tuple):
        y_syn = sample_out[-1]
        if y_syn.ndim > 1 and y_syn.shape[1] != 1:
            y_syn = y_syn[:, :1]
    else:
        raise ValueError('sample() 必须返回包含标签/风险输出的元组')
    return torch.sigmoid(y_syn).cpu().numpy().flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['tsdiff', 'stasy'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='outputs/b2_baseline/formal_runs/layer1')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tracker = EfficiencyTracker()

    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=64, num_workers=4)

    sample = next(iter(train_loader))
    seq_len = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]

    if args.model == 'tsdiff':
        from baselines.tsdiff_landmark_wrapper import TSDiffLandmarkWrapper
        model = TSDiffLandmarkWrapper(seq_len=seq_len, feature_dim=feature_dim)
    else:
        from baselines.stasy_landmark_wrapper import STaSyLandmarkWrapper
        model = STaSyLandmarkWrapper(seq_len=seq_len, feature_dim=feature_dim)

    start = time.time()
    with tracker.track_training():
        model.fit(train_loader, epochs=args.epochs, device=device)
    elapsed = time.time() - start

    inner_model = None
    if hasattr(model, 'model') and getattr(model, 'model') is not None:
        inner_model = model.model
    if inner_model is not None:
        tracker.set_model_size(inner_model)

    val_pred = predict_scores(model, len(val_df), device)
    with tracker.track_inference(len(test_df)):
        test_pred = predict_scores(model, len(test_df), device)
    val_true = val_df['y_2year'].to_numpy(dtype=np.int32)
    test_true = test_df['y_2year'].to_numpy(dtype=np.int32)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, f'{args.model}_seed{args.seed}_predictions.npz'),
        val_y_true=val_true,
        val_y_pred=val_pred,
        test_y_true=test_true,
        test_y_pred=test_pred,
    )
    tracker.metrics['elapsed_wall_clock_sec'] = elapsed
    tracker.save_json(os.path.join(args.output_dir, f'{args.model}_efficiency_seed{args.seed}.json'))
    print(f'✓ {args.model} formal layer1 seed {args.seed} 完成', flush=True)


if __name__ == '__main__':
    main()
