import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.tstr_pipeline import TSTRPipeline, extract_features_and_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['stasy', 'tsdiff'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--pipeline_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/b2_baseline/formal_runs/tstr')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=64)

    pipe = TSTRPipeline(generative_model=None, downstream_classifier='xgboost')
    pipe.load(args.pipeline_path)

    x_val, y_val = extract_features_and_labels(val_loader, device)
    x_test, y_test = extract_features_and_labels(test_loader, device)
    val_pred = pipe.predict(x_val)
    test_pred = pipe.predict(x_test)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, f'{args.model}_seed{args.seed}_predictions.npz'),
        val_y_true=y_val,
        val_y_pred=val_pred,
        test_y_true=y_test,
        test_y_pred=test_pred,
    )
    print(f'✓ recomputed {args.model} seed {args.seed}', flush=True)


if __name__ == '__main__':
    main()
