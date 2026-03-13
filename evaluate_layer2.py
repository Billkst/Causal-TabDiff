"""
Layer 2 Trajectory 评估脚本
"""
import sys
import os
import numpy as np
import json
import argparse

sys.path.insert(0, 'src')
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_trajectory(y_pred, y_true, y_mask):
    """评估 trajectory 预测"""
    # 确保维度匹配
    min_len = min(y_pred.shape[1], y_true.shape[1], y_mask.shape[1])
    y_pred = y_pred[:, :min_len]
    y_true = y_true[:, :min_len]
    y_mask = y_mask[:, :min_len]
    
    # 只在 valid mask 下计算
    valid_pred = y_pred[y_mask > 0]
    valid_true = y_true[y_mask > 0]
    
    if len(valid_pred) == 0:
        return {
            'trajectory_mse': np.nan,
            'trajectory_mae': np.nan,
            'valid_coverage': 0.0
        }
    
    mse = mean_squared_error(valid_true, valid_pred)
    mae = mean_absolute_error(valid_true, valid_pred)
    coverage = y_mask.sum() / y_mask.size
    
    return {
        'trajectory_mse': float(mse),
        'trajectory_mae': float(mae),
        'valid_coverage': float(coverage)
    }


def compute_2year_readout(y_pred, y_true, y_mask):
    """从 trajectory 读出 2-year risk"""
    # 假设前2个时间步对应 2-year risk
    pred_2year = y_pred[:, :2].mean(axis=1)
    true_2year = (y_true[:, :2].sum(axis=1) > 0).astype(int)
    
    return pred_2year, true_2year


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Evaluating Layer 2: {args.model_name}")
    print(f"{'='*60}\n")
    
    data = np.load(args.predictions_file)
    y_pred = data['y_pred']
    y_true = data['y_true']
    y_mask = data['y_mask']
    
    if len(y_pred.shape) == 3:
        y_pred = y_pred[:, :, 0]
    
    metrics = evaluate_trajectory(y_pred, y_true, y_mask)
    
    print(f"=== Layer 2 Metrics ===")
    print(f"Trajectory MSE: {metrics['trajectory_mse']:.4f}")
    print(f"Trajectory MAE: {metrics['trajectory_mae']:.4f}")
    print(f"Valid Coverage: {metrics['valid_coverage']:.4f}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, f'{args.model_name}_layer2_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to: {args.output_dir}\n")


if __name__ == '__main__':
    main()
