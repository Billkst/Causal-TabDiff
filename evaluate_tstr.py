"""
TSTR 评估脚本
"""
import sys
import os
import numpy as np
import json
import argparse

sys.path.insert(0, 'src')
from evaluation.metrics import compute_all_metrics, find_optimal_threshold
from evaluation.plots import generate_all_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Evaluating TSTR: {args.model_name}")
    print(f"{'='*60}\n")
    
    data = np.load(args.predictions_file)
    y_pred = data['y_pred']
    y_true = data['y_true']
    
    print(f"Test: {len(y_true)} samples, {y_true.sum()} positive\n")
    
    threshold = 0.5
    metrics = compute_all_metrics(y_true, y_pred, threshold=threshold)
    
    print(f"=== TSTR Metrics ===")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in metrics.items() if k != 'confusion_matrix'}
    metrics_serializable['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    
    with open(os.path.join(args.output_dir, f'{args.model_name}_tstr_metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    generate_all_plots(y_true, y_pred, y_pred_binary, os.path.join(args.output_dir, args.model_name))
    
    print(f"Results saved to: {args.output_dir}\n")


if __name__ == '__main__':
    main()
