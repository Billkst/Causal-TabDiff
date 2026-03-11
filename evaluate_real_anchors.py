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
    print(f"Evaluating {args.model_name}")
    print(f"{'='*60}\n")
    
    data = np.load(args.predictions_file)
    val_y_true = data['val_y_true']
    val_y_pred = data['val_y_pred']
    test_y_true = data['test_y_true']
    test_y_pred = data['test_y_pred']
    
    print(f"Val:  {len(val_y_true)} samples, {val_y_true.sum()} positive")
    print(f"Test: {len(test_y_true)} samples, {test_y_true.sum()} positive\n")
    
    threshold, f1_val = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
    print(f"Optimal threshold: {threshold:.4f} (Val F1: {f1_val:.4f})\n")
    
    metrics = compute_all_metrics(test_y_true, test_y_pred, threshold=threshold)
    
    print(f"=== Test Metrics ===")
    print(f"AUROC:       {metrics['auroc']:.4f}")
    print(f"AUPRC:       {metrics['auprc']:.4f}")
    print(f"F1:          {metrics['f1']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"NPV:         {metrics['npv']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Balanced Acc:{metrics['balanced_accuracy']:.4f}")
    print(f"MCC:         {metrics['mcc']:.4f}")
    print(f"Brier:       {metrics['brier_score']:.4f}")
    print(f"Cal Intercept:{metrics['calibration_intercept']:.4f}")
    print(f"Cal Slope:   {metrics['calibration_slope']:.4f}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in metrics.items() if k != 'confusion_matrix'}
    metrics_serializable['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    metrics_serializable['threshold'] = float(threshold)
    metrics_serializable['val_f1'] = float(f1_val)
    
    metrics_file = os.path.join(args.output_dir, f'{args.model_name}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"Metrics saved to: {metrics_file}")
    
    test_y_pred_binary = (test_y_pred >= threshold).astype(int)
    plot_prefix = os.path.join(args.output_dir, args.model_name)
    generate_all_plots(test_y_true, test_y_pred, test_y_pred_binary, plot_prefix)
    
    print(f"Plots saved to: {args.output_dir}/\n")


if __name__ == '__main__':
    main()
