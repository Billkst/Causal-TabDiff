import torch
import numpy as np
from src.data.data_module_landmark import get_dataloader
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.evaluation.metrics import compute_all_metrics, find_optimal_threshold
from src.evaluation.plots import generate_all_plots
import json
import os


def evaluate_model(model, dataloader, device, threshold=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            y_2year = batch['y_2year'].to(device)
            alpha_target = torch.rand(x.shape[0], 1).to(device) * 0.8 + 0.1
            
            outputs = model(x, alpha_target)
            risk_pred = outputs['risk_2year'].cpu().numpy()
            y_true = y_2year.cpu().numpy()
            
            all_preds.append(risk_pred)
            all_labels.append(y_true)
    
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten().astype(int)
    
    if threshold is None:
        threshold = 0.5
    
    y_pred_binary = (all_preds >= threshold).astype(int)
    
    return all_labels, all_preds, y_pred_binary


def run_evaluation(model_path, table_path, output_dir, seed=42, debug_n_persons=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_loader = get_dataloader(table_path, 'val', batch_size=32, seed=seed, debug_n_persons=debug_n_persons)
    test_loader = get_dataloader(table_path, 'test', batch_size=32, seed=seed, debug_n_persons=debug_n_persons)
    
    sample_batch = next(iter(val_loader))
    t_steps = sample_batch['x'].shape[1]
    feature_dim = sample_batch['x'].shape[2]
    trajectory_len = sample_batch['trajectory_target'].shape[1]
    
    model = CausalTabDiffTrajectory(t_steps, feature_dim, diffusion_steps=10, trajectory_len=trajectory_len).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    print("Evaluating on validation set...")
    val_y_true, val_y_pred, _ = evaluate_model(model, val_loader, device)
    
    print("Finding optimal threshold...")
    threshold, f1_val = find_optimal_threshold(val_y_true, val_y_pred, metric='f1')
    print(f"Optimal threshold: {threshold:.4f} (Val F1: {f1_val:.4f})")
    
    print("Evaluating on test set...")
    test_y_true, test_y_pred, test_y_binary = evaluate_model(model, test_loader, device, threshold=threshold)
    
    print("Computing metrics...")
    metrics = compute_all_metrics(test_y_true, test_y_pred, threshold=threshold)
    
    print("\n=== Test Set Metrics ===")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"NPV: {metrics['npv']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"Calibration Intercept: {metrics['calibration_intercept']:.4f}")
    print(f"Calibration Slope: {metrics['calibration_slope']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in metrics.items() if k != 'confusion_matrix'}
    metrics_serializable['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print("\nGenerating plots...")
    generate_all_plots(test_y_true, test_y_pred, test_y_binary, output_dir)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/model.pt')
    parser.add_argument('--table_path', type=str, default='data/landmark_tables/unified_person_landmark_table.pkl')
    parser.add_argument('--output_dir', type=str, default='results/evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug_n_persons', type=int, default=None)
    args = parser.parse_args()
    
    run_evaluation(args.model_path, args.table_path, args.output_dir, args.seed, args.debug_n_persons)
