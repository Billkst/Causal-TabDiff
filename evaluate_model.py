import torch
import numpy as np
import json
import os
from src.evaluation.metrics import compute_all_metrics, find_optimal_threshold
from src.evaluation.plots import generate_all_plots


def evaluate_from_predictions(y_true, y_pred_proba, val_y_true=None, val_y_pred_proba=None, 
                               output_dir=None, model_name='model'):
    if val_y_true is not None and val_y_pred_proba is not None:
        threshold, f1_val = find_optimal_threshold(val_y_true, val_y_pred_proba, metric='f1')
        print(f"Optimal threshold: {threshold:.4f} (Val F1: {f1_val:.4f})")
    else:
        raise ValueError('正式 baseline 评估要求提供验证集预测以选择 F1 最优阈值')
    
    metrics = compute_all_metrics(y_true, y_pred_proba, threshold=threshold)
    
    print(f"\n=== {model_name} Metrics ===")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Brier: {metrics['brier_score']:.4f}")
    print(f"Calibration Intercept: {metrics['calibration_intercept']:.4f}")
    print(f"Calibration Slope: {metrics['calibration_slope']:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in metrics.items() if k != 'confusion_matrix'}
        metrics_serializable['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        
        with open(os.path.join(output_dir, f'{model_name}_metrics.json'), 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        generate_all_plots(y_true, y_pred_proba, y_pred_binary, 
                          os.path.join(output_dir, model_name))
    
    return metrics


def get_ours_predictions(model, dataloader, device, alpha_target_value=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            y_2year = batch['y_2year'].to(device)
            alpha_target = torch.full((x.shape[0], 1), alpha_target_value, device=device)
            
            outputs = model(x, alpha_target)
            risk_pred = outputs['risk_2year'].cpu().numpy()
            y_true = y_2year.cpu().numpy()
            
            all_preds.append(risk_pred)
            all_labels.append(y_true)
    
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten().astype(int)
    
    return all_labels, all_preds


def evaluate_ours_model(model_path, dataloader_val, dataloader_test, device, output_dir, alpha_target_value=0.5):
    from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
    
    sample_batch = next(iter(dataloader_val))
    t_steps = sample_batch['x'].shape[1]
    feature_dim = sample_batch['x'].shape[2]
    trajectory_len = sample_batch['trajectory_target'].shape[1]
    
    model = CausalTabDiffTrajectory(t_steps, feature_dim, diffusion_steps=10, trajectory_len=trajectory_len).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    val_y_true, val_y_pred = get_ours_predictions(model, dataloader_val, device, alpha_target_value)
    test_y_true, test_y_pred = get_ours_predictions(model, dataloader_test, device, alpha_target_value)
    
    return evaluate_from_predictions(test_y_true, test_y_pred, val_y_true, val_y_pred, 
                                    output_dir, model_name='ours')


if __name__ == '__main__':
    import argparse
    from src.data.data_module_landmark import get_dataloader
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['ours', 'baseline'])
    parser.add_argument('--model_path', type=str, default='checkpoints/model.pt')
    parser.add_argument('--predictions_file', type=str, default=None)
    parser.add_argument('--table_path', type=str, default='data/landmark_tables/unified_person_landmark_table.pkl')
    parser.add_argument('--output_dir', type=str, default='results/evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug_n_persons', type=int, default=None)
    parser.add_argument('--alpha_target', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default='baseline')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'ours':
        val_loader = get_dataloader(args.table_path, 'val', batch_size=32, seed=args.seed, debug_n_persons=args.debug_n_persons)
        test_loader = get_dataloader(args.table_path, 'test', batch_size=32, seed=args.seed, debug_n_persons=args.debug_n_persons)
        evaluate_ours_model(args.model_path, val_loader, test_loader, device, args.output_dir, args.alpha_target)
    
    elif args.model_type == 'baseline':
        if args.predictions_file is None:
            raise ValueError("--predictions_file required for baseline")
        
        data = np.load(args.predictions_file)
        
        if 'test_y_true' in data:
            y_true = data['test_y_true']
            y_pred = data['test_y_pred']
            val_y_true = data.get('val_y_true')
            val_y_pred = data.get('val_y_pred')
        elif 'y_true' in data:
            y_true = data['y_true']
            y_pred = data['y_pred']
            val_y_true = None
            val_y_pred = None
        else:
            raise KeyError(f"无法找到 y_true/y_pred 或 test_y_true/test_y_pred")
        
        evaluate_from_predictions(y_true, y_pred, 
                                 val_y_true, val_y_pred,
                                 args.output_dir, model_name=args.model_name)
