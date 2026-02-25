import argparse
import random
import numpy as np
import torch
import logging
import os
from collections import defaultdict
from tabulate import tabulate

from src.data.data_module import get_dataloader
from src.baselines import (
    CausalForestWrapper,
    STaSyWrapper,
    TSDiffWrapper,
    TabSynWrapper,
    TabDiffWrapper,
    CausalTabDiffWrapper
)

# Ensure log directory exists
os.makedirs('logs/evaluation', exist_ok=True)
BASELINE_LOG_FILE = os.environ.get('BASELINE_LOG_FILE', 'logs/evaluation/baselines.log')
BASELINE_DISABLE_STREAM = os.environ.get('BASELINE_DISABLE_STREAM', '0') == '1'

_handlers = [logging.FileHandler(BASELINE_LOG_FILE, mode='w', encoding='utf-8')]
if not BASELINE_DISABLE_STREAM:
    _handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=_handlers,
    force=True
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
from xgboost import XGBClassifier
from econml.dml import LinearDML

def cmd_dist(x, y):
    """Correlation Matrix Distance between two numeric datasets x and y."""
    rx = np.corrcoef(x, rowvar=False)
    ry = np.corrcoef(y, rowvar=False)
    rx = np.nan_to_num(rx, 0)
    ry = np.nan_to_num(ry, 0)
    norm_x = np.linalg.norm(rx, ord='fro')
    norm_y = np.linalg.norm(ry, ord='fro')
    if norm_x == 0 or norm_y == 0:
        return 1.0
    trace = np.trace(np.dot(rx, ry))
    return 1 - trace / (norm_x * norm_y)

def compute_metrics(real_x, fake_x, real_y, fake_y, alpha_tgt):
    """
    Computes Distributional Fidelity (Wasserstein & CMD), Causal Bias (ATE Bias via EconML),
    and Efficacy (TSTR: AUC and F1 predicting Y from X).
    """
    real_x_flat = real_x.reshape(real_x.shape[0], -1).cpu().numpy()
    fake_x_flat = np.nan_to_num(fake_x.reshape(fake_x.shape[0], -1).cpu().numpy(), nan=0.0, posinf=1.0, neginf=-1.0)
    
    real_y_flat = real_y.cpu().numpy().reshape(-1)
    fake_y_flat = np.nan_to_num(fake_y.cpu().numpy().reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    print(f"DEBUG SHAPE - real_x: {real_x_flat.shape}, fake_x: {fake_x_flat.shape}")
    t = alpha_tgt.cpu().numpy().reshape(-1)
    t = (t > 0.5).astype(int) # Binarize treatment as requested
    
    # 1. Distributional Fidelity
    w_dists = []
    for dim in range(real_x_flat.shape[1]):
        w_dists.append(wasserstein_distance(real_x_flat[:, dim], fake_x_flat[:, dim]))
    wasserstein = np.mean(w_dists)
    
    cmd = cmd_dist(real_x_flat, fake_x_flat)
    
    # 2. ATE Bias (LinearDML proxy via EconML)
    try:
        from sklearn.linear_model import LogisticRegression
        # User explicitly requested we bound logical values. Ensure Y is constrained to [0, 1] bounds.
        # But we must binarize the generator's Y *before* computing ATE to respect probability diffs.
        fake_y_bounds = (fake_y_flat > 0.5).astype(float)
        real_y_bounds = (real_y_flat > 0.5).astype(float)

        # Reverting to Ridge() as LinearDML natively expects continuous float vectors for Y
        model_real = LinearDML(model_y=Ridge(), model_t=LogisticRegression(max_iter=1000), discrete_treatment=True, random_state=42)
        model_real.fit(Y=real_y_bounds, T=t, X=real_x_flat)
        ate_real = np.mean(model_real.effect(real_x_flat))
        
        model_fake = LinearDML(model_y=Ridge(), model_t=LogisticRegression(max_iter=1000), discrete_treatment=True, random_state=42)
        model_fake.fit(Y=fake_y_bounds, T=t, X=fake_x_flat)
        ate_fake = np.mean(model_fake.effect(fake_x_flat))
        
        ate_bias = np.abs(ate_real - ate_fake)
        ate_bias = float(np.clip(ate_bias, 0.0, 2.0))
        if np.isnan(ate_bias):
            ate_bias = 2.0
            
    except Exception as e:
        logger.error(f"EconML ATE Error: {e}")
        ate_bias = 2.0
    
    # 3. TSTR Efficacy (Binary Classification)
    fake_y_class = (fake_y_flat > 0.5).astype(int)
    real_y_class = (real_y_flat > 0.5).astype(int)
    if len(np.unique(fake_y_class)) < 2 or len(np.unique(real_y_class)) < 2:
        logger.warning("fake_y or real_y lacks both classes. Using baseline AUC=0.5, F1=0.0")
        tstr_auc = 0.5
        tstr_f1 = 0.0
    else:
        try:
            tstr_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
            tstr_model.fit(fake_x_flat, fake_y_class)
            t_pred_proba = tstr_model.predict_proba(real_x_flat)[:, 1]
            t_pred_class = tstr_model.predict(real_x_flat)
            tstr_auc = roc_auc_score(real_y_class, t_pred_proba)
            tstr_f1 = f1_score(real_y_class, t_pred_class)
        except Exception as e:
            logger.error(f"TSTR Error: {e}")
            tstr_auc = float('nan')
            tstr_f1 = float('nan')
        
    return {"ATE_Bias": ate_bias, "Wasserstein": wasserstein, "CMD": cmd, "TSTR_AUC": tstr_auc, "TSTR_F1": tstr_f1}

def format_latex_table(results_mean, results_std, models, metrics):
    r"""Generates LaTeX code for a table containing mean \pm std"""
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l" + "c" * len(metrics) + "}\n\\hline\n"
    latex_str += "Model & " + " & ".join(metrics) + " \\\\\n\\hline\n"
    
    for model in models:
        row = [model]
        for metric in metrics:
            mean_val = results_mean[model][metric]
            std_val = results_std[model][metric]
            row.append(f"${mean_val:.4f} \\pm {std_val:.4f}$")
        latex_str += " & ".join(row) + " \\\\\n"
        
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Baseline Evaluation Results}\n\\label{tab:baselines}\n\\end{table}"
    return latex_str

def main():
    parser = argparse.ArgumentParser(description="Unified Baseline Evaluation Script")
    parser.add_argument('--debug_mode', action='store_true', help='Force fast execution for smoke testing')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to datasets')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 1024, 2024, 2025, 9999], help='Random seeds to loop over')
    parser.add_argument('--model', type=str, default='all', help='Specify a single model to run (e.g. "TabDiff (ICLR 25)") or "all"')
    args = parser.parse_args()

    if args.debug_mode:
        logger.setLevel(logging.DEBUG)
        args.seeds = [42, 1024] # test only 2 seeds to save time

    device = torch.device('cuda' if torch.cuda.is_available() and not args.debug_mode else 'cpu')
    logger.info(f"Starting Evaluation on device: {device}")
    
    # Pre-load dataloader to infer shapes
    dataloader = get_dataloader(data_dir=args.data_dir, batch_size=64 if not args.debug_mode else 4, debug_mode=args.debug_mode)
    sample_batch = next(iter(dataloader))
    t_steps = sample_batch['x'].shape[1]
    feature_dim = sample_batch['x'].shape[2]

    # Models Registry
    model_classes = {
        'CausalForest (Classic)': CausalForestWrapper,
        'STaSy (ICLR 23)': STaSyWrapper,
        'TabSyn (ICLR 24)': TabSynWrapper,
        'TabDiff (ICLR 25)': TabDiffWrapper,
        'TSDiff (ICLR 23)': TSDiffWrapper
    }
    
    if args.model != 'all':
        if args.model in model_classes:
            model_classes = {args.model: model_classes[args.model]}
            logger.info(f"Filtered execution to single model: {args.model}")
        else:
            raise ValueError(f"Unknown model: {args.model}. Available: {list(model_classes.keys())}")

    metrics_list = ["ATE_Bias", "Wasserstein", "CMD", "TSTR_AUC", "TSTR_F1"]
    raw_results = {model: {m: [] for m in metrics_list} for model in model_classes.keys()}

    for seed in args.seeds:
        logger.info(f"=== Running evaluation for SEED: {seed} ===")
        set_seed(seed)
        
        # In a real workflow, we might recreate dataloaders for specific splits per seed
        
        for model_name, ModelCls in model_classes.items():
            logger.info(f"Evaluating {model_name}...")
            
            wrapper = ModelCls(t_steps=t_steps, feature_dim=feature_dim, diffusion_steps=10 if args.debug_mode else 100)
            
            # 1. Train
            epochs = 1 if args.debug_mode else 100
            wrapper.fit(dataloader, epochs=epochs, device=device, debug_mode=args.debug_mode)
            
            # Load metadata
            import json, os
            meta_path = os.path.join(args.data_dir, 'dataset_metadata.json')
            if not os.path.exists(meta_path):
                 meta_path = 'src/data/dataset_metadata.json'
            with open(meta_path, 'r') as f:
                 meta = json.load(f)
                 
            # 2. Evaluate
            logger.info("Sampling and Evaluating metrics...")
            all_real_x, all_fake_x, all_real_y, all_fake_y, all_alpha = [], [], [], [], []
            for i, batch in enumerate(dataloader):
                if args.debug_mode and i >= 2: break 
                
                real_x_analog = batch['x'].to(device)
                alpha_tgt = batch['alpha_target'].to(device)
                real_y = batch['y'].to(device)
                cat_raw = batch['x_cat_raw'].to(device).float()
                
                # Sample
                fake_x, fake_y = wrapper.sample(batch_size=real_x_analog.shape[0], alpha_target=alpha_tgt, device=device)
                
                
                real_x_raw_list = []
                analog_offset = 0
                cat_idx = 0
                dim_offsets = {} # Mapping to reconstruct the semantic feature correctly
                D_orig = len(meta['columns'])
                
                real_x_raw_t = torch.zeros((real_x_analog.shape[0], real_x_analog.shape[1], D_orig), device=device)
                
                analog_offset = 0
                cat_idx = 0
                
                for i_col, col_meta in enumerate(meta['columns']):
                    if col_meta['type'] == 'continuous':
                        dim = col_meta['dim'] # dim is 1 for continuous
                        real_x_raw_t[:, :, i_col:i_col+1] = real_x_analog[:, :, analog_offset:analog_offset+dim]
                        analog_offset += dim
                    else:
                        dim = col_meta['dim'] # dim is analog_bits
                        real_x_raw_t[:, :, i_col:i_col+1] = cat_raw[:, :, cat_idx:cat_idx+1]
                        analog_offset += dim
                        cat_idx += 1
                        
                real_x = real_x_raw_t[:, -1, :]
                
                # Verify that fake_x matches the actual columns schema
                # print(f"[{model_name}] fake_x.shape = {fake_x.shape}, real_x.shape = {real_x.shape}")
                assert fake_x.shape[-1] == real_x.shape[-1], f"{model_name} fake_x failed strict 5D Enforcement! Expected {real_x.shape}"
                
                all_real_x.append(real_x)
                all_fake_x.append(fake_x)
                all_real_y.append(real_y)
                all_fake_y.append(fake_y)
                all_alpha.append(alpha_tgt)
                
            real_x_full = torch.cat(all_real_x, dim=0)
            fake_x_full = torch.cat(all_fake_x, dim=0)
            real_y_full = torch.cat(all_real_y, dim=0)
            fake_y_full = torch.cat(all_fake_y, dim=0)
            alpha_full = torch.cat(all_alpha, dim=0)
            
            # Metrics
            try:
                metrics_dict = compute_metrics(real_x_full, fake_x_full, real_y_full, fake_y_full, alpha_full)
                for m in metrics_list:
                    raw_results[model_name][m].append(metrics_dict[m])
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Metrics Evaluation halted for {model_name} on Seed {seed}: {e}")
                for m in metrics_list:
                    raw_results[model_name][m].append(float('nan'))

    # 3. Aggregate Results
    logger.info("=== Aggregating Results ===")
    results_mean = {}
    results_std = {}
    
    table_data = []
    
    for model in model_classes.keys():
        results_mean[model] = {}
        results_std[model] = {}
        row = [model]
        for m in metrics_list:
            mean_v = np.mean(raw_results[model][m])
            std_v = np.std(raw_results[model][m])
            results_mean[model][m] = mean_v
            results_std[model][m] = std_v
            row.append(f"{mean_v:.4f} ± {std_v:.4f}")
        table_data.append(row)
        
    # Generate Markdown Table
    md_table = tabulate(table_data, headers=["Model"] + metrics_list, tablefmt="github")
    logger.info("\n" + md_table)
    
    with open('markdown_report.md', 'w') as f:
        f.write("# Baseline Evaluation Results\n\n")
        f.write(md_table)
        f.write("\n")
        
    # Generate LaTeX Table
    latex_table = format_latex_table(results_mean, results_std, model_classes.keys(), metrics_list)
    with open('latex_report.txt', 'w') as f:
        f.write("% Baseline Evaluation Results LaTeX Code\n")
        f.write(latex_table)
        f.write("\n")
        
    logger.info("Reports saved to markdown_report.md and latex_report.txt")

if __name__ == '__main__':
    main()
