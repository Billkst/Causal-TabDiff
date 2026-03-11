"""
LEGACY - DO NOT USE FOR NEW EXPERIMENTS

This file uses old data pipeline. For new experiments, use: run_baselines_landmark.py
See: docs/reboot/LEGACY_ENTRYPOINTS.md
"""
import argparse
import random
import numpy as np
import torch
import logging
import os
import time
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


def resolve_metadata_path(data_dir):
    env_path = os.environ.get('DATASET_METADATA_PATH', '').strip()
    if env_path:
        if not os.path.isabs(env_path):
            env_path = os.path.abspath(env_path)
        if os.path.exists(env_path):
            return env_path
        raise FileNotFoundError(f"DATASET_METADATA_PATH points to missing file: {env_path}")

    meta_path = os.path.join(data_dir, 'dataset_metadata.json')
    if os.path.exists(meta_path):
        return meta_path
    return 'src/data/dataset_metadata.json'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from xgboost import XGBClassifier
from econml.dml import LinearDML

METRIC_COLUMNS = [
    "ATE_Bias",
    "Wasserstein",
    "CMD",
    "TSTR_AUC",
    "TSTR_PR_AUC",
    "TSTR_F1",
    "TSTR_F1_RealPrev",
    "TSTR_F1_FakePrev",
]
EXTRA_COLUMNS = ["Params(M)", "AvgInfer(ms/sample)"]
REPORT_COLUMNS = METRIC_COLUMNS + EXTRA_COLUMNS
MODEL_ORDER = [
    'CausalForest (Classic)',
    'STaSy (ICLR 23)',
    'TabSyn (ICLR 24)',
    'TabDiff (ICLR 25)',
    'TSDiff (ICLR 23)'
]

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
    t0 = time.time()
    real_x_flat = real_x.reshape(real_x.shape[0], -1).cpu().numpy()
    fake_x_flat = np.nan_to_num(fake_x.reshape(fake_x.shape[0], -1).cpu().numpy(), nan=0.0, posinf=1.0, neginf=-1.0)
    
    real_y_flat = real_y.cpu().numpy().reshape(-1)
    fake_y_flat = np.nan_to_num(fake_y.cpu().numpy().reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    logger.info(f"[Metrics] input shapes real_x={real_x_flat.shape}, fake_x={fake_x_flat.shape}, real_y={real_y_flat.shape}, fake_y={fake_y_flat.shape}")
    t = alpha_tgt.cpu().numpy().reshape(-1)
    t = (t > 0.5).astype(int) # Binarize treatment as requested

    max_eval_rows = int(os.environ.get('METRICS_MAX_ROWS', '0'))
    if max_eval_rows > 0 and real_x_flat.shape[0] > max_eval_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(real_x_flat.shape[0], size=max_eval_rows, replace=False)
        real_x_flat = real_x_flat[idx]
        fake_x_flat = fake_x_flat[idx]
        real_y_flat = real_y_flat[idx]
        fake_y_flat = fake_y_flat[idx]
        t = t[idx]
        logger.info(f"[Metrics] subsampled rows to {max_eval_rows} for diagnostic speed")
    logger.info(f"[Metrics] preprocess done in {time.time() - t0:.2f}s")
    
    # 1. Distributional Fidelity
    t_w = time.time()
    w_dists = []
    for dim in range(real_x_flat.shape[1]):
        w_dists.append(wasserstein_distance(real_x_flat[:, dim], fake_x_flat[:, dim]))
    wasserstein = np.mean(w_dists)
    
    cmd = cmd_dist(real_x_flat, fake_x_flat)
    logger.info(f"[Metrics] fidelity (Wasserstein+CMD) done in {time.time() - t_w:.2f}s")
    
    # 2. ATE Bias (LinearDML proxy via EconML)
    real_y_bounds = (real_y_flat > 0.5).astype(float)
    fake_y_bounds = (fake_y_flat > 0.5).astype(float)

    def prevalence_align_by_rank(score_vec, target_binary):
        target_rate = float(np.mean(target_binary))
        n = len(score_vec)
        k_pos = int(round(target_rate * n))
        k_pos = max(0, min(n, k_pos))
        out = np.zeros(n, dtype=float)
        if k_pos > 0:
            order = np.argsort(score_vec)
            out[order[-k_pos:]] = 1.0
        return out

    # Collapse-guard: if fake_y collapses to a single class while real_y has both classes,
    # preserve rank information and match real prevalence to avoid degenerate downstream metrics.
    if len(np.unique(fake_y_bounds)) < 2 and len(np.unique(real_y_bounds)) == 2:
        fake_y_bounds = prevalence_align_by_rank(fake_y_flat, real_y_bounds)
        k_pos = int(np.sum(fake_y_bounds))
        logger.warning(f"[Metrics] fake_y collapsed to one class; applied rank-based prevalence calibration with k_pos={k_pos}/{len(fake_y_bounds)}")

    align_fake_prevalence = os.environ.get('METRICS_ALIGN_FAKE_PREVALENCE', '0') == '1'
    if align_fake_prevalence:
        before_rate = float(np.mean(fake_y_bounds))
        fake_y_bounds = prevalence_align_by_rank(fake_y_flat, real_y_bounds)
        after_rate = float(np.mean(fake_y_bounds))
        real_rate = float(np.mean(real_y_bounds))
        logger.info(f"[Metrics] prevalence alignment enabled: fake_rate {before_rate:.4f} -> {after_rate:.4f}, real_rate={real_rate:.4f}")

    t_ate = time.time()
    try:
        from sklearn.linear_model import LogisticRegression

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
    logger.info(f"[Metrics] ATE done in {time.time() - t_ate:.2f}s")
    
    def prevalence_threshold(score_vec, target_binary):
        target_rate = float(np.mean(target_binary))
        q = float(np.clip(1.0 - target_rate, 0.0, 1.0))
        return float(np.quantile(score_vec, q))

    # 3. TSTR Efficacy (Binary Classification)
    t_tstr = time.time()
    fake_y_class = fake_y_bounds.astype(int)
    real_y_class = real_y_bounds.astype(int)
    fake_pos = int(np.sum(fake_y_class == 1))
    fake_neg = int(np.sum(fake_y_class == 0))
    real_pos = int(np.sum(real_y_class == 1))
    real_neg = int(np.sum(real_y_class == 0))
    logger.info(f"[Metrics] label dist fake(y=1:{fake_pos}, y=0:{fake_neg}) real(y=1:{real_pos}, y=0:{real_neg})")
    if len(np.unique(fake_y_class)) < 2 or len(np.unique(real_y_class)) < 2:
        logger.warning("fake_y or real_y lacks both classes. Using baseline AUC=0.5, F1=0.0")
        tstr_auc = 0.5
        tstr_pr_auc = float(np.mean(real_y_class)) if len(real_y_class) > 0 else 0.0
        tstr_f1 = 0.0
        tstr_f1_real_prev = 0.0
        tstr_f1_fake_prev = 0.0
    else:
        try:
            # 引入自适应类权重以应对极度不平衡(2%)数据，防止弱信号下所有概率全部低于0.5导致全部被判阴性
            # 这是处理医学高不平衡数据的通行学术规范
            pos_count = np.sum(fake_y_class == 1)
            neg_count = np.sum(fake_y_class == 0)
            scale_pos_weight = neg_count / max(1, pos_count) if pos_count > 0 else 1.0

            tstr_model = XGBClassifier(
                eval_metric='logloss', 
                random_state=42,
                scale_pos_weight=scale_pos_weight
            )
            tstr_model.fit(fake_x_flat, fake_y_class)
            t_pred_proba = tstr_model.predict_proba(real_x_flat)[:, 1]
            t_pred_class = tstr_model.predict(real_x_flat)
            tstr_auc = roc_auc_score(real_y_class, t_pred_proba)
            tstr_pr_auc = average_precision_score(real_y_class, t_pred_proba)
            tstr_f1 = f1_score(real_y_class, t_pred_class)

            real_prev_threshold = prevalence_threshold(t_pred_proba, real_y_class)
            fake_prev_threshold = prevalence_threshold(t_pred_proba, fake_y_class)
            t_pred_real_prev = (t_pred_proba >= real_prev_threshold).astype(int)
            t_pred_fake_prev = (t_pred_proba >= fake_prev_threshold).astype(int)
            tstr_f1_real_prev = f1_score(real_y_class, t_pred_real_prev, zero_division=0)
            tstr_f1_fake_prev = f1_score(real_y_class, t_pred_fake_prev, zero_division=0)
            logger.info(
                f"[Metrics] prevalence-aware thresholds real_prev={real_prev_threshold:.6f}, "
                f"fake_prev={fake_prev_threshold:.6f}, "
                f"f1_real_prev={tstr_f1_real_prev:.6f}, f1_fake_prev={tstr_f1_fake_prev:.6f}, pr_auc={tstr_pr_auc:.6f}"
            )
        except Exception as e:
            logger.error(f"TSTR Error: {e}")
            tstr_auc = float('nan')
            tstr_pr_auc = float('nan')
            tstr_f1 = float('nan')
            tstr_f1_real_prev = float('nan')
            tstr_f1_fake_prev = float('nan')
    logger.info(f"[Metrics] TSTR done in {time.time() - t_tstr:.2f}s")
    logger.info(f"[Metrics] total compute_metrics time {time.time() - t0:.2f}s")
        
    return {
        "ATE_Bias": ate_bias,
        "Wasserstein": wasserstein,
        "CMD": cmd,
        "TSTR_AUC": tstr_auc,
        "TSTR_PR_AUC": tstr_pr_auc,
        "TSTR_F1": tstr_f1,
        "TSTR_F1_RealPrev": tstr_f1_real_prev,
        "TSTR_F1_FakePrev": tstr_f1_fake_prev,
    }


def safe_mean_std(values):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(np.mean(arr)), float(np.std(arr))


def format_mean_std(mean_v, std_v):
    if not np.isfinite(mean_v) or not np.isfinite(std_v):
        return "N/A"
    return f"{mean_v:.4f} ± {std_v:.4f}"


def estimate_trainable_params_m(wrapper):
    if hasattr(wrapper, 'estimate_params_count'):
        try:
            custom_count = float(wrapper.estimate_params_count())
            if np.isfinite(custom_count) and custom_count > 0:
                return custom_count / 1e6
        except Exception:
            pass

    total = 0
    seen = set()
    for obj in wrapper.__dict__.values():
        if isinstance(obj, torch.nn.Module):
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            total += sum(p.numel() for p in obj.parameters() if p.requires_grad)
    if total <= 0:
        return float('nan')
    return total / 1e6


def parse_existing_markdown_rows(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f.readlines()]

    table_lines = [ln.strip() for ln in lines if ln.strip().startswith('|')]
    if len(table_lines) < 3:
        return {}

    headers = [h.strip() for h in table_lines[0].strip('|').split('|')]
    if not headers or headers[0] != 'Model':
        return {}

    parsed = {}
    for row_line in table_lines[2:]:
        cells = [c.strip() for c in row_line.strip('|').split('|')]
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells))
        model = row.get('Model', '')
        if not model:
            continue
        parsed[model] = row
    return parsed


def merge_report_rows(existing_rows, current_rows):
    merged = {}

    for model, row in existing_rows.items():
        merged[model] = {col: row.get(col, 'N/A') for col in REPORT_COLUMNS}

    for model, row in current_rows.items():
        merged[model] = {col: row.get(col, 'N/A') for col in REPORT_COLUMNS}

    ordered_models = [m for m in MODEL_ORDER if m in merged]
    ordered_models.extend([m for m in merged.keys() if m not in ordered_models])

    return ordered_models, merged


def to_latex_cell(value):
    if value == "N/A":
        return "N/A"
    if "±" in value:
        return "$" + value.replace("±", "\\pm") + "$"
    return value


def write_reports(current_rows):
    existing_rows = parse_existing_markdown_rows('markdown_report.md')
    ordered_models, merged = merge_report_rows(existing_rows, current_rows)

    table_data = []
    for model in ordered_models:
        row = [model] + [merged[model][col] for col in REPORT_COLUMNS]
        table_data.append(row)

    md_table = tabulate(table_data, headers=["Model"] + REPORT_COLUMNS, tablefmt="github")
    with open('markdown_report.md', 'w', encoding='utf-8') as f:
        f.write("# Baseline Evaluation Results\n\n")
        f.write(md_table)
        f.write("\n")

    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(REPORT_COLUMNS) + "}")
    latex_lines.append("\\hline")
    latex_lines.append("Model & " + " & ".join(REPORT_COLUMNS) + " \\\\")
    latex_lines.append("\\hline")
    for model in ordered_models:
        cells = [model] + [to_latex_cell(merged[model][col]) for col in REPORT_COLUMNS]
        latex_lines.append(" & ".join(cells) + " \\\\")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Baseline Evaluation Results}")
    latex_lines.append("\\label{tab:baselines}")
    latex_lines.append("\\end{table}")

    with open('latex_report.txt', 'w', encoding='utf-8') as f:
        f.write("% Baseline Evaluation Results LaTeX Code\n")
        f.write("\n".join(latex_lines))
        f.write("\n")

    logger.info("\n" + md_table)

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
        'TSDiff (ICLR 23)': TSDiffWrapper,
        'Causal-TabDiff (Ours)': CausalTabDiffWrapper
    }
    
    if args.model == 'all':
        raise ValueError("安全约束：当前仅允许单模型运行。请使用 --model 指定一个模型（TabSyn/TabDiff/TSDiff 同样一次只能跑一个）。")

    if args.model in model_classes:
        model_classes = {args.model: model_classes[args.model]}
        logger.info(f"Filtered execution to single model: {args.model}")
    else:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(model_classes.keys())}")

    raw_results = {model: {m: [] for m in REPORT_COLUMNS} for model in model_classes.keys()}

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
            params_m = estimate_trainable_params_m(wrapper)
            raw_results[model_name]["Params(M)"].append(params_m)
            logger.info(f"[Complexity] {model_name} trainable params(M): {params_m if np.isfinite(params_m) else 'N/A'}")
            
            # Load metadata
            import json, os
            meta_path = resolve_metadata_path(args.data_dir)
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                 
            # 2. Evaluate
            logger.info("Sampling and Evaluating metrics...")
            all_real_x, all_fake_x, all_real_y, all_fake_y, all_alpha = [], [], [], [], []
            infer_time_total_s = 0.0
            infer_sample_count = 0
            for i, batch in enumerate(dataloader):
                if args.debug_mode and i >= 2: break 
                
                real_x_analog = batch['x'].to(device)
                alpha_tgt = batch['alpha_target'].to(device)
                real_y = batch['y'].to(device)
                cat_raw = batch['x_cat_raw'].to(device).float()
                
                # Sample
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                sample_t0 = time.perf_counter()
                fake_x, fake_y = wrapper.sample(batch_size=real_x_analog.shape[0], alpha_target=alpha_tgt, device=device)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                infer_time_total_s += (time.perf_counter() - sample_t0)
                infer_sample_count += int(real_x_analog.shape[0])
                
                
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
                for m in METRIC_COLUMNS:
                    raw_results[model_name][m].append(metrics_dict[m])
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Metrics Evaluation halted for {model_name} on Seed {seed}: {e}")
                for m in METRIC_COLUMNS:
                    raw_results[model_name][m].append(float('nan'))

            if infer_sample_count > 0:
                infer_ms_per_sample = infer_time_total_s * 1000.0 / infer_sample_count
            else:
                infer_ms_per_sample = float('nan')
            raw_results[model_name]["AvgInfer(ms/sample)"].append(infer_ms_per_sample)
            logger.info(f"[Latency] {model_name} avg infer(ms/sample): {infer_ms_per_sample if np.isfinite(infer_ms_per_sample) else 'N/A'}")

    # 3. Aggregate Results
    logger.info("=== Aggregating Results ===")
    current_rows = {}

    for model in model_classes.keys():
        row = {}
        for col in REPORT_COLUMNS:
            mean_v, std_v = safe_mean_std(raw_results[model][col])
            row[col] = format_mean_std(mean_v, std_v)
        current_rows[model] = row

    write_reports(current_rows)
        
    logger.info("Reports saved to markdown_report.md and latex_report.txt")

if __name__ == '__main__':
    main()
