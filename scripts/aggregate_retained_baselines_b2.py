import os
import json
import numpy as np
import pandas as pd

SEEDS = [42, 52, 62, 72, 82]
OUTPUT_DIR = 'outputs/retained_baselines_b2'

METRICS = [
    'auprc', 'auroc', 'f1', 'precision', 'recall', 'specificity',
    'npv', 'accuracy', 'balanced_accuracy', 'mcc', 'brier_score',
    'calibration_intercept', 'calibration_slope'
]

def aggregate_causal_forest():
    model_metrics = {metric: [] for metric in METRICS}
    
    for seed in SEEDS:
        metrics_file = os.path.join(OUTPUT_DIR, f'causal_forest_seed{seed}_metrics.json')
        
        if not os.path.exists(metrics_file):
            print(f"⚠️  Missing: {metrics_file}")
            continue
        
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        for metric in METRICS:
            if metric in data:
                model_metrics[metric].append(data[metric])
    
    if not model_metrics['auroc']:
        print(f"⚠️  No results for CausalForest")
        return None
    
    row = {'model': 'CausalForest'}
    for metric in METRICS:
        values = model_metrics[metric]
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0
            row[metric] = f"{mean:.4f} ± {std:.4f}"
        else:
            row[metric] = "N/A"
    
    return row

def main():
    print("\n" + "="*80)
    print("B2-2: Retained Baselines Results (Part A: CausalForest)")
    print("="*80 + "\n")
    
    cf_result = aggregate_causal_forest()
    
    if cf_result:
        df = pd.DataFrame([cf_result])
        print(df.to_string(index=False))
        print("\n")
        
        output_file = os.path.join(OUTPUT_DIR, 'retained_baselines_summary.csv')
        df.to_csv(output_file, index=False)
        print(f"Summary saved to: {output_file}\n")
    else:
        print("No results available yet.\n")
    
    print("="*80)
    print("Part B: TSTR Baselines Status")
    print("="*80)
    print("⚠️  STaSy/TabSyn/TabDiff/TSDiff: Not yet implemented")
    print("   These require TSTR protocol implementation\n")

if __name__ == '__main__':
    main()
