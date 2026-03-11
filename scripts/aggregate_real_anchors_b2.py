import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

MODELS = ['lr', 'xgb', 'brf']
SEEDS = [42, 52, 62, 72, 82]
OUTPUT_DIR = 'outputs/real_anchors_b2'

METRICS = [
    'auprc', 'auroc', 'f1', 'precision', 'recall', 'specificity',
    'npv', 'accuracy', 'balanced_accuracy', 'mcc', 'brier_score',
    'calibration_intercept', 'calibration_slope'
]

def aggregate_results():
    results = []
    
    for model in MODELS:
        model_metrics = {metric: [] for metric in METRICS}
        
        for seed in SEEDS:
            metrics_file = os.path.join(OUTPUT_DIR, f'{model}_seed{seed}_metrics.json')
            
            if not os.path.exists(metrics_file):
                print(f"⚠️  Missing: {metrics_file}")
                continue
            
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            for metric in METRICS:
                if metric in data:
                    model_metrics[metric].append(data[metric])
        
        if not model_metrics['auroc']:
            print(f"⚠️  No results for {model}")
            continue
        
        row = {'model': model.upper()}
        for metric in METRICS:
            values = model_metrics[metric]
            if values:
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0
                row[metric] = f"{mean:.4f} ± {std:.4f}"
            else:
                row[metric] = "N/A"
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("B2-1: Real-data Anchors Results (5 seeds)")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    print("\n")
    
    output_file = os.path.join(OUTPUT_DIR, 'real_anchors_summary.csv')
    df.to_csv(output_file, index=False)
    print(f"Summary saved to: {output_file}\n")
    
    return df

if __name__ == '__main__':
    aggregate_results()
