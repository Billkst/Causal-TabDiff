#!/usr/bin/env python3
"""聚合 Layer1 已完成的 4 个 baseline 的 5-seed 结果"""
import numpy as np
import pandas as pd
from pathlib import Path

traditional_dir = Path("outputs/submission_v3/traditional_baselines")
ours_dir = Path("outputs/submission_v3/official_npz")
seeds = [42, 52, 62, 72, 82]

results = []

for model_name in ['LogisticRegression', 'RandomForest', 'XGBoost']:
    for seed in seeds:
        npz_path = traditional_dir / f"{model_name}_seed{seed}_official.npz"
        data = np.load(npz_path)
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        test_auprc = average_precision_score(data['test_y_true'], data['test_y_pred'])
        test_auroc = roc_auc_score(data['test_y_true'], data['test_y_pred'])
        
        results.append({
            'model': model_name,
            'seed': int(data['seed']),
            'test_auprc': test_auprc,
            'test_auroc': test_auroc,
            'test_f1': float(data['test_f1']),
            'test_precision': float(data['test_precision']),
            'test_recall': float(data['test_recall']),
            'test_specificity': float(data['test_specificity']),
            'test_balanced_accuracy': float(data['test_balanced_accuracy']),
            'test_mcc': float(data['test_mcc'])
        })

for seed in seeds:
    npz_path = ours_dir / f"phase2_seed{seed}_official.npz"
    data = np.load(npz_path)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    test_auprc = average_precision_score(data['test_y_true'], data['test_y_pred'])
    test_auroc = roc_auc_score(data['test_y_true'], data['test_y_pred'])
    
    results.append({
        'model': 'Ours-best',
        'seed': int(data['seed']),
        'test_auprc': test_auprc,
        'test_auroc': test_auroc,
        'test_f1': float(data['test_f1']),
        'test_precision': float(data['test_precision']),
        'test_recall': float(data['test_recall']),
        'test_specificity': float(data['test_specificity']),
        'test_balanced_accuracy': float(data['test_balanced_accuracy']),
        'test_mcc': float(data['test_mcc'])
    })

df = pd.DataFrame(results)

summary = df.groupby('model').agg(['mean', 'std']).reset_index()
summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

output_dir = Path("outputs/submission_v3")
df.to_csv(output_dir / "layer1_4models_5seed_detail.csv", index=False, float_format='%.6f')
summary.to_csv(output_dir / "layer1_4models_5seed_summary.csv", index=False, float_format='%.6f')

with open(output_dir / "layer1_4models_5seed_summary.md", "w") as f:
    f.write("# Layer1 Baseline 4-Model 5-Seed Summary\n\n")
    f.write("Models: Logistic Regression, Random Forest, XGBoost, Ours-best\n\n")
    f.write("## Mean ± Std\n\n")
    
    for _, row in summary.iterrows():
        model = row['model']
        f.write(f"### {model}\n\n")
        f.write(f"- Test AUPRC: {row['test_auprc_mean']:.6f} ± {row['test_auprc_std']:.6f}\n")
        f.write(f"- Test AUROC: {row['test_auroc_mean']:.6f} ± {row['test_auroc_std']:.6f}\n")
        f.write(f"- Test F1: {row['test_f1_mean']:.6f} ± {row['test_f1_std']:.6f}\n")
        f.write(f"- Test Precision: {row['test_precision_mean']:.6f} ± {row['test_precision_std']:.6f}\n")
        f.write(f"- Test Recall: {row['test_recall_mean']:.6f} ± {row['test_recall_std']:.6f}\n")
        f.write(f"- Test Specificity: {row['test_specificity_mean']:.6f} ± {row['test_specificity_std']:.6f}\n")
        f.write(f"- Test Balanced Accuracy: {row['test_balanced_accuracy_mean']:.6f} ± {row['test_balanced_accuracy_std']:.6f}\n")
        f.write(f"- Test MCC: {row['test_mcc_mean']:.6f} ± {row['test_mcc_std']:.6f}\n\n")

print("✓ 聚合完成")
print(f"\n生成文件:")
print(f"  - layer1_4models_5seed_detail.csv")
print(f"  - layer1_4models_5seed_summary.csv")
print(f"  - layer1_4models_5seed_summary.md")
