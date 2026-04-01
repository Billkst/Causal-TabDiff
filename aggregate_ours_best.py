#!/usr/bin/env python3
"""聚合 Ours-best 的 5-seed official 结果"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

npz_dir = Path("outputs/submission_v3/official_npz")
seeds = [42, 52, 62, 72, 82]

results = []
for seed in seeds:
    npz_path = npz_dir / f"phase2_seed{seed}_official.npz"
    data = np.load(npz_path, allow_pickle=True)
    
    test_auprc = average_precision_score(data['test_y_true'], data['test_y_pred'])
    test_auroc = roc_auc_score(data['test_y_true'], data['test_y_pred'])
    
    results.append({
        'seed': int(data['seed']),
        'best_epoch': int(data['best_epoch']),
        'best_val_auprc': float(data['best_val_auprc']),
        'test_auprc': test_auprc,
        'test_auroc': test_auroc,
        'optimal_threshold': float(data['optimal_threshold']),
        'test_f1': float(data['test_f1']),
        'test_precision': float(data['test_precision']),
        'test_recall': float(data['test_recall']),
        'test_specificity': float(data['test_specificity']),
        'test_balanced_accuracy': float(data['test_balanced_accuracy']),
        'test_mcc': float(data['test_mcc'])
    })

df = pd.DataFrame(results)

# 计算统计量
stats = df.describe().loc[['mean', 'std']].T
stats.columns = ['mean', 'std']

# 保存详细结果
df.to_csv("outputs/submission_v3/ours_best_5seed_summary.csv", index=False, float_format='%.6f')

# 生成 markdown 报告
with open("outputs/submission_v3/ours_best_5seed_summary.md", "w") as f:
    f.write("# Ours-best 5-Seed Official Results\n\n")
    f.write("## Individual Seed Results\n\n")
    f.write(df.to_markdown(index=False, floatfmt=".6f"))
    f.write("\n\n## Summary Statistics (Mean ± Std)\n\n")
    
    for metric in stats.index:
        if metric == 'seed':
            continue
        mean_val = stats.loc[metric, 'mean']
        std_val = stats.loc[metric, 'std']
        f.write(f"- **{metric}**: {mean_val:.6f} ± {std_val:.6f}\n")
    
    # 找出最佳和最差 seed
    best_seed = df.loc[df['test_auprc'].idxmax(), 'seed']
    worst_seed = df.loc[df['test_auprc'].idxmin(), 'seed']
    best_auprc = df['test_auprc'].max()
    worst_auprc = df['test_auprc'].min()
    
    f.write(f"\n## Performance Range\n\n")
    f.write(f"- **Best seed**: {int(best_seed)} (test AUPRC = {best_auprc:.6f})\n")
    f.write(f"- **Worst seed**: {int(worst_seed)} (test AUPRC = {worst_auprc:.6f})\n")
    f.write(f"- **Range**: {best_auprc - worst_auprc:.6f}\n")
    f.write(f"- **CV (coefficient of variation)**: {stats.loc['test_auprc', 'std'] / stats.loc['test_auprc', 'mean'] * 100:.2f}%\n")

print("✓ 聚合完成")
print(f"\n主要指标 (Mean ± Std):")
print(f"  Test AUPRC: {stats.loc['test_auprc', 'mean']:.6f} ± {stats.loc['test_auprc', 'std']:.6f}")
print(f"  Test AUROC: {stats.loc['test_auroc', 'mean']:.6f} ± {stats.loc['test_auroc', 'std']:.6f}")
print(f"  Test F1: {stats.loc['test_f1', 'mean']:.6f} ± {stats.loc['test_f1', 'std']:.6f}")
print(f"\n最佳 seed: {int(best_seed)} (AUPRC = {best_auprc:.6f})")
print(f"最差 seed: {int(worst_seed)} (AUPRC = {worst_auprc:.6f})")
