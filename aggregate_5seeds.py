import numpy as np
import pandas as pd

seeds = [42, 52, 62, 72, 82]
results = []

for seed in seeds:
    npz_path = f'predictions/landmark/phase2_seed{seed}.npz'
    d = np.load(npz_path)
    
    results.append({
        'seed': seed,
        'best_epoch': int(d['best_epoch']),
        'best_val_auprc': float(d['best_val_auprc']),
        'optimal_threshold': float(d['optimal_threshold']),
        'test_f1': float(d['test_f1']),
        'test_precision': float(d['test_precision']),
        'test_recall': float(d['test_recall']),
        'test_specificity': float(d['test_specificity']),
        'test_balanced_accuracy': float(d['test_balanced_accuracy']),
        'test_mcc': float(d['test_mcc']),
    })
    
    test_y_true = d['test_y_true']
    test_y_pred = d['test_y_pred']
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    test_auroc = roc_auc_score(test_y_true, test_y_pred)
    test_auprc = average_precision_score(test_y_true, test_y_pred)
    
    results[-1]['test_auroc'] = test_auroc
    results[-1]['test_auprc'] = test_auprc

df = pd.DataFrame(results)

print('=' * 100)
print('Ours-best 5-Seed Results Summary')
print('=' * 100)
print()
print(df.to_string(index=False))
print()
print('=' * 100)
print('Mean ± Std')
print('=' * 100)

metrics = ['best_val_auprc', 'test_auprc', 'test_auroc', 'test_f1', 
           'test_precision', 'test_recall', 'test_specificity', 
           'test_balanced_accuracy', 'test_mcc']

for metric in metrics:
    mean = df[metric].mean()
    std = df[metric].std()
    print(f'{metric:<25}: {mean:.4f} ± {std:.4f}')

print('=' * 100)

df.to_csv('results/ours_best_5seeds_summary.csv', index=False)
print('\nSaved to: results/ours_best_5seeds_summary.csv')
