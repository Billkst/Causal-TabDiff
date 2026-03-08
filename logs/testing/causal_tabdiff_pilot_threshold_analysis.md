# CausalTabDiff Pilot Threshold Analysis

- Generated at: 2026-03-07T09:44:36.723979
- Seed: 7
- Device: cuda
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Params(M): 0.0218
- Train seconds: 9.76
- Real Y rate: 0.0286
- Fake Y rate: 0.0273
- XGB scale_pos_weight: 35.5714

## Base Metrics

- ATE_Bias: 0.014587
- Wasserstein: 0.665603
- CMD: 0.009883
- TSTR_AUC: 0.553381
- TSTR_F1: 0.000000

## Score Diagnostics

- ROC_AUC (recomputed): 0.553381
- PR_AUC: 0.034014

## Threshold Sweeps

### Default threshold 0.5

- threshold: 0.500000
- predicted_positive_rate: 0.0017
- F1: 0.000000
- precision: 0.000000
- recall: 0.000000
- confusion_matrix[[TN,FP],[FN,TP]]: [[3972, 7], [117, 0]]

### Real-prevalence-matched threshold

- threshold: 0.001054
- predicted_positive_rate: 0.0300
- F1: 0.058333
- precision: 0.056911
- recall: 0.059829
- confusion_matrix[[TN,FP],[FN,TP]]: [[3863, 116], [110, 7]]

### Fake-prevalence-matched threshold

- threshold: 0.001057
- predicted_positive_rate: 0.0273
- F1: 0.052402
- precision: 0.053571
- recall: 0.051282
- confusion_matrix[[TN,FP],[FN,TP]]: [[3873, 106], [111, 6]]

### Oracle best F1 on eval set

- threshold: 0.000032
- predicted_positive_rate: 0.4636
- F1: 0.069444
- precision: 0.036862
- recall: 0.598291
- confusion_matrix[[TN,FP],[FN,TP]]: [[2150, 1829], [47, 70]]

## Interpretation

- The zero F1 is primarily a thresholding/cutoff problem rather than a complete loss of ranking signal.
- This analysis remains pilot-only and does not authorize full-scale training.