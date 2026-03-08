# CausalTabDiff Multi-Seed Pilot

- Generated at: 2026-03-07T09:47:55.396269
- Device: cuda
- Seeds: [7, 42, 123, 1024, 2024]
- Train size per seed: 4096
- Eval size per seed: 2048
- Batch size: 256
- Epochs: 6
- Diffusion steps: 20
- Treatment source: cigsmok
- Total wall time seconds: 31.69

## Aggregate Summary

- ATE_Bias: 0.032618 ± 0.028460
- Wasserstein: 0.933889 ± 0.136792
- CMD: 0.013849 ± 0.005131
- Base TSTR_AUC: 0.577027 ± 0.024149
- Base TSTR_F1 @0.5: 0.065739 ± 0.020295
- PR_AUC: 0.042132 ± 0.008784
- Real-prev F1: 0.056690 ± 0.018202
- Fake-prev F1: 0.056143 ± 0.016001
- Oracle F1: 0.093053 ± 0.016682
- Seeds with base F1 > 0: 5/5
- Seeds with real-prev F1 > 0.05: 2/5

## Per-Seed Results

### Seed 7

- train_seconds: 4.64
- params_m: 0.0218
- real_y_rate: 0.0293
- fake_y_rate: 0.0352
- ATE_Bias: 0.032210
- Wasserstein: 0.993357
- CMD: 0.006627
- base_TSTR_AUC: 0.575905
- base_TSTR_F1@0.5: 0.054422
- PR_AUC: 0.040866
- real_prev_threshold: 0.986283
- real_prev_F1: 0.066667
- real_prev_precision: 0.066667
- real_prev_recall: 0.066667
- fake_prev_F1: 0.065359
- oracle_F1: 0.084211
- oracle_threshold: 0.994404

### Seed 42

- train_seconds: 2.80
- params_m: 0.0218
- real_y_rate: 0.0376
- fake_y_rate: 0.0312
- ATE_Bias: 0.051714
- Wasserstein: 0.941841
- CMD: 0.011887
- base_TSTR_AUC: 0.622345
- base_TSTR_F1@0.5: 0.103093
- PR_AUC: 0.058432
- real_prev_threshold: 0.959039
- real_prev_F1: 0.038710
- real_prev_precision: 0.038462
- real_prev_recall: 0.038961
- fake_prev_F1: 0.041667
- oracle_F1: 0.120344
- oracle_threshold: 0.239441

### Seed 123

- train_seconds: 2.94
- params_m: 0.0218
- real_y_rate: 0.0239
- fake_y_rate: 0.0312
- ATE_Bias: 0.003140
- Wasserstein: 1.153474
- CMD: 0.022518
- base_TSTR_AUC: 0.562036
- base_TSTR_F1@0.5: 0.053872
- PR_AUC: 0.031776
- real_prev_threshold: 0.999372
- real_prev_F1: 0.046875
- real_prev_precision: 0.037975
- real_prev_recall: 0.061224
- fake_prev_F1: 0.046875
- oracle_F1: 0.071429
- oracle_threshold: 0.997727

### Seed 1024

- train_seconds: 2.94
- params_m: 0.0218
- real_y_rate: 0.0278
- fake_y_rate: 0.0312
- ATE_Bias: 0.075147
- Wasserstein: 0.807094
- CMD: 0.013686
- base_TSTR_AUC: 0.552187
- base_TSTR_F1@0.5: 0.046512
- PR_AUC: 0.039387
- real_prev_threshold: 0.918016
- real_prev_F1: 0.043478
- real_prev_precision: 0.037037
- real_prev_recall: 0.052632
- fake_prev_F1: 0.043478
- oracle_F1: 0.101562
- oracle_threshold: 0.027935

### Seed 2024

- train_seconds: 3.04
- params_m: 0.0218
- real_y_rate: 0.0273
- fake_y_rate: 0.0312
- ATE_Bias: 0.000878
- Wasserstein: 0.773681
- CMD: 0.014530
- base_TSTR_AUC: 0.572661
- base_TSTR_F1@0.5: 0.070796
- PR_AUC: 0.040198
- real_prev_threshold: 0.967611
- real_prev_F1: 0.087719
- real_prev_precision: 0.086207
- real_prev_recall: 0.089286
- fake_prev_F1: 0.083333
- oracle_F1: 0.087719
- oracle_threshold: 0.967611

## Interpretation

- Threshold-aware recovery appears on a subset of seeds, but robustness is still incomplete and should be treated as a controlled pilot signal only.
- This multi-seed pilot is still below the bar for full-scale training authorization.