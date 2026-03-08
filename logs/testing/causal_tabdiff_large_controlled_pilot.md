# CausalTabDiff Large Controlled Pilot

- Generated at: 2026-03-07T09:52:21.898438
- Device: cuda
- Seeds: [7, 42, 2024]
- Train size per seed: 8192
- Eval size per seed: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Total wall time seconds: 41.39

## Aggregate Summary

- ATE_Bias: 0.048927 ± 0.010432
- Wasserstein: 0.671427 ± 0.126475
- CMD: 0.009622 ± 0.005582
- Base TSTR_AUC: 0.599046 ± 0.022175
- Base TSTR_F1 @0.5: 0.066856 ± 0.009807
- PR_AUC: 0.042965 ± 0.003657
- Real-prev F1: 0.059541 ± 0.015358
- Fake-prev F1: 0.055413 ± 0.014298
- Oracle F1: 0.082555 ± 0.011513
- Seeds with base F1 > 0.05: 3/3
- Seeds with real-prev F1 > 0.05: 2/3

## Per-Seed Results

### Seed 7

- train_seconds: 10.24
- params_m: 0.0218
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.044503
- Wasserstein: 0.648391
- CMD: 0.004533
- base_TSTR_AUC: 0.618709
- base_TSTR_F1@0.5: 0.079914
- PR_AUC: 0.047114
- real_prev_threshold: 0.991193
- real_prev_F1: 0.079646
- fake_prev_F1: 0.074689
- oracle_F1: 0.088235
- oracle_threshold: 0.993692

### Seed 42

- train_seconds: 8.74
- params_m: 0.0218
- real_y_rate: 0.0283
- fake_y_rate: 0.0312
- ATE_Bias: 0.038951
- Wasserstein: 0.836554
- CMD: 0.017393
- base_TSTR_AUC: 0.610372
- base_TSTR_F1@0.5: 0.064378
- PR_AUC: 0.043563
- real_prev_threshold: 0.976155
- real_prev_F1: 0.042373
- fake_prev_F1: 0.040486
- oracle_F1: 0.092929
- oracle_threshold: 0.011705

### Seed 2024

- train_seconds: 8.59
- params_m: 0.0218
- real_y_rate: 0.0259
- fake_y_rate: 0.0312
- ATE_Bias: 0.063327
- Wasserstein: 0.529336
- CMD: 0.006940
- base_TSTR_AUC: 0.568057
- base_TSTR_F1@0.5: 0.056277
- PR_AUC: 0.038217
- real_prev_threshold: 0.970281
- real_prev_F1: 0.056604
- fake_prev_F1: 0.051064
- oracle_F1: 0.066499
- oracle_threshold: 0.007424

## Interpretation

- The larger controlled pilot shows non-trivial downstream utility across all tested seeds, but prevalence-aware gains are still not fully uniform.
- This report does not authorize full-scale training.