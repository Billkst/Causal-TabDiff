# CausalTabDiff V2 Locked 5-Seed Validation

- Generated at: 2026-03-07T12:33:58.129575
- Device: cuda
- Seeds: [7, 42, 123, 1024, 2024]
- Train size per seed: 8192
- Eval size per seed: 4096
- Batch size: 256
- Epochs: 10
- Diffusion steps: 35
- Sample model score weight: 1.0
- Sample guidance scale: 1.5
- Outcome loss weight: 0.35
- Risk smoothness weight: 0.1
- Counterfactual consistency weight: 0.05
- Treatment source: cigsmok
- Metadata path: src/data/dataset_metadata_noleak.json
- Total wall time seconds: 92.25

## Aggregate Summary

- ATE_Bias: 0.033718 ± 0.011563
- Wasserstein: 0.486878 ± 0.078877
- CMD: 0.015125 ± 0.011414
- TSTR_AUC: 0.591250 ± 0.032970
- TSTR_F1: 0.062596 ± 0.022798
- TSTR_PR_AUC: 0.039893 ± 0.006787
- TSTR_F1_RealPrev: 0.045421 ± 0.012237
- TSTR_F1_FakePrev: 0.050596 ± 0.005809
- Params(M): 0.025344 ± 0.000000
- AvgInfer(ms/sample): 0.526933 ± 0.036701
- Gate pass seeds: 1/5
- Mean hard wins vs table frontier: 1/5
- Mean hard metric wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

## Current Baseline Frontier

- CausalForest (Classic): ATE_Bias=0.0382 ± 0.0110, Wasserstein=0.4781 ± 0.0003, CMD=0.5021 ± 0.0000, TSTR_AUC=0.5890 ± 0.0001, TSTR_F1=0.0745 ± 0.0000
- STaSy (ICLR 23): ATE_Bias=0.0061 ± 0.0011, Wasserstein=0.0646 ± 0.0192, CMD=0.0005 ± 0.0001, TSTR_AUC=0.5643 ± 0.0076, TSTR_F1=0.0685 ± 0.0030
- TabSyn (ICLR 24): ATE_Bias=0.0007 ± 0.0002, Wasserstein=0.0846 ± 0.0122, CMD=0.0048 ± 0.0006, TSTR_AUC=0.5047 ± 0.0075, TSTR_F1=0.0552 ± 0.0020
- TabDiff (ICLR 25): ATE_Bias=0.0012 ± 0.0006, Wasserstein=0.1168 ± 0.0047, CMD=0.0043 ± 0.0002, TSTR_AUC=0.4867 ± 0.0166, TSTR_F1=0.0515 ± 0.0017
- TSDiff (ICLR 23): ATE_Bias=0.0009 ± 0.0004, Wasserstein=1.7501 ± 0.1827, CMD=0.0091 ± 0.0086, TSTR_AUC=0.4897 ± 0.0238, TSTR_F1=0.0513 ± 0.0068

## Per-Seed Results

### Seed 7

- formal_gate_pass: False
- train_seconds: 17.45
- infer_seconds: 2.29
- params_m: 0.0253
- ATE_Bias: 0.036887
- Wasserstein: 0.520269
- CMD: 0.001423
- TSTR_AUC: 0.606872
- TSTR_F1: 0.058824
- TSTR_PR_AUC: 0.038387
- TSTR_F1_RealPrev: 0.044053
- TSTR_F1_FakePrev: 0.057613

### Seed 42

- formal_gate_pass: False
- train_seconds: 15.53
- infer_seconds: 2.27
- params_m: 0.0253
- ATE_Bias: 0.037163
- Wasserstein: 0.615709
- CMD: 0.029695
- TSTR_AUC: 0.543160
- TSTR_F1: 0.039841
- TSTR_PR_AUC: 0.034318
- TSTR_F1_RealPrev: 0.034335
- TSTR_F1_FakePrev: 0.040650

### Seed 123

- formal_gate_pass: True
- train_seconds: 15.24
- infer_seconds: 1.97
- params_m: 0.0253
- ATE_Bias: 0.042299
- Wasserstein: 0.454851
- CMD: 0.026864
- TSTR_AUC: 0.594414
- TSTR_F1: 0.090323
- TSTR_PR_AUC: 0.049224
- TSTR_F1_RealPrev: 0.068729
- TSTR_F1_FakePrev: 0.054902

### Seed 1024

- formal_gate_pass: False
- train_seconds: 16.21
- infer_seconds: 1.98
- params_m: 0.0253
- ATE_Bias: 0.010996
- Wasserstein: 0.465906
- CMD: 0.012947
- TSTR_AUC: 0.640824
- TSTR_F1: 0.087464
- TSTR_PR_AUC: 0.046096
- TSTR_F1_RealPrev: 0.043290
- TSTR_F1_FakePrev: 0.049180

### Seed 2024

- formal_gate_pass: False
- train_seconds: 15.22
- infer_seconds: 2.28
- params_m: 0.0253
- ATE_Bias: 0.041245
- Wasserstein: 0.377654
- CMD: 0.004694
- TSTR_AUC: 0.570981
- TSTR_F1: 0.036530
- TSTR_PR_AUC: 0.031441
- TSTR_F1_RealPrev: 0.036697
- TSTR_F1_FakePrev: 0.050633

## Interpretation

- The locked V2 configuration does not yet beat the current 5-baseline table on all hard metrics at the 5-seed mean level.
- This report is the deciding checkpoint for whether V2 can advance directly to ablation and parameter discussion, or whether a structural V2.1 adjustment is still required.