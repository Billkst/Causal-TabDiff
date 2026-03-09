# CausalTabDiff V2 Champion Search

- Generated at: 2026-03-07T11:15:41.809171
- Device: cuda
- Seed: 7
- Search profile: refine
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Metadata path: src/data/dataset_metadata_noleak.json
- Baseline report path: /home/UserData/ljx/Project_2/Causal-TabDiff/markdown_report.md
- Total wall time seconds: 147.48

## Instinct Anchors Used

- Forced noleak metadata and real treatment `cigsmok`.
- Search anchored around the noleak instinct defaults: early `0.50/0.50` blend for directional safety, later stronger candidate around `0.75 model score / 1.0 guidance`.
- V2 search kept `use_trajectory_risk_head=True` and `sample_use_trajectory_risk=True` for all runs.

## Baseline Frontier From Current Table

- CausalForest (Classic): ATE_Bias=0.0382 ± 0.0110, Wasserstein=0.4781 ± 0.0003, CMD=0.5021 ± 0.0000, TSTR_AUC=0.5890 ± 0.0001, TSTR_F1=0.0745 ± 0.0000
- STaSy (ICLR 23): ATE_Bias=0.0061 ± 0.0011, Wasserstein=0.0646 ± 0.0192, CMD=0.0005 ± 0.0001, TSTR_AUC=0.5643 ± 0.0076, TSTR_F1=0.0685 ± 0.0030
- TabSyn (ICLR 24): ATE_Bias=0.0007 ± 0.0002, Wasserstein=0.0846 ± 0.0122, CMD=0.0048 ± 0.0006, TSTR_AUC=0.5047 ± 0.0075, TSTR_F1=0.0552 ± 0.0020
- TabDiff (ICLR 25): ATE_Bias=0.0012 ± 0.0006, Wasserstein=0.1168 ± 0.0047, CMD=0.0043 ± 0.0002, TSTR_AUC=0.4867 ± 0.0166, TSTR_F1=0.0515 ± 0.0017
- TSDiff (ICLR 23): ATE_Bias=0.0009 ± 0.0004, Wasserstein=1.7501 ± 0.1827, CMD=0.0091 ± 0.0086, TSTR_AUC=0.4897 ± 0.0238, TSTR_F1=0.0513 ± 0.0068

- Frontier targets:
  - ATE_Bias: 0.000700
  - Wasserstein: 0.064600
  - CMD: 0.000500
  - TSTR_AUC: 0.589000
  - TSTR_F1: 0.074500
  - Params(M): 0.026400
  - AvgInfer(ms/sample): 0.410100

## Best Current Candidate

- Name: refine_tasklean_out035_ep10_d35_g150
- Hard wins: 2/5
- Table all-win: False
- Formal gate pass: True
- Params(M): 0.025344 (acceptable)
- AvgInfer(ms/sample): 0.493235 (acceptable)
- ATE_Bias: 0.034821
- Wasserstein: 0.531647
- CMD: 0.001143
- TSTR_AUC: 0.603105
- TSTR_PR_AUC: 0.041310
- TSTR_F1: 0.075893
- TSTR_F1_RealPrev: 0.070485
- TSTR_F1_FakePrev: 0.080972

## All Config Results

### refine_tasklean_out035_ep10_d35_g150

- config: model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: True
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.21
- infer_seconds: 2.02
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.034821
- Wasserstein: 0.531647
- CMD: 0.001143
- TSTR_AUC: 0.603105
- TSTR_PR_AUC: 0.041310
- TSTR_F1: 0.075893
- TSTR_F1_RealPrev: 0.070485
- TSTR_F1_FakePrev: 0.080972
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.493235
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### refine_tasklean_out040_ep12_d35_g150

- config: model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.40, epochs=12, diffusion_steps=35
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 17.96
- infer_seconds: 1.65
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.038405
- Wasserstein: 0.444490
- CMD: 0.001456
- TSTR_AUC: 0.615995
- TSTR_PR_AUC: 0.039079
- TSTR_F1: 0.060453
- TSTR_F1_RealPrev: 0.033058
- TSTR_F1_FakePrev: 0.033058
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.402240
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### refine_balance_out030_ep10_d40_g125

- config: model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.30, epochs=10, diffusion_steps=40
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.27
- infer_seconds: 2.16
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.047901
- Wasserstein: 0.542300
- CMD: 0.002016
- TSTR_AUC: 0.603763
- TSTR_PR_AUC: 0.038923
- TSTR_F1: 0.071006
- TSTR_F1_RealPrev: 0.052632
- TSTR_F1_FakePrev: 0.049793
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.528337
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### refine_gatelean_out030_ep12_d40_g100

- config: model_w=1.00, guidance=1.00, smooth=0.10, cf=0.05, outcome_w=0.30, epochs=12, diffusion_steps=40
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 17.91
- infer_seconds: 1.86
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.056436
- Wasserstein: 0.502376
- CMD: 0.001271
- TSTR_AUC: 0.597026
- TSTR_PR_AUC: 0.037856
- TSTR_F1: 0.070853
- TSTR_F1_RealPrev: 0.052174
- TSTR_F1_FakePrev: 0.059480
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.454012
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### refine_balance_out035_ep10_d40_g125

- config: model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=40
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.26
- infer_seconds: 2.36
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.047214
- Wasserstein: 0.542285
- CMD: 0.001951
- TSTR_AUC: 0.595524
- TSTR_PR_AUC: 0.037575
- TSTR_F1: 0.073665
- TSTR_F1_RealPrev: 0.053097
- TSTR_F1_FakePrev: 0.057851
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.575506
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### refine_bridge_out035_ep10_d35_g125

- config: model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 16.91
- infer_seconds: 2.31
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.037638
- Wasserstein: 0.529452
- CMD: 0.001185
- TSTR_AUC: 0.594571
- TSTR_PR_AUC: 0.037776
- TSTR_F1: 0.058212
- TSTR_F1_RealPrev: 0.035088
- TSTR_F1_FakePrev: 0.033195
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.563803
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### refine_bridge_out045_ep10_d35_g125

- config: model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.45, epochs=10, diffusion_steps=35
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.41
- infer_seconds: 1.92
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.037557
- Wasserstein: 0.530420
- CMD: 0.001242
- TSTR_AUC: 0.591052
- TSTR_PR_AUC: 0.047004
- TSTR_F1: 0.060038
- TSTR_F1_RealPrev: 0.044053
- TSTR_F1_FakePrev: 0.049587
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.468133
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### refine_bridge_out040_ep10_d35_g125

- config: model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.40, epochs=10, diffusion_steps=35
- hard_wins: 0/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.33
- infer_seconds: 2.11
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.037613
- Wasserstein: 0.529447
- CMD: 0.001221
- TSTR_AUC: 0.585243
- TSTR_PR_AUC: 0.037246
- TSTR_F1: 0.058577
- TSTR_F1_RealPrev: 0.034783
- TSTR_F1_FakePrev: 0.041494
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.514745
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=N

## Interpretation

- No searched V2 configuration yet dominates all current table metrics simultaneously; continue with targeted search around the best candidate rather than broad expansion.
- The best searched configuration also passes the formal prevalence-aware gate, so it is eligible for the next multi-seed confirmation stage.