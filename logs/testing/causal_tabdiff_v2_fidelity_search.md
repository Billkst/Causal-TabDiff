# CausalTabDiff V2 Champion Search

- Generated at: 2026-03-07T11:31:15.462551
- Device: cuda
- Seed: 7
- Search profile: fidelity
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Metadata path: src/data/dataset_metadata_noleak.json
- Baseline report path: /home/UserData/ljx/Project_2/Causal-TabDiff/markdown_report.md
- Total wall time seconds: 104.31

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

- Name: fidelity_out038_ep10_d40_g140_m100
- Hard wins: 2/5
- Table all-win: False
- Formal gate pass: True
- Params(M): 0.025344 (acceptable)
- AvgInfer(ms/sample): 0.562446 (acceptable)
- ATE_Bias: 0.037913
- Wasserstein: 0.563538
- CMD: 0.001643
- TSTR_AUC: 0.603394
- TSTR_PR_AUC: 0.040303
- TSTR_F1: 0.076046
- TSTR_F1_RealPrev: 0.064516
- TSTR_F1_FakePrev: 0.064516

## All Config Results

### fidelity_out038_ep10_d40_g140_m100

- config: model_w=1.00, guidance=1.40, smooth=0.10, cf=0.05, outcome_w=0.38, epochs=10, diffusion_steps=40
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: True
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 13.60
- infer_seconds: 2.30
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.037913
- Wasserstein: 0.563538
- CMD: 0.001643
- TSTR_AUC: 0.603394
- TSTR_PR_AUC: 0.040303
- TSTR_F1: 0.076046
- TSTR_F1_RealPrev: 0.064516
- TSTR_F1_FakePrev: 0.064516
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.562446
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### fidelity_out032_ep10_d40_g140_m100

- config: model_w=1.00, guidance=1.40, smooth=0.10, cf=0.05, outcome_w=0.32, epochs=10, diffusion_steps=40
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 13.71
- infer_seconds: 2.40
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.037254
- Wasserstein: 0.562755
- CMD: 0.001732
- TSTR_AUC: 0.601470
- TSTR_PR_AUC: 0.039516
- TSTR_F1: 0.076923
- TSTR_F1_RealPrev: 0.058824
- TSTR_F1_FakePrev: 0.058091
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.584808
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### fidelity_out025_ep12_d45_g125_m090

- config: model_w=0.90, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.25, epochs=12, diffusion_steps=45
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: True
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 16.56
- infer_seconds: 2.28
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.049577
- Wasserstein: 0.527607
- CMD: 0.001564
- TSTR_AUC: 0.617882
- TSTR_PR_AUC: 0.041420
- TSTR_F1: 0.067797
- TSTR_F1_RealPrev: 0.061947
- TSTR_F1_FakePrev: 0.073469
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.556171
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### fidelity_out030_ep12_d45_g125_m100

- config: model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.30, epochs=12, diffusion_steps=45
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: True
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=borderline
- train_seconds: 16.30
- infer_seconds: 2.64
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.046556
- Wasserstein: 0.527624
- CMD: 0.001575
- TSTR_AUC: 0.608445
- TSTR_PR_AUC: 0.041318
- TSTR_F1: 0.071048
- TSTR_F1_RealPrev: 0.070175
- TSTR_F1_FakePrev: 0.066390
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.645337
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### fidelity_out035_ep10_d40_g140_m090

- config: model_w=0.90, guidance=1.40, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=40
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 13.61
- infer_seconds: 2.39
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.033113
- Wasserstein: 0.563601
- CMD: 0.001647
- TSTR_AUC: 0.605365
- TSTR_PR_AUC: 0.039179
- TSTR_F1: 0.071661
- TSTR_F1_RealPrev: 0.062016
- TSTR_F1_FakePrev: 0.062016
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.582308
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### fidelity_anchor_out035_ep10_d35_g150_m100

- config: model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.12
- infer_seconds: 2.10
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.034110
- Wasserstein: 0.526792
- CMD: 0.001551
- TSTR_AUC: 0.596752
- TSTR_PR_AUC: 0.034542
- TSTR_F1: 0.047431
- TSTR_F1_RealPrev: 0.033898
- TSTR_F1_FakePrev: 0.032520
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.512248
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

## Interpretation

- No searched V2 configuration yet dominates all current table metrics simultaneously; continue with targeted search around the best candidate rather than broad expansion.
- The best searched configuration also passes the formal prevalence-aware gate, so it is eligible for the next multi-seed confirmation stage.