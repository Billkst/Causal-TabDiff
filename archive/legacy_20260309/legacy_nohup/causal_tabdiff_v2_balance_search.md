# CausalTabDiff V2 Champion Search

- Generated at: 2026-03-07T11:05:54.917487
- Device: cuda
- Seed: 7
- Search profile: balance
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Metadata path: src/data/dataset_metadata_noleak.json
- Baseline report path: /home/UserData/ljx/Project_2/Causal-TabDiff/markdown_report.md
- Total wall time seconds: 126.37

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

- Name: balance_task_anchor_out050_ep10_d30
- Hard wins: 2/5
- Table all-win: False
- Formal gate pass: False
- Params(M): 0.025344 (acceptable)
- AvgInfer(ms/sample): 0.408123 (acceptable)
- ATE_Bias: 0.033744
- Wasserstein: 0.561424
- CMD: 0.005731
- TSTR_AUC: 0.618399
- TSTR_PR_AUC: 0.041523
- TSTR_F1: 0.080925
- TSTR_F1_RealPrev: 0.043103
- TSTR_F1_FakePrev: 0.049793

## All Config Results

### balance_task_anchor_out050_ep10_d30

- config: model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.50, epochs=10, diffusion_steps=30
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 17.46
- infer_seconds: 1.67
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.033744
- Wasserstein: 0.561424
- CMD: 0.005731
- TSTR_AUC: 0.618399
- TSTR_PR_AUC: 0.041523
- TSTR_F1: 0.080925
- TSTR_F1_RealPrev: 0.043103
- TSTR_F1_FakePrev: 0.049793
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.408123
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### balance_mid_out025_ep10_d40

- config: model_w=1.00, guidance=1.00, smooth=0.10, cf=0.05, outcome_w=0.25, epochs=10, diffusion_steps=40
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: True
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 16.04
- infer_seconds: 2.28
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.024770
- Wasserstein: 0.552072
- CMD: 0.001755
- TSTR_AUC: 0.612600
- TSTR_PR_AUC: 0.045161
- TSTR_F1: 0.072993
- TSTR_F1_RealPrev: 0.061947
- TSTR_F1_FakePrev: 0.057851
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.557797
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### balance_mid_out050_ep12_d40

- config: model_w=1.00, guidance=1.00, smooth=0.10, cf=0.05, outcome_w=0.50, epochs=12, diffusion_steps=40
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: True
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 20.00
- infer_seconds: 2.41
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.046695
- Wasserstein: 0.496558
- CMD: 0.001091
- TSTR_AUC: 0.606564
- TSTR_PR_AUC: 0.040386
- TSTR_F1: 0.061706
- TSTR_F1_RealPrev: 0.066116
- TSTR_F1_FakePrev: 0.066116
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.587701
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### balance_blend_out050_ep10_d30

- config: model_w=0.75, guidance=1.00, smooth=0.10, cf=0.05, outcome_w=0.50, epochs=10, diffusion_steps=30
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 16.18
- infer_seconds: 2.06
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.032830
- Wasserstein: 0.563863
- CMD: 0.004300
- TSTR_AUC: 0.600972
- TSTR_PR_AUC: 0.037570
- TSTR_F1: 0.061927
- TSTR_F1_RealPrev: 0.054902
- TSTR_F1_FakePrev: 0.054902
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.503591
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### balance_genheavy_out010_ep14_d40

- config: model_w=0.75, guidance=0.50, smooth=0.10, cf=0.05, outcome_w=0.10, epochs=14, diffusion_steps=40
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=borderline
- train_seconds: 22.54
- infer_seconds: 2.59
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.034619
- Wasserstein: 0.396989
- CMD: 0.001532
- TSTR_AUC: 0.616422
- TSTR_PR_AUC: 0.040119
- TSTR_F1: 0.064140
- TSTR_F1_RealPrev: 0.035242
- TSTR_F1_FakePrev: 0.074074
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.631620
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### balance_task_anchor_out025_ep12_d30

- config: model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.25, epochs=12, diffusion_steps=30
- hard_wins: 0/5
- table_all_win: False
- formal_gate_pass: True
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 19.82
- infer_seconds: 1.96
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.048077
- Wasserstein: 0.478879
- CMD: 0.004347
- TSTR_AUC: 0.587881
- TSTR_PR_AUC: 0.040273
- TSTR_F1: 0.063315
- TSTR_F1_RealPrev: 0.072727
- TSTR_F1_FakePrev: 0.072727
- Params(M): 0.025344
- AvgInfer(ms/sample): 0.477654
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=N

## Interpretation

- No searched V2 configuration yet dominates all current table metrics simultaneously; continue with targeted search around the best candidate rather than broad expansion.
- The best searched configuration still fails the formal prevalence-aware gate, so do not escalate to multi-seed until the gate is recovered.