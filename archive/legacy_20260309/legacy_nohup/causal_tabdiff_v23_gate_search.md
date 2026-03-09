# CausalTabDiff V2 Champion Search

- Generated at: 2026-03-07T13:37:24.490880
- Device: cuda
- Seed: 7
- Search profile: v23_gate
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Metadata path: src/data/dataset_metadata_noleak.json
- Baseline report path: /home/UserData/ljx/Project_2/Causal-TabDiff/markdown_report.md
- Total wall time seconds: 133.57

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

- Name: v23_g175_r003_o035_m100
- Hard wins: 2/5
- Table all-win: False
- Formal gate pass: False
- Params(M): 0.025456 (acceptable)
- AvgInfer(ms/sample): 0.538139 (acceptable)
- ATE_Bias: 0.033280
- Wasserstein: 0.543558
- CMD: 0.001684
- TSTR_AUC: 0.592537
- TSTR_PR_AUC: 0.037284
- TSTR_F1: 0.077295
- TSTR_F1_RealPrev: 0.035398
- TSTR_F1_FakePrev: 0.041322

## All Config Results

### v23_g175_r003_o035_m100

- config: use_noise_head=False, model_w=1.00, guidance=1.75, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.86
- infer_seconds: 2.20
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.033280
- Wasserstein: 0.543558
- CMD: 0.001684
- TSTR_AUC: 0.592537
- TSTR_PR_AUC: 0.037284
- TSTR_F1: 0.077295
- TSTR_F1_RealPrev: 0.035398
- TSTR_F1_FakePrev: 0.041322
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.538139
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### v23_g175_r002_o040_m090

- config: use_noise_head=False, model_w=0.90, guidance=1.75, smooth=0.10, cf=0.05, outcome_w=0.40, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.02, moment=0.00
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 12.70
- infer_seconds: 1.63
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.035512
- Wasserstein: 0.543140
- CMD: 0.001664
- TSTR_AUC: 0.608992
- TSTR_PR_AUC: 0.037567
- TSTR_F1: 0.073620
- TSTR_F1_RealPrev: 0.026087
- TSTR_F1_FakePrev: 0.032653
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.397952
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### v23_g150_r003_o040_m090

- config: use_noise_head=False, model_w=0.90, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.40, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 12.18
- infer_seconds: 1.87
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.036257
- Wasserstein: 0.543499
- CMD: 0.001717
- TSTR_AUC: 0.605573
- TSTR_PR_AUC: 0.037592
- TSTR_F1: 0.073600
- TSTR_F1_RealPrev: 0.017699
- TSTR_F1_FakePrev: 0.024793
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.456598
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### v23_g150_r004_o035_m100

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.04, moment=0.00
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.63
- infer_seconds: 2.11
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.037710
- Wasserstein: 0.544109
- CMD: 0.001698
- TSTR_AUC: 0.604575
- TSTR_PR_AUC: 0.036252
- TSTR_F1: 0.058939
- TSTR_F1_RealPrev: 0.035242
- TSTR_F1_FakePrev: 0.033195
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.514588
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### v23_g150_r003_o035_m090

- config: use_noise_head=False, model_w=0.90, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 13.22
- infer_seconds: 2.26
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.036313
- Wasserstein: 0.544107
- CMD: 0.001698
- TSTR_AUC: 0.602532
- TSTR_PR_AUC: 0.036121
- TSTR_F1: 0.067797
- TSTR_F1_RealPrev: 0.017467
- TSTR_F1_FakePrev: 0.016598
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.551888
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### v23_anchor_g150_r003_o035_m100

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 17.28
- infer_seconds: 2.25
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.037706
- Wasserstein: 0.544107
- CMD: 0.001698
- TSTR_AUC: 0.600433
- TSTR_PR_AUC: 0.036429
- TSTR_F1: 0.060606
- TSTR_F1_RealPrev: 0.035242
- TSTR_F1_FakePrev: 0.033058
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.549444
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### v23_g150_r003_o040_m100

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.40, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 12.95
- infer_seconds: 2.01
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.037632
- Wasserstein: 0.543499
- CMD: 0.001717
- TSTR_AUC: 0.595795
- TSTR_PR_AUC: 0.036350
- TSTR_F1: 0.066536
- TSTR_F1_RealPrev: 0.026316
- TSTR_F1_FakePrev: 0.040816
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.491226
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### v23_g125_r003_o040_m090

- config: use_noise_head=False, model_w=0.90, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.40, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.56
- infer_seconds: 2.00
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.037322
- Wasserstein: 0.544753
- CMD: 0.001643
- TSTR_AUC: 0.591549
- TSTR_PR_AUC: 0.034755
- TSTR_F1: 0.057823
- TSTR_F1_RealPrev: 0.017699
- TSTR_F1_FakePrev: 0.016598
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.489010
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

## Interpretation

- No searched V2 configuration yet dominates all current table metrics simultaneously; continue with targeted search around the best candidate rather than broad expansion.
- The best searched configuration still fails the formal prevalence-aware gate, so do not escalate to multi-seed until the gate is recovered.