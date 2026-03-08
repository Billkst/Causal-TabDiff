# CausalTabDiff V2 Champion Search

- Generated at: 2026-03-07T13:24:45.895675
- Device: cuda
- Seed: 7
- Search profile: v22_refine
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Metadata path: src/data/dataset_metadata_noleak.json
- Baseline report path: /home/UserData/ljx/Project_2/Causal-TabDiff/markdown_report.md
- Total wall time seconds: 145.32

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

- Name: v22_lightrecon_g150_r003_m000
- Hard wins: 2/5
- Table all-win: False
- Formal gate pass: False
- Params(M): 0.025456 (acceptable)
- AvgInfer(ms/sample): 0.544940 (acceptable)
- ATE_Bias: 0.036313
- Wasserstein: 0.564058
- CMD: 0.001845
- TSTR_AUC: 0.609457
- TSTR_PR_AUC: 0.038439
- TSTR_F1: 0.080808
- TSTR_F1_RealPrev: 0.034934
- TSTR_F1_FakePrev: 0.049587

## All Config Results

### v22_lightrecon_g150_r003_m000

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.47
- infer_seconds: 2.23
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.036313
- Wasserstein: 0.564058
- CMD: 0.001845
- TSTR_AUC: 0.609457
- TSTR_PR_AUC: 0.038439
- TSTR_F1: 0.080808
- TSTR_F1_RealPrev: 0.034934
- TSTR_F1_FakePrev: 0.049587
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.544940
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### v22_lightrecon_g150_r002_m000

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.02, moment=0.00
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.43
- infer_seconds: 2.26
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.036314
- Wasserstein: 0.564056
- CMD: 0.001845
- TSTR_AUC: 0.602229
- TSTR_PR_AUC: 0.038106
- TSTR_F1: 0.079339
- TSTR_F1_RealPrev: 0.043478
- TSTR_F1_FakePrev: 0.041494
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.551254
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### v22_reconmoment_g125_r003_m002

- config: use_noise_head=False, model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.02
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.82
- infer_seconds: 2.22
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.028924
- Wasserstein: 0.566039
- CMD: 0.001713
- TSTR_AUC: 0.601494
- TSTR_PR_AUC: 0.039125
- TSTR_F1: 0.078799
- TSTR_F1_RealPrev: 0.035088
- TSTR_F1_FakePrev: 0.041152
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.542659
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### v22_reconmoment_g150_r005_m002

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.05, moment=0.02
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.59
- infer_seconds: 2.17
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.036301
- Wasserstein: 0.563930
- CMD: 0.001830
- TSTR_AUC: 0.600611
- TSTR_PR_AUC: 0.038430
- TSTR_F1: 0.080672
- TSTR_F1_RealPrev: 0.034483
- TSTR_F1_FakePrev: 0.040650
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.529461
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### v22_base_const_g150_r000_m000

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.00, moment=0.00
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 16.87
- infer_seconds: 2.30
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.036274
- Wasserstein: 0.564115
- CMD: 0.001841
- TSTR_AUC: 0.598842
- TSTR_PR_AUC: 0.038114
- TSTR_F1: 0.081218
- TSTR_F1_RealPrev: 0.026087
- TSTR_F1_FakePrev: 0.041322
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.562381
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### v22_reconmoment_g150_r003_m002

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.02
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.69
- infer_seconds: 2.25
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.036303
- Wasserstein: 0.563927
- CMD: 0.001830
- TSTR_AUC: 0.597686
- TSTR_PR_AUC: 0.038443
- TSTR_F1: 0.080000
- TSTR_F1_RealPrev: 0.034783
- TSTR_F1_FakePrev: 0.041494
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.550187
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### v22_lightrecon_g125_r003_m000

- config: use_noise_head=False, model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.68
- infer_seconds: 2.22
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.028927
- Wasserstein: 0.566169
- CMD: 0.001720
- TSTR_AUC: 0.596476
- TSTR_PR_AUC: 0.038609
- TSTR_F1: 0.080153
- TSTR_F1_RealPrev: 0.034632
- TSTR_F1_FakePrev: 0.052830
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.542120
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### v22_lightrecon_out030_g150_r003_m000

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.30, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.47
- infer_seconds: 1.79
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.035699
- Wasserstein: 0.563619
- CMD: 0.001843
- TSTR_AUC: 0.595779
- TSTR_PR_AUC: 0.038263
- TSTR_F1: 0.080775
- TSTR_F1_RealPrev: 0.035242
- TSTR_F1_FakePrev: 0.032389
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.437844
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

## Interpretation

- No searched V2 configuration yet dominates all current table metrics simultaneously; continue with targeted search around the best candidate rather than broad expansion.
- The best searched configuration still fails the formal prevalence-aware gate, so do not escalate to multi-seed until the gate is recovered.