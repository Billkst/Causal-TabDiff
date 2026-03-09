# CausalTabDiff V2 Champion Search

- Generated at: 2026-03-07T13:09:47.359984
- Device: cuda
- Seed: 7
- Search profile: v21
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Metadata path: src/data/dataset_metadata_noleak.json
- Baseline report path: /home/UserData/ljx/Project_2/Causal-TabDiff/markdown_report.md
- Total wall time seconds: 99.89

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

- Name: v21_balance_lateg_out030_d40_r010_m010
- Hard wins: 1/5
- Table all-win: False
- Formal gate pass: False
- Params(M): 0.025456 (acceptable)
- AvgInfer(ms/sample): 0.624092 (borderline)
- ATE_Bias: 0.003272
- Wasserstein: 1.653395
- CMD: 0.004230
- TSTR_AUC: 0.593200
- TSTR_PR_AUC: 0.035079
- TSTR_F1: 0.017316
- TSTR_F1_RealPrev: 0.017699
- TSTR_F1_FakePrev: 0.024896

## All Config Results

### v21_balance_lateg_out030_d40_r010_m010

- config: model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.30, epochs=10, diffusion_steps=40, guidance_schedule=late, guidance_power=2.0, recon=0.10, moment=0.10
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=borderline
- train_seconds: 16.97
- infer_seconds: 2.56
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.003272
- Wasserstein: 1.653395
- CMD: 0.004230
- TSTR_AUC: 0.593200
- TSTR_PR_AUC: 0.035079
- TSTR_F1: 0.017316
- TSTR_F1_RealPrev: 0.017699
- TSTR_F1_FakePrev: 0.024896
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.624092
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### v21_moment_stronger_lateg_out035_d35_r010_m010

- config: model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=late, guidance_power=2.0, recon=0.10, moment=0.10
- hard_wins: 0/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 17.13
- infer_seconds: 2.27
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.008001
- Wasserstein: 1.622118
- CMD: 0.004741
- TSTR_AUC: 0.567520
- TSTR_PR_AUC: 0.035347
- TSTR_F1: 0.064748
- TSTR_F1_RealPrev: 0.043860
- TSTR_F1_FakePrev: 0.049587
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.554722
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=N

### v21_blend_lateg_out035_d35_r010_m005

- config: model_w=0.75, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=late, guidance_power=2.0, recon=0.10, moment=0.05
- hard_wins: 0/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 17.03
- infer_seconds: 2.28
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.008795
- Wasserstein: 1.628341
- CMD: 0.004550
- TSTR_AUC: 0.564153
- TSTR_PR_AUC: 0.036077
- TSTR_F1: 0.069444
- TSTR_F1_RealPrev: 0.034632
- TSTR_F1_FakePrev: 0.041494
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.555588
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=N

### v21_anchor_lateg_out035_d35_r010_m005

- config: model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=late, guidance_power=2.0, recon=0.10, moment=0.05
- hard_wins: 0/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 18.68
- infer_seconds: 2.29
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.007875
- Wasserstein: 1.628814
- CMD: 0.004842
- TSTR_AUC: 0.563056
- TSTR_PR_AUC: 0.033776
- TSTR_F1: 0.029630
- TSTR_F1_RealPrev: 0.026549
- TSTR_F1_FakePrev: 0.032787
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.559604
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=N

### v21_lowguide_lateg_out035_d35_r015_m005

- config: model_w=1.00, guidance=1.25, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=late, guidance_power=2.0, recon=0.15, moment=0.05
- hard_wins: 0/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 17.04
- infer_seconds: 2.28
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.008172
- Wasserstein: 1.621307
- CMD: 0.004592
- TSTR_AUC: 0.561867
- TSTR_PR_AUC: 0.034720
- TSTR_F1: 0.060201
- TSTR_F1_RealPrev: 0.052632
- TSTR_F1_FakePrev: 0.049383
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.557703
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=N

## Interpretation

- No searched V2 configuration yet dominates all current table metrics simultaneously; continue with targeted search around the best candidate rather than broad expansion.
- The best searched configuration still fails the formal prevalence-aware gate, so do not escalate to multi-seed until the gate is recovered.