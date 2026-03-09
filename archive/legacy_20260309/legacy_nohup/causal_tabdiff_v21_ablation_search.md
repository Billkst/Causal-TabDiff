# CausalTabDiff V2 Champion Search

- Generated at: 2026-03-07T13:17:51.799315
- Device: cuda
- Seed: 7
- Search profile: v21_ablate
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Metadata path: src/data/dataset_metadata_noleak.json
- Baseline report path: /home/UserData/ljx/Project_2/Causal-TabDiff/markdown_report.md
- Total wall time seconds: 91.25

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

- Name: ablate_recon_only_oldguide
- Hard wins: 1/5
- Table all-win: False
- Formal gate pass: False
- Params(M): 0.025456 (acceptable)
- AvgInfer(ms/sample): 0.564238 (acceptable)
- ATE_Bias: 0.031045
- Wasserstein: 0.555312
- CMD: 0.001811
- TSTR_AUC: 0.591388
- TSTR_PR_AUC: 0.036013
- TSTR_F1: 0.067454
- TSTR_F1_RealPrev: 0.049180
- TSTR_F1_FakePrev: 0.049180

## All Config Results

### ablate_recon_only_oldguide

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.10, moment=0.05
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.90
- infer_seconds: 2.31
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.031045
- Wasserstein: 0.555312
- CMD: 0.001811
- TSTR_AUC: 0.591388
- TSTR_PR_AUC: 0.036013
- TSTR_F1: 0.067454
- TSTR_F1_RealPrev: 0.049180
- TSTR_F1_FakePrev: 0.049180
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.564238
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

### ablate_lightrecon_oldguide

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.43
- infer_seconds: 2.30
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.029448
- Wasserstein: 0.554596
- CMD: 0.001870
- TSTR_AUC: 0.588353
- TSTR_PR_AUC: 0.036463
- TSTR_F1: 0.078712
- TSTR_F1_RealPrev: 0.052863
- TSTR_F1_FakePrev: 0.049793
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.560693
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=Y

### ablate_noisehead_only_oldguide

- config: use_noise_head=True, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.00, moment=0.00
- hard_wins: 0/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 17.55
- infer_seconds: 2.39
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.032874
- Wasserstein: 1.563631
- CMD: 0.004994
- TSTR_AUC: 0.567881
- TSTR_PR_AUC: 0.035940
- TSTR_F1: 0.067797
- TSTR_F1_RealPrev: 0.061947
- TSTR_F1_FakePrev: 0.058091
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.582823
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=N

### ablate_lateg_only

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=late, guidance_power=2.0, recon=0.00, moment=0.00
- hard_wins: 0/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.10
- infer_seconds: 2.27
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.011172
- Wasserstein: 0.557675
- CMD: 0.002016
- TSTR_AUC: 0.550386
- TSTR_PR_AUC: 0.032198
- TSTR_F1: 0.055838
- TSTR_F1_RealPrev: 0.044248
- TSTR_F1_FakePrev: 0.040486
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.554491
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=N

### ablate_noisehead_lateg

- config: use_noise_head=True, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=late, guidance_power=2.0, recon=0.00, moment=0.00
- hard_wins: 0/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=acceptable
- train_seconds: 15.18
- infer_seconds: 1.72
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.011173
- Wasserstein: 1.561038
- CMD: 0.004332
- TSTR_AUC: 0.524992
- TSTR_PR_AUC: 0.029215
- TSTR_F1: 0.016327
- TSTR_F1_RealPrev: 0.017241
- TSTR_F1_FakePrev: 0.016393
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.420823
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=N, TSTR_F1=N

## Interpretation

- No searched V2 configuration yet dominates all current table metrics simultaneously; continue with targeted search around the best candidate rather than broad expansion.
- The best searched configuration still fails the formal prevalence-aware gate, so do not escalate to multi-seed until the gate is recovered.