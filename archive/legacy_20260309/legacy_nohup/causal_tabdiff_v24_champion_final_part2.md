# CausalTabDiff V2 Champion Search

- Generated at: 2026-03-07T14:08:01.989238
- Device: cuda
- Seed: 7
- Search profile: v24_champion_final
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Metadata path: src/data/dataset_metadata_noleak.json
- Baseline report path: /home/UserData/ljx/Project_2/Causal-TabDiff/markdown_report.md
- Total wall time seconds: 66.65

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

- Name: v24_focal_blend_50
- Hard wins: 2/5
- Table all-win: False
- Formal gate pass: False
- Params(M): 0.025456 (acceptable)
- AvgInfer(ms/sample): 0.771573 (borderline)
- ATE_Bias: 0.007219
- Wasserstein: 0.556814
- CMD: 0.001531
- TSTR_AUC: 0.602179
- TSTR_PR_AUC: 0.037394
- TSTR_F1: 0.081448
- TSTR_F1_RealPrev: 0.035088
- TSTR_F1_FakePrev: 0.033195

## All Config Results

### v24_focal_blend_50

- config: use_noise_head=False, model_w=0.50, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 2/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=borderline
- train_seconds: 29.70
- infer_seconds: 3.16
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.007219
- Wasserstein: 0.556814
- CMD: 0.001531
- TSTR_AUC: 0.602179
- TSTR_PR_AUC: 0.037394
- TSTR_F1: 0.081448
- TSTR_F1_RealPrev: 0.035088
- TSTR_F1_FakePrev: 0.033195
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.771573
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=Y

### v24_focal_internal_100

- config: use_noise_head=False, model_w=1.00, guidance=1.50, smooth=0.10, cf=0.05, outcome_w=0.35, epochs=10, diffusion_steps=35, guidance_schedule=constant, guidance_power=2.0, recon=0.03, moment=0.00
- hard_wins: 1/5
- table_all_win: False
- formal_gate_pass: False
- soft_status: Params(M)=acceptable, AvgInfer(ms/sample)=weak
- train_seconds: 29.88
- infer_seconds: 3.43
- real_y_rate: 0.0276
- fake_y_rate: 0.0312
- ATE_Bias: 0.012012
- Wasserstein: 0.556814
- CMD: 0.001531
- TSTR_AUC: 0.598594
- TSTR_PR_AUC: 0.036490
- TSTR_F1: 0.055172
- TSTR_F1_RealPrev: 0.026316
- TSTR_F1_FakePrev: 0.032922
- Params(M): 0.025456
- AvgInfer(ms/sample): 0.837863
- hard_metric_wins: ATE_Bias=N, Wasserstein=N, CMD=N, TSTR_AUC=Y, TSTR_F1=N

## Interpretation

- No searched V2 configuration yet dominates all current table metrics simultaneously; continue with targeted search around the best candidate rather than broad expansion.
- The best searched configuration still fails the formal prevalence-aware gate, so do not escalate to multi-seed until the gate is recovered.