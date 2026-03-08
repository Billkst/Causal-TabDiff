# CausalTabDiff Readout + Guidance Ablation

- Generated at: 2026-03-07T10:03:38.821610
- Seed: 7
- Device: cuda
- Train size: 4096
- Eval size: 2048
- Batch size: 256
- Epochs: 6
- Diffusion steps: 20
- Treatment source: cigsmok
- Outcome loss weight: 1.0
- Params(M): 0.0218
- Train seconds: 4.77

## Ranked Configurations

### Rank 1: model_w=0.75, guidance_scale=1.0

- gate_like_pass: False
- TSTR_AUC: 0.584369
- TSTR_PR_AUC: 0.032290
- TSTR_F1: 0.062500
- TSTR_F1_RealPrev: 0.049180
- TSTR_F1_FakePrev: 0.049180
- ATE_Bias: 0.047212
- Wasserstein: 0.978012
- CMD: 0.006439

### Rank 2: model_w=0.75, guidance_scale=3.0

- gate_like_pass: False
- TSTR_AUC: 0.537861
- TSTR_PR_AUC: 0.030402
- TSTR_F1: 0.052830
- TSTR_F1_RealPrev: 0.020000
- TSTR_F1_FakePrev: 0.020000
- ATE_Bias: 0.060921
- Wasserstein: 0.942367
- CMD: 0.004408

### Rank 3: model_w=0.75, guidance_scale=2.0

- gate_like_pass: False
- TSTR_AUC: 0.556350
- TSTR_PR_AUC: 0.026559
- TSTR_F1: 0.020725
- TSTR_F1_RealPrev: 0.000000
- TSTR_F1_FakePrev: 0.000000
- ATE_Bias: 0.061503
- Wasserstein: 0.991851
- CMD: 0.007304

### Rank 4: model_w=0.50, guidance_scale=3.0

- gate_like_pass: False
- TSTR_AUC: 0.543231
- TSTR_PR_AUC: 0.026544
- TSTR_F1: 0.000000
- TSTR_F1_RealPrev: 0.020202
- TSTR_F1_FakePrev: 0.020619
- ATE_Bias: 0.054961
- Wasserstein: 0.979567
- CMD: 0.006607

### Rank 5: model_w=0.25, guidance_scale=1.0

- gate_like_pass: False
- TSTR_AUC: 0.525370
- TSTR_PR_AUC: 0.025914
- TSTR_F1: 0.000000
- TSTR_F1_RealPrev: 0.019802
- TSTR_F1_FakePrev: 0.019802
- ATE_Bias: 0.008238
- Wasserstein: 1.033165
- CMD: 0.007101

### Rank 6: model_w=0.50, guidance_scale=1.0

- gate_like_pass: False
- TSTR_AUC: 0.486156
- TSTR_PR_AUC: 0.024768
- TSTR_F1: 0.000000
- TSTR_F1_RealPrev: 0.000000
- TSTR_F1_FakePrev: 0.000000
- ATE_Bias: 0.038304
- Wasserstein: 1.039387
- CMD: 0.004492

### Rank 7: model_w=0.50, guidance_scale=2.0

- gate_like_pass: False
- TSTR_AUC: 0.514921
- TSTR_PR_AUC: 0.024435
- TSTR_F1: 0.000000
- TSTR_F1_RealPrev: 0.000000
- TSTR_F1_FakePrev: 0.000000
- ATE_Bias: 0.034674
- Wasserstein: 0.987723
- CMD: 0.007090

### Rank 8: model_w=0.25, guidance_scale=2.0

- gate_like_pass: False
- TSTR_AUC: 0.495391
- TSTR_PR_AUC: 0.024226
- TSTR_F1: 0.000000
- TSTR_F1_RealPrev: 0.034188
- TSTR_F1_FakePrev: 0.034188
- ATE_Bias: 0.020006
- Wasserstein: 0.964817
- CMD: 0.005628

### Rank 9: model_w=0.25, guidance_scale=3.0

- gate_like_pass: False
- TSTR_AUC: 0.457668
- TSTR_PR_AUC: 0.022920
- TSTR_F1: 0.000000
- TSTR_F1_RealPrev: 0.000000
- TSTR_F1_FakePrev: 0.000000
- ATE_Bias: 0.000605
- Wasserstein: 0.971038
- CMD: 0.005471

## Recommendation

- No sampled configuration clears the local gate-like heuristic. Best candidate is model_w=0.75, guidance_scale=1.0, but TSTR_AUC=0.584369, TSTR_PR_AUC=0.032290, TSTR_F1_RealPrev=0.049180 remain below the desired joint bar.
- This is a minimal structure/sampling optimization only; it does not authorize full-scale training by itself.