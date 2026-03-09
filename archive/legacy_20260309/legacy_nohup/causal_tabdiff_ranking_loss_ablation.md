# CausalTabDiff Ranking Loss Ablation

- Generated at: 2026-03-07T10:08:11.646016
- Seed: 7
- Device: cuda
- Train size: 4096
- Eval size: 2048
- Batch size: 256
- Epochs: 6
- Diffusion steps: 20
- Treatment source: cigsmok
- Fixed sample model score weight: 0.75
- Fixed guidance scale: 1.0

## Ranked Results

### Rank 1: rank_loss_weight=0.00

- train_seconds: 4.94
- params_m: 0.0218
- TSTR_AUC: 0.658820
- TSTR_PR_AUC: 0.075509
- TSTR_F1: 0.109091
- TSTR_F1_RealPrev: 0.116667
- TSTR_F1_FakePrev: 0.118519
- ATE_Bias: 0.021475
- Wasserstein: 0.890634
- CMD: 0.011406

### Rank 2: rank_loss_weight=0.25

- train_seconds: 3.50
- params_m: 0.0218
- TSTR_AUC: 0.591356
- TSTR_PR_AUC: 0.038339
- TSTR_F1: 0.048951
- TSTR_F1_RealPrev: 0.033058
- TSTR_F1_FakePrev: 0.044776
- ATE_Bias: 0.029023
- Wasserstein: 0.886562
- CMD: 0.012506

### Rank 3: rank_loss_weight=1.00

- train_seconds: 3.46
- params_m: 0.0218
- TSTR_AUC: 0.573483
- TSTR_PR_AUC: 0.037688
- TSTR_F1: 0.080645
- TSTR_F1_RealPrev: 0.000000
- TSTR_F1_FakePrev: 0.028571
- ATE_Bias: 0.030218
- Wasserstein: 0.893270
- CMD: 0.012170

### Rank 4: rank_loss_weight=0.50

- train_seconds: 3.85
- params_m: 0.0218
- TSTR_AUC: 0.576647
- TSTR_PR_AUC: 0.035041
- TSTR_F1: 0.036232
- TSTR_F1_RealPrev: 0.016393
- TSTR_F1_FakePrev: 0.030075
- ATE_Bias: 0.028752
- Wasserstein: 0.888187
- CMD: 0.012439

## Recommendation

- Best ranking-loss setting is rank_loss_weight=0.00 with TSTR_AUC=0.658820, TSTR_PR_AUC=0.075509, TSTR_F1=0.109091, TSTR_F1_RealPrev=0.116667.
- This is the first minimal ranking-oriented change that clears the local heuristic and should be prioritized for the next gated rehearsal.