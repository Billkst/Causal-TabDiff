# CausalTabDiff Controlled Preview

- Generated at: 2026-03-07T09:34:15.972735
- Seed: 7
- Train size: 2048
- Eval size: 1024
- Epochs: 3
- Diffusion steps: 10
- Treatment source: cigsmok
- Params(M): 0.0218
- Real Y rate: 0.0225
- Fake Y rate: 0.0234

## Metrics

- ATE_Bias: 0.041025
- Wasserstein: 1.276005
- CMD: 0.003729
- TSTR_AUC: 0.610303
- TSTR_F1: 0.082902

## Interpretation

- This is a controlled preview only, not a full experiment.
- It uses the admitted noleak treatment `cigsmok` and the currently most stable `blend` score path.
- Results should be used only to decide whether the model is ready for a larger single-model pilot run.