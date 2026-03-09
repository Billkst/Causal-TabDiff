# CausalTabDiff Single-Model Pilot

- Generated at: 2026-03-07T09:41:37.211602
- Seed: 7
- Device: cuda
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Outcome loss weight: 1.0
- Sample model score weight: 0.5
- Params(M): 0.0218
- Train seconds: 9.52
- Inference seconds: 1.25
- Real Y rate: 0.0286
- Fake Y rate: 0.0273

## Metrics

- ATE_Bias: 0.014587
- Wasserstein: 0.665603
- CMD: 0.009883
- TSTR_AUC: 0.553381
- TSTR_F1: 0.000000

## Interpretation

- This is a single-model pilot, not a full-scale experiment.
- It intentionally remains inside the noleak + cigsmok treatment gate.
- The result is only suitable for deciding whether a larger pilot is justified.