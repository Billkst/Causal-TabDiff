# CausalTabDiff Final Gated Rehearsal

- Generated at: 2026-03-07T10:19:21.642789
- Seed: 7
- Device: cuda
- Train size: 8192
- Eval size: 4096
- Batch size: 256
- Epochs: 8
- Diffusion steps: 20
- Treatment source: cigsmok
- Outcome loss weight: 1.0
- Outcome rank loss weight: 0.0
- Sample model score weight: 0.75
- Sample guidance scale: 1.0
- Params(M): 0.0349
- Train seconds: 11.78
- Inference seconds: 1.53
- Real Y rate: 0.0286
- Fake Y rate: 0.0273

## Formalized Metrics

- ATE_Bias: 0.002906
- Wasserstein: 0.670444
- CMD: 0.008021
- TSTR_AUC: 0.557635
- TSTR_PR_AUC: 0.035682
- TSTR_F1: 0.000000
- TSTR_F1_RealPrev: 0.000000
- TSTR_F1_FakePrev: 0.000000

## Gate Criteria

- TSTR_AUC >= 0.58
- TSTR_F1 >= 0.05
- TSTR_F1_RealPrev >= 0.05
- TSTR_PR_AUC >= 0.04
- Gate decision: HOLD

## Interpretation

- This rehearsal does not clear the minimal pre-full gate under the formalized prevalence-aware metric bundle.
- The project should remain in controlled mode rather than moving to full-scale training.