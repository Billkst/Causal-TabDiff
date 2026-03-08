# CausalTabDiff Final Gated Rehearsal

- Generated at: 2026-03-07T10:32:55.062204
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
- Use trajectory risk head: True
- Risk smoothness weight: 0.1
- Counterfactual consistency weight: 0.05
- Sample use trajectory risk: True
- Sample model score weight: 1.0
- Sample guidance scale: 1.0
- Params(M): 0.0253
- Train seconds: 13.56
- Inference seconds: 1.40
- Real Y rate: 0.0286
- Fake Y rate: 0.0273

## Formalized Metrics

- ATE_Bias: 0.004861
- Wasserstein: 0.679477
- CMD: 0.008644
- TSTR_AUC: 0.666018
- TSTR_PR_AUC: 0.053880
- TSTR_F1: 0.115646
- TSTR_F1_RealPrev: 0.050847
- TSTR_F1_FakePrev: 0.051948

## Gate Criteria

- TSTR_AUC >= 0.58
- TSTR_F1 >= 0.05
- TSTR_F1_RealPrev >= 0.05
- TSTR_PR_AUC >= 0.04
- Gate decision: PASS

## Interpretation

- This rehearsal clears the minimal pre-full gate under the formalized prevalence-aware metric bundle.
- Full-scale training is not auto-approved, but a single-model full run can now be considered for approval.