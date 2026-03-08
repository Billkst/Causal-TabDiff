---
description: In the noleak NLST setting, treat `cigsmok` as the only currently admissible real treatment for Causal-TabDiff, forbid post-screen descendants as treatment, and require seed-level smoke gating before any larger experiment.
triggers:
  - user asks to run or validate CausalTabDiff under noleak mode
  - task mentions `alpha_target`, `cigsmok`, `can_scr`, `scr_res0`, `scr_iso0`, or `dataset_metadata_noleak.json`
  - before any medium/large CausalTabDiff training or evaluation
confidence: true
---

# Instinct: Gate CausalTabDiff with Real Treatment and Seed Stability

## Mistake Captured
The original CausalTabDiff pipeline used a synthetic uniform `alpha_target`, random placeholder `Y`, and no seed-level stability gate. This made early successes uninterpretable in the noleak clinical setting.

## Hard Causal Rules
1. Under the current noleak schema, `cigsmok` is the **only** admissible real treatment/exposure candidate.
2. Never promote `scr_res0`, `scr_iso0`, `scr_days0`, or `can_scr` into treatment:
   - `scr_res0` / `scr_iso0`: post-screen readouts / mediator-like variables
   - `scr_days0`: post-randomization process variable
   - `can_scr`: explicit outcome descendant / leakage
3. Treat `race`, `gender`, and `age` as baseline covariates/confounders, not treatments.

## Engineering Lessons Captured
1. `Y` must not remain outside the model contract; keep an in-model outcome head.
2. Wrapper-level random `Y` is forbidden.
3. For early directional stability smoke tests, the sample-time `50/50` blend of glue score and in-model score was the safest readout among the first compared options.
4. Under the later formalized efficacy gate (`TSTR_AUC` / `TSTR_PR_AUC` / prevalence-aware `F1`), the current best candidate default sampling configuration is `sample_model_score_weight=0.75` with `sample_guidance_scale=1.0`.
5. Simple outcome-loss reweighting alone does **not** fix the remaining reversing seeds.

## Experiment Gating Rule
Before any larger-than-smoke CausalTabDiff run, require all of the following:
1. `DATASET_METADATA_PATH=src/data/dataset_metadata_noleak.json`
2. `ALPHA_TREATMENT_COLUMN=cigsmok`
3. model/wrapper tests green
4. seed-level directional smoke confirmation already passed at least a majority threshold
5. for pre-experiment readiness, prefer a stricter multi-seed check rather than jumping directly to full training

## Current Stability Snapshot
- model-only 10-seed strict smoke: `8/10` seeds in the expected direction
- blend readout 10-seed source ablation: `10/10` seeds in the expected direction
- therefore, `50/50` blend remains the safest early directional preview choice, while `0.75 model score / 1.0 guidance` is the stronger candidate default for later formalized efficacy checks

## BAD Pattern
```bash
# BAD: synthetic treatment or leaked descendants
ALPHA_TREATMENT_COLUMN=can_scr python run_experiment.py
```

## GOOD Pattern
```bash
# GOOD: noleak + real admissible treatment + controlled preview
export DATASET_METADATA_PATH=src/data/dataset_metadata_noleak.json
export ALPHA_TREATMENT_COLUMN=cigsmok
conda run -n causal_tabdiff python scripts/<controlled_preview>.py
```

## Action Rule
1. First bind real treatment to `cigsmok`.
2. Then validate seed stability on smoke-scale runs.
3. Only after that, run a medium-scale controlled rehearsal.
4. Do not claim formal causal readiness until instability is materially reduced or explicitly bounded.
