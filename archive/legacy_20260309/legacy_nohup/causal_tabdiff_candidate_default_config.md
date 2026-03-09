# CausalTabDiff Candidate Default Sampling Configuration

- Generated at: 2026-03-07T10:12:00+00:00
- Scope: candidate operational default update for controlled no-leak runs
- Treatment gate: `cigsmok`
- Metadata gate: `src/data/dataset_metadata_noleak.json`

## Candidate Default

For controlled `CausalTabDiff` runs, the current candidate default sampling configuration is:

- `sample_model_score_weight = 0.75`
- `sample_guidance_scale = 1.0`
- `outcome_rank_loss_weight = 0.0`

## Why this became the candidate default

### 1. Minimal readout + guidance ablation
来源：[logs/testing/causal_tabdiff_readout_guidance_ablation.md](logs/testing/causal_tabdiff_readout_guidance_ablation.md)

Best configuration in the 3x3 sample-time grid:
- `model_w = 0.75`
- `guidance_scale = 1.0`

Observed metrics on the controlled split:
- `TSTR_AUC = 0.584369`
- `TSTR_PR_AUC = 0.032290`
- `TSTR_F1 = 0.062500`
- `TSTR_F1_RealPrev = 0.049180`

Interpretation:
- relative to the former default-like setting (`0.50`, `2.0`), this configuration improved ranking quality and fixed-threshold F1
- stronger reliance on model score plus weaker guidance was better than pushing guidance harder

### 2. Ranking-loss ablation
来源：[logs/testing/causal_tabdiff_ranking_loss_ablation.md](logs/testing/causal_tabdiff_ranking_loss_ablation.md)

With the sampling side fixed to the above candidate:
- `rank_loss_weight = 0.0` was best
- positive ranking-loss weights degraded the main formalized metrics

Best small controlled result under this setting:
- `TSTR_AUC = 0.658820`
- `TSTR_PR_AUC = 0.075509`
- `TSTR_F1 = 0.109091`
- `TSTR_F1_RealPrev = 0.116667`

Interpretation:
- the helpful part was the sample-time configuration
- the added pairwise ranking loss is **not** the current fix

### 3. Final gated rehearsal recheck
来源：[logs/testing/causal_tabdiff_final_gated_rehearsal.md](logs/testing/causal_tabdiff_final_gated_rehearsal.md)

When the candidate sampling configuration was rechecked on the larger rehearsal:
- `TSTR_AUC = 0.605371`
- `TSTR_PR_AUC = 0.039527`
- `TSTR_F1 = 0.000000`
- `TSTR_F1_RealPrev = 0.053435`
- final decision remained `HOLD`

Interpretation:
- candidate defaults are better than the previous operational defaults
- but they still do **not** authorize full-scale training

## Decision

**Adopt this as the new candidate operational default for controlled runs only.**

This means:
- wrapper-level default sampling settings may use `0.75 / 1.0`
- formal rehearsals may use these values as the first-choice controlled configuration
- but the project remains under gate control

## Non-decision

This report does **not** mean:
- full-scale training is approved
- current architecture is clinically ready
- prevalence-aware performance is fully stabilized

## Recommended use

Use the candidate default configuration for:
1. future controlled pilot reruns
2. formal gated rehearsals
3. any next-stage outcome-head/readout redesign comparisons

Do not use this report as authorization for unattended full-scale training.
