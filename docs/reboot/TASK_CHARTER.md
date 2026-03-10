# Task Charter: Dual-Layer Task Reformulation

**Version**: 1.0  
**Date**: 2026-03-10  
**Status**: Active

---

## 1. Executive Summary

This document establishes the **mandatory dual-layer task definition** for the Causal-TabDiff project, replacing the deprecated pseudo-temporal full covariate trajectory generation task with a scientifically grounded formulation.

**Core Principle**: We preserve the trajectory generation narrative from the thesis proposal while adapting it to what the data actually supports—**risk trajectory generation** rather than full future covariate trajectory generation.

---

## 2. Why the Old Task Definition Failed

### 2.1 The Deprecated Task

**Old Claim**: Generate complete future tabular trajectories with full covariate supervision from T3 to T7.

**Critical Flaws**:

1. **Data Reality Mismatch**: NLST only provides genuine observed covariates at T0, T1, T2. Future time points (T3-T7) only contain outcome information (cancer diagnosis year), not complete feature observations.

2. **Pseudo-Temporal Replication**: Previous implementation artificially replicated T0 static features across T1-T7 to create fake "temporal sequences", violating causal observation logic.

3. **Supervision Impossibility**: Cannot supervise generation of future covariates that were never measured in the dataset.

4. **Causal Violation**: Treating replicated static features as temporal evolution introduces spurious temporal patterns that don't reflect real disease progression.

### 2.2 Why This Matters

- **Scientific Integrity**: Cannot claim to model temporal dynamics using fabricated sequences
- **Reviewer Credibility**: Pseudo-temporal tricks would be immediately flagged in peer review
- **Clinical Validity**: Predictions based on fake trajectories have no real-world applicability

---

## 3. The Dual-Layer Task Definition (Mandatory)

### Layer 1: Primary Validation Task (Must Land, Must Evaluate)

**Task**: Landmark-conditioned 2-year first-lung-cancer risk prediction

**Definition**:
- **Sample Unit**: (person, landmark) pair
- **Landmark**: t ∈ {T0, T1, T2}
- **Input**: All genuinely observed historical information up to and including landmark t
- **Output**: Binary prediction of first lung cancer diagnosis within 2 years

**Label Definition**:

```
y_{i,t}^{(2)} = 1{t < cancyr_i ≤ t+2}
```

Where:
- `cancyr_i`: Year of first cancer diagnosis for person i (0 if never diagnosed)
- `t`: Landmark time point
- Event window: (t, t+2]

**Exclusion Rules** (MANDATORY):

1. **Pre-existing Cancer**: If `cancyr_i ≤ t`, exclude this (person, landmark) sample entirely
   - Rationale: Cannot predict "first diagnosis" for someone already diagnosed
   - Old bug: Treated `cancyr > 0.5` as negative class, incorrectly including T0 samples of patients diagnosed at T0

2. **Insufficient Follow-up**: If censoring occurs before t+2 and no event observed, handle as censored (not as negative)

**Why 2-Year Window**:
- Clinically actionable timeframe for screening intervals
- Sufficient follow-up data in NLST
- Aligns with Lung-RADS and clinical guidelines

### Layer 2: Trajectory Narrative Goal (Must Preserve, Must Reformulate)

**Task**: Future risk trajectory generation

**What Changed**:
- **Old**: Generate future T3-T7 complete covariate tables
- **New**: Generate future yearly risk/hazard trajectories

**Definition**:

The model outputs a **risk trajectory** rather than covariate trajectory:

```
Risk Trajectory: [r_{t+1}, r_{t+2}, ..., r_{T_max}]
```

Or equivalently, hazard trajectory:

```
Hazard Trajectory: [h_{t+1}, h_{t+2}, ..., h_{T_max}]
```

Where:
- `r_τ`: Cumulative risk of cancer by year τ
- `h_τ`: Instantaneous hazard (yearly incidence rate) at year τ
- `T_max`: Maximum follow-up (can extend to T7 since outcome data exists)

**Relationship to 2-Year Task**:

The 2-year risk is a **derived aggregation** of the trajectory:

```
P(event within 2 years | landmark t) = 1 - (1 - h_{t+1})(1 - h_{t+2})
```

Or using cumulative risk:

```
P(event within 2 years | landmark t) = r_{t+2} - r_t
```

**Why This Works**:

1. **Data Support**: Outcome years (cancyr) are observed through T7, enabling risk trajectory supervision
2. **Trajectory Preservation**: Still generates temporal sequences, maintaining thesis narrative
3. **Scientific Validity**: Risk trajectories are standard in survival analysis and clinical prediction
4. **Generative Flavor**: Diffusion models naturally generate continuous risk distributions over time

---

## 4. Mathematical Formulation

### 4.1 Sample Space

**Person-Landmark Sample**:

```
S_{i,t} = {
    pid: i,
    landmark: t,
    X_{i,≤t}: [X_{i,0}, X_{i,1}, ..., X_{i,t}],  // Observed history
    y_{i,t}^{(2)}: 1{t < cancyr_i ≤ t+2},        // 2-year label
    R_{i,t}: [r_{t+1}, ..., r_{T_max}],          // Risk trajectory target
    cancyr_i: cancer diagnosis year (if any)
}
```

### 4.2 Input Constraints (Causal Observation)

**Strict Temporal Ordering**:

For landmark t, input X_{i,≤t} contains ONLY:
- Baseline features at T0
- Screening results at T0, T1, ..., t (if t ≥ 1)
- Abnormality findings at T0, T1, ..., t
- Change features comparing consecutive visits up to t

**Forbidden Inputs**:
- Any feature from time > t
- Cancer diagnosis information (cancyr, clinical_stag, path_stag)
- Any feature derived from future outcomes

### 4.3 Variable-Length History

**Real Temporal Sequences** (NOT pseudo-replication):

- T0 sample: history length = 1, contains [X_{i,0}]
- T1 sample: history length = 2, contains [X_{i,0}, X_{i,1}]
- T2 sample: history length = 3, contains [X_{i,0}, X_{i,1}, X_{i,2}]

**Handling Variable Length**:
- Padding with explicit mask
- Length tensor for attention mechanisms
- NO replication of static features to fill time axis

### 4.4 Risk Trajectory Target Construction

**Option A: Discrete Yearly Hazard**

```
h_{i,τ} = 1{cancyr_i = τ}  // Indicator for diagnosis in year τ
```

**Option B: Smooth Risk Curve**

```
r_{i,τ} = 1{cancyr_i ≤ τ}  // Cumulative indicator
```

**Option C: Survival-Based Encoding**

```
S_{i,τ} = 1{cancyr_i > τ}  // Survival function
h_{i,τ} = (S_{i,τ-1} - S_{i,τ}) / S_{i,τ-1}  // Hazard from survival
```

**Implementation Choice**: Start with Option A (discrete hazard), can upgrade to smooth curves later.

---

## 5. What We Can Claim

### 5.1 Valid Claims

✅ **Landmark-based 2-year lung cancer risk prediction**
- Standard clinical prediction task
- Directly evaluable with AUROC, AUPRC, calibration

✅ **Future risk trajectory generation**
- Generates yearly hazard/risk sequences
- Enables dynamic risk monitoring
- Supports "what-if" counterfactual scenarios

✅ **Temporal modeling with genuine short histories**
- Models real temporal evolution from T0→T1→T2
- Captures screening dynamics and nodule progression

✅ **Causal guidance for risk trajectory**
- Alpha_target guides trajectory generation
- Enables treatment effect estimation on risk curves

### 5.2 Invalid Claims (DO NOT MAKE)

❌ **Full future covariate trajectory generation with complete supervision**
- Data does not support this

❌ **Predicting future screening results or nodule characteristics**
- Only outcome (cancer/no cancer) is observed in future

❌ **Long-term temporal sequences (T0-T7) with complete features**
- Only T0-T2 have genuine feature observations

---

## 6. Communication Strategy for Advisors/Thesis

### 6.1 Framing for Thesis Committee

**Narrative**:

> "Following the thesis proposal, we implement trajectory generation for lung cancer risk prediction. However, given the data structure of NLST (complete covariates only at T0-T2, outcome tracking through T7), we formulate trajectory generation as **risk trajectory modeling** rather than full covariate trajectory generation.
>
> This approach:
> 1. Preserves the trajectory generation framework from the proposal
> 2. Aligns with clinical practice (risk curves are standard in oncology)
> 3. Enables the same causal inference goals (counterfactual risk under interventions)
> 4. Provides a concrete 2-year prediction task for validation
>
> The dual-layer formulation allows us to maintain the generative/trajectory narrative while ensuring rigorous empirical validation."

### 6.2 Key Talking Points

1. **Not a Downgrade**: Risk trajectories are clinically more relevant than raw covariate trajectories
2. **Still Generative**: Diffusion model generates continuous risk distributions
3. **Still Causal**: Alpha_target guidance enables counterfactual risk estimation
4. **More Rigorous**: Avoids pseudo-temporal artifacts, uses only genuine observations
5. **Thesis-Aligned**: Fulfills "trajectory generation" goal with scientifically valid formulation

### 6.3 If Asked: "Why Not Full Covariate Trajectories?"

**Honest Answer**:

> "NLST's data structure provides complete covariate measurements only during active screening (T0-T2). Post-screening follow-up (T3-T7) tracks cancer outcomes but not detailed covariates. Attempting to supervise full covariate generation beyond T2 would require either:
> 1. Imputation (introduces bias)
> 2. Synthetic data (not grounded in real observations)
> 3. Pseudo-temporal replication (violates causal logic)
>
> Risk trajectory generation leverages the outcome data that IS available (cancer diagnosis years through T7) while respecting the observational structure of the data."

---

## 7. Data Requirements Summary

### 7.1 Minimum Viable Dataset

**Required Tables**:
1. Person baseline (prsn): demographics, smoking history, comorbidities
2. Screening summary (screen): per-visit screening metadata
3. Abnormality summary (ctab, ctabc): nodule findings and changes
4. Outcome table: cancyr, censoring info

**Required Fields**:
- `pid`: person identifier
- `landmark`: {0, 1, 2}
- `cancyr`: year of first cancer diagnosis (0 if never)
- `X_baseline`: age, gender, smoking, etc.
- `X_temporal`: screening results, nodule features per visit

### 7.2 Derived Fields

**Must Construct**:
- `y_2year`: binary label for 2-year risk
- `risk_trajectory`: [h_1, h_2, ..., h_7] or equivalent
- `history_length`: actual number of visits up to landmark
- `mask`: padding mask for variable-length sequences

---

## 8. Success Criteria

### 8.1 Layer 1 (Primary Task)

**Must Achieve**:
- AUROC > 0.70 on held-out test set
- AUPRC significantly above prevalence baseline
- Calibration slope near 1.0, intercept near 0
- Confusion matrix with reasonable sensitivity/specificity trade-off

### 8.2 Layer 2 (Trajectory Quality)

**Must Demonstrate**:
- Generated risk trajectories are monotonically increasing (or properly shaped)
- Counterfactual trajectories respond sensibly to alpha_target changes
- Trajectory-derived 2-year risk matches direct prediction

### 8.3 Causal Validity

**Must Show**:
- ATE_Bias lower than baselines
- Guidance mechanism produces interpretable risk modulation

---

## 9. Revision History

- **v1.0 (2026-03-10)**: Initial charter defining dual-layer task reformulation

---

## 10. Approval & Enforcement

This charter is **mandatory** for all subsequent development. Any code, experiment, or documentation that violates these definitions must be rejected.

**Enforcement Checkpoints**:
1. Data module must implement landmark-based sampling
2. Model output must include risk trajectory
3. Evaluation must report both Layer 1 and Layer 2 metrics
4. No pseudo-temporal replication allowed in any form
