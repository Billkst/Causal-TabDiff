# Causal Evaluation Protocol

**Version**: 1.0  
**Date**: 2026-03-10  
**Status**: Active

---

## 1. Overview

This document defines how to evaluate the causal validity of Causal-TabDiff's trajectory generation, distinguishing between what can be rigorously evaluated on real data vs what requires semi-synthetic benchmarks.

---

## 2. Core Causal Concepts

### 2.1 Treatment Definition

**Alpha_Target (α)**:
- Continuous causal guidance variable ∈ [0, 1]
- Represents intervention intensity or exposure level
- Examples: smoking cessation intensity, screening frequency

**In NLST Context**:
- Can use actual treatment variables (e.g., smoking status, pack-years)
- Or synthetic uniform random α for counterfactual exploration

### 2.2 Outcome Definition

**Primary Outcome**: Risk trajectory R_t = [r_{t+1}, ..., r_{T_max}]

**Derived Outcome**: 2-year cancer risk (aggregation of trajectory)

### 2.3 Causal Estimand

**Average Treatment Effect (ATE)**:
```
ATE = E[Y(α=1)] - E[Y(α=0)]
```

Where Y(α) is potential outcome under treatment level α.

**Conditional ATE (CATE)**:
```
CATE(x) = E[Y(α=1) | X=x] - E[Y(α=0) | X=x]
```

Heterogeneous treatment effects conditional on covariates X.

---

## 3. ATE_Bias Computation

### 3.1 Ground Truth Estimation (via EconML)

**Method**: Double Machine Learning (DML)

```python
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Estimate true ATE from real data
dml = LinearDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestClassifier()
)

dml.fit(
    Y=real_outcomes,           # Observed outcomes
    T=real_treatments,         # Observed treatments (alpha_target)
    X=real_covariates          # Confounders
)

ate_true = dml.ate(X=real_covariates)
```

**Interpretation**: This is our best estimate of the true causal effect in the data.

### 3.2 Model-Based ATE Estimation

**Method**: Generate counterfactual trajectories

```python
# Generate under α=0 (no treatment)
trajectories_0 = model.generate(X_history, alpha_target=0.0)
outcomes_0 = derive_2year_risk(trajectories_0)

# Generate under α=1 (full treatment)
trajectories_1 = model.generate(X_history, alpha_target=1.0)
outcomes_1 = derive_2year_risk(trajectories_1)

# Model's estimated ATE
ate_model = outcomes_1.mean() - outcomes_0.mean()
```

### 3.3 ATE_Bias Metric

```
ATE_Bias = |ate_model - ate_true|
```

**Interpretation**:
- Lower is better
- Measures how well model preserves causal effects
- Zero bias = perfect causal fidelity

---

## 4. What We Can Evaluate on Real NLST Data

### 4.1 Feasible Evaluations

✅ **ATE_Bias** (as defined above)
- Use observed treatment variables (smoking, screening adherence)
- Estimate ground truth via DML
- Compare with model's counterfactual predictions

✅ **Consistency with Observational Associations**
- Known risk factors (age, smoking) should increase predicted risk
- Model's α-sensitivity should align with epidemiological knowledge

✅ **Monotonicity Checks**
- Higher α (more exposure) should increase risk
- Risk trajectories should be non-decreasing

✅ **Counterfactual Plausibility**
- Generated trajectories under different α should be realistic
- No impossible values (negative risks, risks > 1)

### 4.2 Limitations on Real Data

❌ **Cannot Rigorously Validate**:
- True counterfactual outcomes (never observed)
- Unconfounded treatment effects (always some hidden confounding)
- Long-term causal chains (limited follow-up)

**Why**: Fundamental problem of causal inference—we never observe both Y(α=0) and Y(α=1) for same individual.

---

## 5. Semi-Synthetic Benchmark (Future Work)

### 5.1 Motivation

To rigorously validate causal mechanisms, need ground truth counterfactuals.

**Solution**: Semi-synthetic data where true causal effects are known by construction.

### 5.2 Construction Method

**Step 1**: Fit structural causal model (SCM) to real data
```
X ~ P(X)                          # Covariates
α ~ P(α | X)                      # Treatment assignment
Y ~ P(Y | X, α, noise)            # Outcome generation
```

**Step 2**: Generate synthetic samples with known causal structure
```python
# Sample covariates from real distribution
X_syn = sample_from_real_distribution(X_real)

# Assign treatment (can be randomized or confounded)
alpha_syn = assign_treatment(X_syn, confounding_strength=0.5)

# Generate outcome with known causal effect
Y_syn = generate_outcome(X_syn, alpha_syn, true_ate=0.15)
```

**Step 3**: Evaluate model on synthetic data
```python
ate_model = model.estimate_ate(X_syn, alpha_syn)
ate_true = 0.15  # Known by construction

ate_bias_synthetic = abs(ate_model - ate_true)
```

### 5.3 Advantages

✅ **Ground Truth Available**: Know true ATE by construction

✅ **Controlled Confounding**: Can vary confounding strength

✅ **Ablation Studies**: Test model under different causal structures

---

## 6. Evaluation Protocol

### 6.1 On Real NLST Data

**Step 1**: Select treatment variable
- Option A: Use real variable (e.g., `cigsmok`, pack-years)
- Option B: Use synthetic uniform α

**Step 2**: Estimate ground truth ATE via DML
```python
ate_true, ate_std = estimate_ate_dml(
    Y=real_2year_outcomes,
    T=real_alpha_target,
    X=real_covariates
)
```

**Step 3**: Generate model counterfactuals
```python
ate_model = model.estimate_ate_counterfactual(
    X_history=test_histories,
    alpha_0=0.0,
    alpha_1=1.0
)
```

**Step 4**: Compute ATE_Bias
```python
ate_bias = abs(ate_model - ate_true)
```

**Step 5**: Validate monotonicity
```python
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    risks = model.predict_risk(X_history, alpha_target=alpha)
    assert risks are non-decreasing with alpha
```

### 6.2 Reporting

**Main Table**: Include ATE_Bias alongside predictive metrics

**Supplementary Analysis**:
- Plot: α vs predicted risk (should be monotonic)
- Table: ATE estimates at different α levels
- Sensitivity analysis: ATE_Bias across different treatment variables

---

## 7. Baseline Comparison

### 7.1 Causal Baselines

**CausalForest**:
- Directly estimates CATE
- Can compute ATE_Bias similarly

**Other Generative Models** (STaSy, TabSyn, TabDiff, TSDiff):
- If they support α-guidance, compute ATE_Bias
- If not, report "N/A" for causal metrics

### 7.2 Non-Causal Baselines

**Logistic Regression, XGBoost, Random Forest**:
- Cannot generate counterfactuals
- Report "N/A" for ATE_Bias
- Still report predictive metrics

---

## 8. Interpretation Guidelines

### 8.1 ATE_Bias Thresholds

**Excellent**: ATE_Bias < 0.05  
**Good**: 0.05 ≤ ATE_Bias < 0.10  
**Acceptable**: 0.10 ≤ ATE_Bias < 0.20  
**Poor**: ATE_Bias ≥ 0.20

**Context-Dependent**: Depends on baseline risk and effect size.

### 8.2 Clinical Interpretation

**Example**:
- True ATE = 0.15 (15 percentage point increase in 2-year risk)
- Model ATE = 0.18
- ATE_Bias = 0.03 (3 percentage points)

**Interpretation**: Model slightly overestimates treatment effect, but within acceptable range for clinical decision support.

---

## 9. Advanced Causal Metrics (Future Extensions)

### 9.1 Conditional ATE (CATE)

**Definition**: Treatment effect heterogeneity across subgroups

**Evaluation**:
```python
for subgroup in ['young', 'old', 'heavy_smoker', 'light_smoker']:
    cate_true = estimate_cate_dml(subgroup)
    cate_model = model.estimate_cate(subgroup)
    cate_bias[subgroup] = abs(cate_model - cate_true)
```

### 9.2 Dose-Response Curves

**Definition**: Outcome as function of continuous treatment

**Evaluation**:
```python
alphas = np.linspace(0, 1, 11)
for alpha in alphas:
    risk_model[alpha] = model.predict_risk(X, alpha_target=alpha)
    risk_true[alpha] = estimate_true_risk_at_alpha(alpha)

dose_response_error = mse(risk_model, risk_true)
```

### 9.3 Mediation Analysis

**Question**: Does model capture causal pathways?

**Example**: Smoking → Nodule Growth → Cancer

**Evaluation**: Check if model's intermediate representations align with known mediators.

---

## 10. Limitations & Caveats

### 10.1 Observational Data Challenges

**Unmeasured Confounding**: Always possible in observational data

**Selection Bias**: NLST participants are not representative of general population

**Temporal Confounding**: Time-varying confounders complicate causal inference

### 10.2 Model Assumptions

**Positivity**: Need overlap in treatment distributions

**Consistency**: Same treatment definition across individuals

**No Interference**: One person's treatment doesn't affect another's outcome

### 10.3 Honest Reporting

**In Papers**: Clearly state that ATE_Bias is estimated, not ground truth

**Limitations Section**: Acknowledge observational data constraints

**Future Work**: Propose semi-synthetic validation

---

## 11. Implementation Checklist

- [ ] Select treatment variable (real or synthetic α)
- [ ] Implement DML-based ATE estimation
- [ ] Implement model counterfactual generation
- [ ] Compute ATE_Bias
- [ ] Validate monotonicity of α → risk
- [ ] Generate dose-response plots
- [ ] Compare ATE_Bias across baselines
- [ ] Document assumptions and limitations

---

## 12. Revision History

- **v1.0 (2026-03-10)**: Initial causal evaluation protocol
