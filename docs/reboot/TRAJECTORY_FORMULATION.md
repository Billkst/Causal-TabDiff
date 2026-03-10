# Trajectory Formulation: Risk Trajectory vs Covariate Trajectory

**Version**: 1.0  
**Date**: 2026-03-10  
**Status**: Active

---

## 1. Core Distinction

### 1.1 What We Mean by "Trajectory"

**Covariate Trajectory** (NOT what we're doing):
```
Complete future feature table:
T3: [age, smoking_status, nodule_size, CT_findings, ...]
T4: [age, smoking_status, nodule_size, CT_findings, ...]
T5: [age, smoking_status, nodule_size, CT_findings, ...]
...
```

**Risk Trajectory** (what we ARE doing):
```
Future risk sequence:
T3: P(cancer by T3 | history up to landmark)
T4: P(cancer by T4 | history up to landmark)
T5: P(cancer by T5 | history up to landmark)
...
```

### 1.2 Why This Matters

**Data Reality**:
- NLST measures complete covariates only during active screening (T0, T1, T2)
- Post-screening follow-up (T3-T7) tracks cancer outcomes, not detailed features
- Cannot supervise generation of unmeasured future covariates

**Scientific Validity**:
- Risk trajectories are standard in survival analysis and clinical prediction
- Generating risk curves is clinically meaningful and actionable
- Avoids fabricating data that was never observed

---

## 2. Mathematical Formulation

### 2.1 Notation

- `t`: Landmark time point ∈ {T0, T1, T2}
- `τ`: Future time point, τ > t
- `X_{≤t}`: Observed covariate history up to landmark t
- `cancyr_i`: Year of first cancer diagnosis for person i
- `h_τ`: Hazard (instantaneous risk) at year τ
- `r_τ`: Cumulative risk by year τ
- `S_τ`: Survival probability at year τ

### 2.2 Risk Trajectory Definition

**Hazard Trajectory**:
```
H_t = [h_{t+1}, h_{t+2}, ..., h_{T_max}]
```

Where hazard at year τ is:
```
h_τ = P(cancer diagnosed at τ | survived to τ, X_{≤t})
```

**Cumulative Risk Trajectory**:
```
R_t = [r_{t+1}, r_{t+2}, ..., r_{T_max}]
```

Where cumulative risk by year τ is:
```
r_τ = P(cancer by τ | X_{≤t})
```

**Survival Trajectory**:
```
S_t = [S_{t+1}, S_{t+2}, ..., S_{T_max}]
```

Where survival at year τ is:
```
S_τ = P(cancer-free at τ | X_{≤t})
```

### 2.3 Relationships

**Survival to Cumulative Risk**:
```
r_τ = 1 - S_τ
```

**Hazard to Survival** (discrete time):
```
S_τ = S_{τ-1} × (1 - h_τ)
```

**Cumulative Risk from Hazards**:
```
r_τ = 1 - ∏_{k=t+1}^{τ} (1 - h_k)
```

---

## 3. Implementation Strategy

### 3.1 Trajectory Target Construction

**From NLST Data**:

For each (person i, landmark t) sample:

```python
def construct_risk_trajectory(cancyr_i, landmark_t, T_max=7):
    """
    Construct risk trajectory target from cancer diagnosis year.
    
    Args:
        cancyr_i: Year of cancer diagnosis (0 if never diagnosed)
        landmark_t: Current landmark time point
        T_max: Maximum follow-up year
    
    Returns:
        hazard_trajectory: [h_{t+1}, ..., h_{T_max}]
    """
    trajectory_length = T_max - landmark_t
    hazard_trajectory = np.zeros(trajectory_length)
    
    if cancyr_i > 0 and cancyr_i > landmark_t:
        # Cancer diagnosed at specific year
        diagnosis_offset = int(cancyr_i - landmark_t - 1)
        if diagnosis_offset < trajectory_length:
            hazard_trajectory[diagnosis_offset] = 1.0
    
    return hazard_trajectory
```

**Example**:
- Person diagnosed at T4, landmark at T1
- Trajectory: [0, 0, 1, 0, 0, 0] (diagnosis at offset 2, i.e., T4)

### 3.2 Model Output Structure

**Diffusion Model Output**:

```python
class CausalTabDiff:
    def forward(self, x_history, alpha_target):
        """
        Args:
            x_history: [batch, seq_len, feature_dim] - observed history
            alpha_target: [batch, 1] - causal guidance target
        
        Returns:
            risk_trajectory: [batch, trajectory_len] - future risk sequence
            diff_loss: diffusion reconstruction loss
            guidance_loss: causal guidance alignment loss
        """
        # Encode history
        h = self.encoder(x_history)
        
        # Generate risk trajectory via diffusion
        risk_trajectory = self.diffusion_decoder(h, alpha_target)
        
        return risk_trajectory, diff_loss, guidance_loss
```

### 3.3 Deriving 2-Year Risk

**From Hazard Trajectory**:

```python
def derive_2year_risk(hazard_trajectory):
    """
    Derive 2-year risk from hazard trajectory.
    
    Args:
        hazard_trajectory: [h_1, h_2, h_3, ...]
    
    Returns:
        risk_2year: P(event within 2 years)
    """
    h1 = hazard_trajectory[0]
    h2 = hazard_trajectory[1]
    
    # Probability of surviving both years
    survival_2year = (1 - h1) * (1 - h2)
    
    # 2-year risk
    risk_2year = 1 - survival_2year
    
    return risk_2year
```

---

## 4. Advantages of Risk Trajectory Formulation

### 4.1 Clinical Relevance

✅ **Standard Practice**: Risk curves are routinely used in oncology (e.g., Kaplan-Meier curves)

✅ **Actionable**: Clinicians use risk trajectories to schedule follow-up intervals

✅ **Interpretable**: "Your 5-year cancer risk is 15%" is more meaningful than raw covariate predictions

### 4.2 Technical Advantages

✅ **Data-Grounded**: Supervised by actual outcome observations (cancyr)

✅ **Continuous Output**: Diffusion models naturally generate smooth risk curves

✅ **Flexible Aggregation**: Can derive any time-window risk (1-year, 2-year, 5-year) from trajectory

✅ **Counterfactual-Ready**: Can generate "what-if" risk curves under different interventions

### 4.3 Thesis Alignment

✅ **Preserves Trajectory Narrative**: Still generating temporal sequences

✅ **Maintains Generative Framework**: Diffusion model generates distributions over time

✅ **Enables Causal Claims**: Alpha_target guides counterfactual risk trajectories

---

## 5. Comparison with Covariate Trajectory

| Aspect | Covariate Trajectory | Risk Trajectory |
|--------|---------------------|-----------------|
| **Supervision** | Requires future feature measurements | Uses observed outcome years |
| **Data Support** | NLST lacks T3-T7 covariates | NLST tracks outcomes through T7 |
| **Clinical Use** | Indirect (features → risk) | Direct (risk curves) |
| **Interpretability** | Complex (many features) | Simple (single risk value per year) |
| **Validation** | Hard to evaluate quality | Standard metrics (C-index, calibration) |
| **Thesis Fit** | Literal interpretation | Scientifically adapted interpretation |

---

## 6. Handling Edge Cases

### 6.1 Censoring

**Scenario**: Person censored at T5, no cancer observed

**Trajectory Construction**:
```
hazard_trajectory = [0, 0, 0, 0, NaN, NaN, NaN]
                     T1 T2 T3 T4  T5  T6  T7
```

**Loss Computation**: Mask out NaN positions, only supervise observed years

### 6.2 Early Diagnosis

**Scenario**: Person diagnosed at T1, landmark at T0

**Trajectory Construction**:
```
hazard_trajectory = [1, 0, 0, 0, 0, 0, 0]
                     T1 T2 T3 T4 T5 T6 T7
```

**Sample Exclusion**: This person's T1 and T2 landmark samples should be excluded (pre-existing cancer)

### 6.3 Late Diagnosis

**Scenario**: Person diagnosed at T6, landmark at T2

**Trajectory Construction**:
```
hazard_trajectory = [0, 0, 0, 1, 0]
                     T3 T4 T5 T6 T7
```

---

## 7. Future Upgrade Path

### 7.1 If More Complete Data Becomes Available

**Scenario**: Future dataset with complete covariate measurements through T7

**Upgrade Strategy**:
1. Keep risk trajectory as primary output
2. Add auxiliary covariate trajectory generation
3. Use covariate trajectory to improve risk trajectory accuracy
4. Validate that generated covariates are consistent with risk predictions

### 7.2 Hybrid Formulation

**Possible Extension**:
```
Model Output:
- Primary: Risk trajectory [r_1, r_2, ..., r_7]
- Auxiliary: Key covariate trajectory (e.g., nodule size progression)
- Constraint: Covariate trajectory must be consistent with risk trajectory
```

---

## 8. Implementation Checklist

### 8.1 Data Module

- [ ] Construct hazard trajectory targets from cancyr
- [ ] Handle censoring with proper masking
- [ ] Validate trajectory lengths match T_max - landmark
- [ ] Ensure no leakage from future outcomes into input features

### 8.2 Model Architecture

- [ ] Output layer produces trajectory_len risk values
- [ ] Apply sigmoid activation for risk ∈ [0, 1]
- [ ] Support variable landmark positions (T0, T1, T2)
- [ ] Integrate alpha_target guidance into trajectory generation

### 8.3 Loss Functions

- [ ] Trajectory supervision loss (e.g., BCE per time point)
- [ ] 2-year risk loss (derived from trajectory)
- [ ] Causal guidance loss (alpha_target alignment)
- [ ] Optional: Monotonicity constraint (risk should not decrease)

### 8.4 Evaluation

- [ ] Trajectory-level metrics (MSE, MAE on risk curves)
- [ ] 2-year prediction metrics (AUROC, AUPRC)
- [ ] Calibration of trajectory predictions
- [ ] Counterfactual trajectory quality (alpha_target sensitivity)

---

## 9. Communication Guidelines

### 9.1 In Papers/Thesis

**Recommended Phrasing**:

> "We formulate the trajectory generation task as **future risk trajectory prediction**, where the model generates a sequence of yearly cancer risk estimates from a given landmark time point. This formulation aligns with clinical practice in oncology, where risk curves guide screening and intervention decisions."

**Avoid**:
- "We generate complete future covariate tables" (not supported by data)
- "We predict future screening results" (not what we're doing)

### 9.2 In Presentations

**Slide Title**: "Risk Trajectory Generation"

**Visual**: Show risk curve increasing over time, with 2-year window highlighted

**Caption**: "Model generates personalized risk trajectories, enabling dynamic risk monitoring and counterfactual 'what-if' analysis"

---

## 10. Validation Criteria

### 10.1 Trajectory Quality

**Must Pass**:
1. Generated risks are in [0, 1]
2. Cumulative risk is non-decreasing (or hazards are non-negative)
3. Trajectory-derived 2-year risk matches direct prediction
4. Counterfactual trajectories respond sensibly to alpha_target

### 10.2 Clinical Plausibility

**Must Check**:
1. High-risk patients have steeper risk curves
2. Risk trajectories align with known risk factors (age, smoking)
3. Censored patients have lower predicted risks than diagnosed patients

---

## 11. Revision History

- **v1.0 (2026-03-10)**: Initial formulation clarifying risk vs covariate trajectories

---

## 12. References

**Standard Survival Analysis**:
- Cox proportional hazards model
- Kaplan-Meier survival curves
- Competing risks framework

**Clinical Risk Prediction**:
- Lung-RADS risk stratification
- Brock model for nodule malignancy
- PLCOm2012 lung cancer risk model

**Diffusion Models for Survival**:
- DiffPO (Ma et al., 2023): Causal diffusion for potential outcomes
- DeepSurv (Katzman et al., 2018): Deep learning for survival analysis
