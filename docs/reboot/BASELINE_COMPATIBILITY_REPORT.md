# Baseline Compatibility Report

**Version**: 1.0  
**Date**: 2026-03-10  
**Status**: Active

---

## 1. Overview

This document assesses each baseline's compatibility with the dual-layer task reformulation and defines adaptation strategies.

---

## 2. Original Baseline Suite

### 2.1 Existing Baselines

1. **CausalForest** - Causal inference via random forests
2. **STaSy** (ICLR 2023) - Conditional tabular GAN
3. **TabSyn** (ICLR 2024) - Diffusion-based tabular synthesis
4. **TabDiff** (ICLR 2025) - Tabular diffusion model
5. **TSDiff** (ICLR 2023) - Time-series diffusion

---

## 3. Compatibility Assessment

### 3.1 CausalForest

**Original Purpose**: Heterogeneous treatment effect estimation

**Compatibility**: ✅ **COMPATIBLE** with modifications

**Adaptation Strategy**:
- Use as binary classifier for 2-year risk
- Treatment variable = alpha_target
- Outcome = y_2year
- Can estimate CATE and ATE_Bias

**Implementation**:
```python
from econml.drf import DMLOrthoForest

cf = DMLOrthoForest(
    n_trees=100,
    min_leaf_size=10,
    max_depth=None
)

cf.fit(
    Y=y_2year,
    T=alpha_target,
    X=X_history_flattened  # Flatten temporal features
)

# Predict 2-year risk
risk_pred = cf.effect(X_test)
```

**Limitations**:
- Cannot generate risk trajectories (only point predictions)
- Requires flattening temporal sequences

**Verdict**: **RETAIN** - Core causal baseline

---

### 3.2 STaSy

**Original Purpose**: Conditional tabular data generation

**Compatibility**: ⚠️ **PARTIALLY COMPATIBLE**

**Adaptation Strategy**:
- Generate synthetic (X, y_2year) samples
- Train downstream classifier on synthetic data
- Evaluate via TSTR protocol

**Implementation**:
```python
# Train STaSy on real data
stasy.fit(X_real, y_real)

# Generate synthetic samples
X_syn, y_syn = stasy.generate(n_samples=len(X_real))

# Train classifier on synthetic
clf = XGBClassifier()
clf.fit(X_syn, y_syn)

# Test on real
risk_pred = clf.predict_proba(X_test)[:, 1]
```

**Challenges**:
- Original STaSy designed for static tables, not temporal sequences
- Need to adapt for variable-length histories
- May require flattening or padding

**Verdict**: **RETAIN** with temporal adaptation

---

### 3.3 TabSyn

**Original Purpose**: High-fidelity tabular synthesis via diffusion

**Compatibility**: ⚠️ **PARTIALLY COMPATIBLE**

**Adaptation Strategy**: Same as STaSy (TSTR protocol)

**Implementation**:
```python
# Train TabSyn
tabsyn.fit(X_real, y_real)

# Generate and evaluate via TSTR
X_syn, y_syn = tabsyn.generate(n_samples=len(X_real))
clf.fit(X_syn, y_syn)
risk_pred = clf.predict_proba(X_test)[:, 1]
```

**Challenges**:
- Static table assumption
- No native temporal modeling
- No causal guidance mechanism

**Verdict**: **RETAIN** as generative baseline

---

### 3.4 TabDiff

**Original Purpose**: Tabular diffusion without causal constraints

**Compatibility**: ⚠️ **PARTIALLY COMPATIBLE**

**Adaptation Strategy**: TSTR + optional trajectory extension

**Implementation**:
```python
# Standard TSTR
tabdiff.fit(X_real, y_real)
X_syn, y_syn = tabdiff.generate(n_samples=len(X_real))

# Optional: Extend to trajectory generation
# (requires architecture modification)
```

**Challenges**:
- No alpha_target guidance
- No trajectory output
- Static table focus

**Verdict**: **RETAIN** as non-causal diffusion baseline

---

### 3.5 TSDiff

**Original Purpose**: Time-series diffusion

**Compatibility**: ✅ **MOST COMPATIBLE**

**Adaptation Strategy**:
- Naturally handles temporal sequences
- Can adapt for risk trajectory generation
- Add alpha_target conditioning

**Implementation**:
```python
# Train TSDiff on temporal sequences
tsdiff.fit(
    X_history=X_temporal,
    y=y_2year
)

# Generate risk trajectories (if extended)
risk_traj = tsdiff.generate_trajectory(X_history, alpha_target)
```

**Advantages**:
- Already designed for time-series
- Closest architecture to Causal-TabDiff
- Can potentially generate trajectories

**Verdict**: **RETAIN** - Key temporal baseline

---

## 4. New Baselines (Real-Data Anchors)

### 4.1 Logistic Regression

**Purpose**: Simple linear baseline

**Implementation**:
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_flat, y_train)
risk_pred = lr.predict_proba(X_test_flat)[:, 1]
```

**Advantages**:
- Fast, interpretable
- Establishes lower bound
- No hyperparameter tuning needed

**Verdict**: **ADD** - Essential baseline

---

### 4.2 XGBoost

**Purpose**: Strong non-linear baseline

**Implementation**:
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight=neg_count/pos_count,  # Handle imbalance
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
xgb.fit(X_train_flat, y_train)
risk_pred = xgb.predict_proba(X_test_flat)[:, 1]
```

**Advantages**:
- State-of-art tabular performance
- Handles imbalance well
- Fast inference

**Verdict**: **ADD** - Critical competitive baseline

---

### 4.3 Balanced Random Forest

**Purpose**: Ensemble baseline with imbalance handling

**Implementation**:
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train_flat, y_train)
risk_pred = rf.predict_proba(X_test_flat)[:, 1]
```

**Advantages**:
- Robust to overfitting
- Feature importance analysis
- No scaling required

**Verdict**: **ADD** - Standard ML baseline

---

## 5. Adaptation Priority

### 5.1 Phase 1: Real-Data Anchors (Immediate)

1. Logistic Regression
2. XGBoost
3. Random Forest

**Rationale**: Establish performance floor, fast to implement

### 5.2 Phase 2: Causal Baseline (High Priority)

4. CausalForest

**Rationale**: Only baseline with native causal inference

### 5.3 Phase 3: Generative Baselines (Medium Priority)

5. TSDiff (temporal, closest to our method)
6. TabDiff (diffusion, no causal)
7. STaSy (GAN-based)
8. TabSyn (diffusion, high-fidelity)

**Rationale**: Validate generative approach, TSTR evaluation

---

## 6. Fair Comparison Requirements

### 6.1 Mandatory Consistency

**Same Data**:
- All baselines use identical master modeling table
- Same train/val/test split (by pid)
- Same preprocessing

**Same Evaluation**:
- Same metrics (AUPRC, AUROC, F1, etc.)
- Same threshold selection protocol
- Same 5-seed procedure

**Same Task**:
- All predict 2-year risk
- Trajectory generation optional (only if baseline supports)

### 6.2 Baseline-Specific Allowances

**Feature Representation**:
- Temporal models: Use sequential X_history
- Static models: Use flattened features or last-visit-only

**Training Protocol**:
- Generative models: TSTR/TRTR evaluation
- Discriminative models: Direct training

**Hyperparameters**:
- Each baseline uses its published defaults or grid search on validation set

---

## 7. Evaluation Matrix

| Baseline | 2-Year Risk | Trajectory | Causal (ATE_Bias) | TSTR | Temporal |
|----------|-------------|------------|-------------------|------|----------|
| **Causal-TabDiff** | ✅ | ✅ | ✅ | ✅ | ✅ |
| CausalForest | ✅ | ❌ | ✅ | ❌ | ⚠️ (flatten) |
| TSDiff | ✅ | ⚠️ (extend) | ❌ | ✅ | ✅ |
| TabDiff | ✅ | ❌ | ❌ | ✅ | ❌ |
| STaSy | ✅ | ❌ | ❌ | ✅ | ❌ |
| TabSyn | ✅ | ❌ | ❌ | ✅ | ❌ |
| XGBoost | ✅ | ❌ | ❌ | ❌ | ⚠️ (flatten) |
| Random Forest | ✅ | ❌ | ❌ | ❌ | ⚠️ (flatten) |
| Logistic Reg | ✅ | ❌ | ❌ | ❌ | ⚠️ (flatten) |

**Legend**:
- ✅ Fully supported
- ⚠️ Requires adaptation
- ❌ Not applicable

---

## 8. Implementation Checklist

### 8.1 Per Baseline

- [ ] Adapt to landmark-based data structure
- [ ] Handle variable-length histories (or flatten)
- [ ] Implement 2-year risk prediction
- [ ] Use same train/val/test split
- [ ] Apply same evaluation metrics
- [ ] Run 5-seed experiments
- [ ] Document any baseline-specific modifications

### 8.2 Reporting

- [ ] Main table: All baselines, all metrics
- [ ] Supplementary: Baseline-specific details
- [ ] Note which baselines support trajectories/causal
- [ ] Fair comparison statement in paper

---

## 9. Expected Outcomes

### 9.1 Hypothesis

**Causal-TabDiff should win or tie on**:
- AUPRC, AUROC (predictive performance)
- ATE_Bias (causal validity)
- TSTR metrics (generative quality)

**XGBoost likely competitive on**:
- Pure predictive metrics (AUPRC, AUROC)
- But lacks causal/generative capabilities

**CausalForest should be competitive on**:
- ATE_Bias (designed for causal inference)
- But may lag on predictive performance

### 9.2 Success Criteria

**Minimum**: Causal-TabDiff in top 2 for all primary metrics

**Target**: Causal-TabDiff ranks #1 on AUPRC, AUROC, ATE_Bias

**Stretch**: Across-the-board wins on all metrics

---

## 10. Fallback Strategies

### 10.1 If Baseline Adaptation Fails

**Option 1**: Report "N/A" for incompatible metrics
- Example: TabSyn cannot compute ATE_Bias → report N/A

**Option 2**: Exclude baseline with justification
- Example: If STaSy cannot handle temporal data → exclude with explanation

**Option 3**: Simplified version
- Example: Use only T2 (last visit) features for static baselines

### 10.2 If Performance is Poor

**Diagnose**:
- Check data leakage
- Verify label construction
- Inspect class imbalance handling
- Review hyperparameters

**Iterate**:
- Tune on validation set
- Try different architectures
- Ensemble methods

---

## 11. Documentation Requirements

**For Each Baseline**:
- Original paper citation
- Adaptation details
- Hyperparameters used
- Training time
- Any modifications from original

**In Paper**:
- Fair comparison statement
- Baseline descriptions
- Adaptation justifications
- Limitations acknowledged

---

## 12. Revision History

- **v1.0 (2026-03-10)**: Initial compatibility assessment
