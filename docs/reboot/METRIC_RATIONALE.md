# Metric Rationale: Comprehensive Evaluation Framework

**Version**: 1.0  
**Date**: 2026-03-10  
**Status**: Active

---

## 1. Philosophy: Multi-Report, Single-Goal

**Principle**: Report many metrics, optimize for across-the-board wins.

**Why Not Single Metric**:
- Different stakeholders care about different aspects
- Clinical deployment requires multiple validation angles
- Peer review expects comprehensive evaluation

**Primary Goal**: Demonstrate Causal-TabDiff superiority across ALL metric categories, not just cherry-picked ones.

---

## 2. Metric Hierarchy

### 2.1 Tier 1: Primary Ranking Metrics (Main Table)

**AUPRC** (Area Under Precision-Recall Curve)
- **Why Primary**: Handles class imbalance better than AUROC
- **Interpretation**: Average precision across all recall levels
- **Clinical Relevance**: Focuses on positive class (cancer cases)
- **Threshold-Free**: No arbitrary cutoff needed

**AUROC** (Area Under ROC Curve)
- **Why Include**: Standard in medical ML, enables comparison with literature
- **Interpretation**: Probability model ranks positive higher than negative
- **Limitation**: Can be misleading with severe imbalance
- **Threshold-Free**: Evaluates all possible thresholds

**F1_opt** (F1 Score at Optimal Threshold)
- **Why Include**: Balances precision and recall
- **Threshold Selection**: Maximize F1 on validation set, apply to test
- **Clinical Relevance**: Represents practical operating point
- **Limitation**: Single-point metric, doesn't show full trade-off curve

**ATE_Bias** (Average Treatment Effect Bias)
- **Why Include**: Validates causal mechanism
- **Interpretation**: How well model preserves causal effects
- **Unique to**: Causal-TabDiff's core innovation
- **Computation**: Via EconML double machine learning

---

### 2.2 Tier 2: Threshold-Dependent Metrics (Supplementary Table)

**Precision / PPV** (Positive Predictive Value)
- **Clinical Question**: "If model says high-risk, what's probability of actual cancer?"
- **Use Case**: Resource allocation (how many false alarms?)

**Recall / Sensitivity**
- **Clinical Question**: "What fraction of actual cancers does model catch?"
- **Use Case**: Screening effectiveness (how many missed cases?)

**Specificity / TNR** (True Negative Rate)
- **Clinical Question**: "What fraction of healthy people correctly identified?"
- **Use Case**: Avoiding unnecessary interventions

**NPV** (Negative Predictive Value)
- **Clinical Question**: "If model says low-risk, how confident can we be?"
- **Use Case**: Reassurance for negative predictions

**Accuracy**
- **Why Supplementary**: Misleading with imbalance (99% accuracy by predicting all negative)
- **Use Case**: Overall correctness, but NOT for ranking models

**Balanced Accuracy**
- **Why Better**: Average of sensitivity and specificity
- **Use Case**: Accounts for imbalance better than raw accuracy

**MCC** (Matthews Correlation Coefficient)
- **Why Include**: Single metric considering all confusion matrix cells
- **Range**: [-1, 1], where 1 is perfect, 0 is random
- **Advantage**: Robust to class imbalance

---

### 2.3 Tier 3: Calibration & Clinical Utility

**Brier Score**
- **Definition**: Mean squared error of probabilistic predictions
- **Range**: [0, 1], lower is better
- **Why Critical**: Measures probability quality, not just ranking

**Calibration Intercept & Slope**
- **Ideal Values**: Intercept = 0, Slope = 1
- **Interpretation**: Do predicted probabilities match observed frequencies?
- **Clinical Importance**: Doctors need trustworthy probabilities

**Calibration Plot**
- **Visual**: Predicted probability vs observed frequency
- **Ideal**: Points lie on diagonal
- **Diagnostic**: Shows over/under-confidence

**Decision Curve Analysis**
- **Metric**: Net Benefit across threshold range
- **Clinical Question**: "At what risk threshold should we intervene?"
- **Advantage**: Incorporates clinical consequences (false positives vs false negatives)

**ECE** (Expected Calibration Error)
- **Definition**: Weighted average of calibration errors across bins
- **Why Include**: Single-number calibration summary
- **Implementation**: If computationally feasible

---

### 2.4 Tier 4: Generative/Distribution Diagnostics

**TRTR** (Train on Real, Test on Real)
- **Purpose**: Baseline performance on real data
- **Use Case**: Reference point for synthetic data quality

**TSTR** (Train on Synthetic, Test on Real)
- **Purpose**: Can synthetic data train useful models?
- **Interpretation**: Measures utility of generated samples
- **Report**: AUPRC, AUROC, F1 on TSTR setup

**TR+S→TR** (Train on Real+Synthetic, Test on Real)
- **Purpose**: Does synthetic data augmentation help?
- **Interpretation**: Additive value of generation
- **Expected**: Should improve or match TRTR

**Wasserstein Distance**
- **Purpose**: Distributional similarity between real and synthetic
- **Interpretation**: Lower = more similar
- **Limitation**: Doesn't directly measure downstream task quality

**CMD** (Correlation Matrix Distance)
- **Purpose**: Preserves feature correlations?
- **Interpretation**: Lower = better correlation preservation
- **Why Include**: Validates structural fidelity

---

### 2.5 Tier 5: Efficiency Metrics

**Params(M)** (Model Parameters in Millions)
- **Purpose**: Model complexity
- **Trade-off**: Performance vs deployability

**AvgInfer(ms/sample)** (Average Inference Time)
- **Purpose**: Computational cost
- **Clinical Relevance**: Real-time screening feasibility

**Training Time** (hours)
- **Purpose**: Development cost
- **Practical Consideration**: Research iteration speed

**GPU Hours**
- **Purpose**: Resource consumption
- **Environmental/Cost**: Carbon footprint, cloud costs

**Peak VRAM** (GB)
- **Purpose**: Hardware requirements
- **Deployment Constraint**: Can it run on available hardware?

---

## 3. Why Accuracy is NOT Primary

**Problem with Accuracy in Imbalanced Data**:

Example: 1% cancer prevalence
- Model A: Predicts all negative → 99% accuracy
- Model B: Catches 80% of cancers, 10% false positive rate → 89% accuracy
- **Accuracy says A is better, but B is clinically superior**

**Why AUPRC is Better**:
- Focuses on positive class performance
- Not inflated by large negative class
- Directly relevant to screening (finding rare cancers)

**Accuracy's Role**: Supplementary metric, reported but not used for ranking.

---

## 4. Why Confusion Matrix is Mandatory

**Information Content**:
```
                Predicted
                Neg    Pos
Actual  Neg     TN     FP
        Pos     FN     TP
```

**Enables Computation of**:
- Sensitivity = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- PPV = TP / (TP + FP)
- NPV = TN / (TN + FN)

**Clinical Interpretation**:
- FN (False Negatives): Missed cancers - most critical error
- FP (False Positives): Unnecessary follow-up - resource waste
- Trade-off visualization: Can't minimize both simultaneously

**Reporting Format**:
- Raw counts (for absolute numbers)
- Normalized by row (for rates)
- Both must be provided

---

## 5. Why Calibration is Critical

**Scenario**: Model with AUROC = 0.85 but poor calibration

**Problem**:
- Predicts 30% risk for patient
- Actual risk is 5%
- Doctor over-treats based on inflated probability

**Solution**: Calibration metrics ensure probabilities are trustworthy

**Clinical Decision-Making**:
- Doctors use probabilities, not just binary predictions
- "Your 2-year cancer risk is 15%" guides screening intervals
- Miscalibrated probabilities lead to wrong decisions

**Calibration Methods** (if needed):
- Platt scaling
- Isotonic regression
- Temperature scaling

---

## 6. Why Decision Curves Matter

**Beyond AUROC**:
- AUROC treats all errors equally
- Clinical reality: FN and FP have different costs

**Net Benefit**:
```
NB(t) = (TP / N) - (FP / N) × [t / (1 - t)]
```

Where:
- t = risk threshold for intervention
- Second term = weighted cost of false positives

**Interpretation**:
- Positive NB = model adds value over "treat all" or "treat none"
- Peak NB = optimal threshold
- Curve comparison = which model provides most value

---

## 7. Metric Reporting Format

### 7.1 Main Results Table

```
| Model          | AUPRC↑ | AUROC↑ | F1↑   | ATE_Bias↓ | Params(M) | Infer(ms) |
|----------------|--------|--------|-------|-----------|-----------|-----------|
| Causal-TabDiff | 0.XXX  | 0.XXX  | 0.XXX | 0.XXX     | XX.X      | X.XX      |
| TabDiff        | 0.XXX  | 0.XXX  | 0.XXX | 0.XXX     | XX.X      | X.XX      |
| ...            |        |        |       |           |           |           |
```

**Format**: Mean ± Std (5 seeds)

### 7.2 Supplementary Table A: Threshold Metrics

```
| Model          | Precision | Recall | Specificity | NPV   | Acc   | Bal_Acc | MCC   |
|----------------|-----------|--------|-------------|-------|-------|---------|-------|
| Causal-TabDiff | 0.XXX     | 0.XXX  | 0.XXX       | 0.XXX | 0.XXX | 0.XXX   | 0.XXX |
```

### 7.3 Supplementary Table B: Calibration

```
| Model          | Brier↓ | Cal_Intercept | Cal_Slope | ECE↓  |
|----------------|--------|---------------|-----------|-------|
| Causal-TabDiff | 0.XXX  | 0.XXX         | 0.XXX     | 0.XXX |
```

### 7.4 Supplementary Table C: Generative Quality

```
| Model          | TRTR_AUPRC | TSTR_AUPRC | TR+S_AUPRC | Wasserstein↓ | CMD↓  |
|----------------|------------|------------|------------|--------------|-------|
| Causal-TabDiff | 0.XXX      | 0.XXX      | 0.XXX      | 0.XXX        | 0.XXX |
```

---

## 8. Threshold Selection Protocol

**Validation Set**:
1. Compute predictions for all validation samples
2. For each threshold t ∈ [0, 1] (step 0.01):
   - Compute F1 score
3. Select t* = argmax F1
4. Record t* for test evaluation

**Test Set**:
1. Apply fixed threshold t* from validation
2. Compute all threshold-dependent metrics
3. NO re-optimization on test set

**Rationale**: Prevents overfitting to test set, ensures fair comparison.

---

## 9. Statistical Significance Testing

**5-Seed Protocol**:
- Train 5 models with different random seeds
- Report mean ± std for each metric
- Enables statistical comparison

**Significance Tests** (if needed):
- Paired t-test for metric differences
- Wilcoxon signed-rank test (non-parametric)
- Bonferroni correction for multiple comparisons

**Reporting**:
- Bold best result in each column
- * for p < 0.05, ** for p < 0.01

---

## 10. Figures (Mandatory Outputs)

**Figure 1: ROC Curves**
- All models on same plot
- Diagonal reference line (random classifier)
- Legend with AUC values

**Figure 2: PR Curves**
- All models on same plot
- Horizontal line at prevalence (random baseline)
- Legend with AUPRC values

**Figure 3: Calibration Plots**
- Predicted probability (x-axis) vs observed frequency (y-axis)
- Diagonal reference line (perfect calibration)
- Separate subplot per model or overlay

**Figure 4: Decision Curves**
- Threshold (x-axis) vs net benefit (y-axis)
- Include "treat all" and "treat none" baselines
- Identify optimal threshold range

**Figure 5: Confusion Matrices**
- Best model only (or top 3)
- Both raw counts and normalized
- Heatmap visualization

**Figure 6: Risk Trajectory Examples**
- Causal-TabDiff specific
- Show 3-5 example patients
- Compare predicted vs actual trajectories

---

## 11. What NOT to Do

❌ **Cherry-pick metrics**: Report all, not just favorable ones

❌ **Optimize on test set**: All tuning on validation only

❌ **Report accuracy as primary**: Use AUPRC/AUROC for ranking

❌ **Ignore calibration**: Probabilities must be trustworthy

❌ **Skip confusion matrix**: Essential for clinical interpretation

❌ **Use different thresholds per model**: Fair comparison requires same threshold selection protocol

---

## 12. References (TODO if Internet Available)

**Calibration**:
- Steyerberg et al., "Assessing the performance of prediction models"
- Van Calster et al., "Calibration of risk prediction models"

**Decision Curves**:
- Vickers & Elkin, "Decision curve analysis: a novel method for evaluating prediction models"

**Imbalanced Learning**:
- Saito & Rehmsmeier, "The precision-recall plot is more informative than the ROC plot"

**Causal Metrics**:
- Chernozhukov et al., "Double/debiased machine learning for treatment and causal parameters"

---

## 13. Revision History

- **v1.0 (2026-03-10)**: Initial rationale for comprehensive metrics system
