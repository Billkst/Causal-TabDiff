# Experiment Protocol

**Version**: 1.0  
**Date**: 2026-03-10  
**Status**: Active

---

## 1. Overview

This document defines the complete experimental workflow from data construction to final evaluation, ensuring reproducibility and scientific rigor.

---

## 2. Data Construction Pipeline

### 2.1 Stage 1: Raw Table Loading

**Input**: 5 NLST raw CSV files
- `nlst_780_prsn_idc_20210527.csv` (person baseline)
- `nlst_780_screen_idc_20210527.csv` (screening visits)
- `nlst_780_ctab_idc_20210527.csv` (abnormality findings)
- `nlst_780_ctabc_idc_20210527.csv` (abnormality changes)
- `nlst_780_canc_idc_20210527.csv` (cancer outcomes)

**Output**: Pandas DataFrames with validated schemas

### 2.2 Stage 2: Intermediate Table Construction (B1-1 IMPLEMENTED)

**实现**: `src/data/build_landmark_tables.py`

**1. person_baseline_table** (from prsn):
```
Columns: pid, age, gender, race, ethnic, bmi, cigsmok, smokeage, smokeyr, 
         cigsperday, smokeday, smokequit, copd, emphysema, chronic_bronchitis, 
         fhx_lung_cancer, prior_cancer
Grain: One row per person
Output: data/landmark_tables/person_baseline_table.pkl
```

**2. person_year_screening_summary** (from screen):
```
Columns: pid, study_yr, ctdxqual, techpara_kvp, techpara_ma, techpara_fov
Grain: One row per (person, year)
Aggregation: groupby(['pid', 'study_yr']).agg({'ctdxqual': 'first', ...})
Output: data/landmark_tables/person_year_screening_summary.pkl
```

**3. person_year_abnormality_summary** (from ctab):
```
Columns: pid, study_yr, abnormality_count, max_long_dia, max_perp_dia, has_spiculated
Grain: One row per (person, year)
Aggregation: groupby(['pid', 'study_yr']).agg(
    abnormality_count=('sct_ab_num', 'count'),
    max_long_dia=('sct_long_dia', 'max'),
    has_spiculated=('sct_margins', lambda x: (x == 4).any())
)
Output: data/landmark_tables/person_year_abnormality_summary.pkl
```

**4. person_year_change_summary** (from ctabc):
```
Columns: pid, study_yr, has_growth, has_attn_change, change_count
Grain: One row per (person, year)
Aggregation: groupby(['pid', 'study_yr']).agg(
    has_growth=('sct_ab_gwth', lambda x: (x == 1).any()),
    has_attn_change=('sct_ab_attn', lambda x: (x > 0).any())
)
Output: data/landmark_tables/person_year_change_summary.pkl
```

**5. event_label_table** (from prsn + canc):
```
Columns: pid, cancyr
Grain: One row per person
Logic: Merge prsn.cancyr with canc.study_yr (first event), cancyr=0 means no cancer
Output: data/landmark_tables/event_label_table.pkl
```

### 2.3 Stage 3: Unified Master Table Construction (B1-1 IMPLEMENTED)

**实现**: `src/data/build_landmark_tables.py::build_unified_landmark_table()`

**unified_person_landmark_table**:
```
Grain: One row per (person, landmark)
Landmarks: {0, 1, 2}
Exclusion: cancyr > 0 and cancyr <= landmark
Label: y_2year = 1 if (cancyr > landmark and cancyr <= landmark + 2) else 0
Temporal filtering: All features from study_yr <= landmark only
Output: data/landmark_tables/unified_person_landmark_table.pkl
```

**真实短历史**:
- T0 样本: 只包含 T0 时间点的特征
- T1 样本: 包含 T0 + T1 时间点的特征
- T2 样本: 包含 T0 + T1 + T2 时间点的特征

**字段结构**:
- pid, landmark, y_2year, cancyr (bookkeeping)
- baseline_*: 16 个基线特征
- screen_t{0,1,2}_*: 每个时间点的筛查特征
- abn_t{0,1,2}_*: 每个时间点的异常特征
- change_t{0,1,2}_*: 每个时间点的变化特征

**运行命令**:
```bash
python src/data/build_landmark_tables.py
```

**Event/Label Table**:
```
Columns: pid, cancyr, censor_yr, event_occurred
Grain: One row per person
```

### 2.3 Stage 3: Master Modeling Table Construction

**Landmark Expansion**:

For each person, create up to 3 samples (one per landmark):

```python
for pid in persons:
    for landmark in [0, 1, 2]:
        # Exclusion check
        if cancyr[pid] > 0 and cancyr[pid] <= landmark:
            continue  # Skip: pre-existing cancer
        
        # Construct sample
        sample = {
            'pid': pid,
            'landmark': landmark,
            'history': extract_history(pid, 0, landmark),
            'y_2year': compute_2year_label(pid, landmark),
            'risk_trajectory': compute_risk_trajectory(pid, landmark),
            'split': get_split(pid),  # train/val/test
        }
```

**Master Table Schema**:
```
- pid: int
- landmark: int (0, 1, or 2)
- X_baseline: dict of baseline features
- X_history: list of dicts, length = landmark + 1
- y_2year: binary (0 or 1)
- risk_trajectory: array of length (7 - landmark)
- history_length: int
- split: str ('train', 'val', 'test')
- cancyr: float (for bookkeeping, NOT input feature)
```

### 2.4 Stage 4: Leakage Validation

**Mandatory Checks**:
1. No future information in X_history
2. cancyr not in feature columns
3. All diagnosis-related fields excluded
4. Temporal ordering verified

---

## 3. Train/Val/Test Split Strategy

### 3.1 Person-Level Splitting

**Critical Rule**: Split by `pid`, NOT by samples

```python
# CORRECT
train_pids, test_pids = train_test_split(unique_pids, test_size=0.2, random_state=seed)

# WRONG - causes leakage
train_samples, test_samples = train_test_split(all_samples, test_size=0.2)
```

**Rationale**: Same person's T0, T1, T2 samples must stay in same split

### 3.2 Split Proportions

- Train: 60%
- Validation: 20%
- Test: 20%

### 3.3 Stratification

Stratify by outcome to maintain class balance:
```python
stratify_labels = [1 if cancyr[pid] > 0 else 0 for pid in unique_pids]
```

---

## 4. Training Sequence

### 4.1 Phase 1: Data Preparation (One-Time)

1. Run data construction pipeline
2. Generate master modeling table
3. Validate leakage checks
4. Save processed data to disk
5. Log data statistics

**Expected Output**:
```
Total persons: ~53,000
Total samples: ~120,000 (after exclusions)
Train samples: ~72,000
Val samples: ~24,000
Test samples: ~24,000
Positive rate: ~1-2%
```

### 4.2 Phase 2: Model Training

**For Causal-TabDiff**:

```bash
python run_experiment.py \
    --data_dir data \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-3 \
    --diffusion_steps 100 \
    --guidance_scale 2.0 \
    --seed 42
```

**Loss Components**:
1. Trajectory reconstruction loss (MSE or BCE)
2. 2-year risk loss (BCE)
3. Causal guidance loss (alpha_target alignment)
4. Optional: Monotonicity regularization

**Training Loop**:
```python
for epoch in epochs:
    for batch in train_loader:
        x_history = batch['X_history']
        y_2year = batch['y_2year']
        risk_traj = batch['risk_trajectory']
        alpha = batch['alpha_target']
        
        # Forward
        pred_traj, pred_2year = model(x_history, alpha)
        
        # Losses
        loss_traj = trajectory_loss(pred_traj, risk_traj)
        loss_2year = bce_loss(pred_2year, y_2year)
        loss_guidance = guidance_loss(pred_traj, alpha)
        
        total_loss = loss_traj + loss_2year + 0.1 * loss_guidance
        
        # Backward
        total_loss.backward()
        optimizer.step()
```

### 4.3 Phase 3: Validation & Early Stopping

**Validation Metrics** (computed every epoch):
- AUPRC (primary for early stopping)
- AUROC
- Brier score

**Early Stopping**:
- Patience: 10 epochs
- Monitor: validation AUPRC
- Save best checkpoint

---

## 5. Baseline Adaptation Sequence

### 5.1 Priority Order

1. **Real-data anchors** (implement first):
   - Logistic Regression
   - XGBoost
   - Random Forest

2. **Retained generative baselines** (adapt existing):
   - CausalForest
   - STaSy
   - TabSyn
   - TabDiff
   - TSDiff

### 5.2 Adaptation Requirements

**For each baseline**:
1. Use same master modeling table
2. Use same train/val/test split
3. Output 2-year risk predictions
4. (Optional) Output risk trajectories if architecture supports

**Baseline-Specific Notes**:
- **CausalForest**: Adapt for binary classification, use alpha_target as treatment
- **STaSy/TabSyn/TabDiff/TSDiff**: Generate synthetic samples, train downstream classifier

---

## 6. Evaluation Sequence

### 6.1 Single-Seed Evaluation

**Step 1**: Load trained model checkpoint

**Step 2**: Generate predictions on test set
```python
test_preds_2year = []
test_preds_traj = []

for batch in test_loader:
    with torch.no_grad():
        pred_traj, pred_2year = model(batch['X_history'], batch['alpha_target'])
        test_preds_2year.append(pred_2year)
        test_preds_traj.append(pred_traj)
```

**Step 3**: Compute all metrics (see METRIC_RATIONALE.md)

**Step 4**: Generate plots
- ROC curve
- PR curve
- Calibration plot
- Confusion matrix
- Decision curve

### 6.2 Multi-Seed Evaluation (5 Seeds)

**Seeds**: [42, 123, 456, 789, 1024]

**For each seed**:
1. Re-split data with seed
2. Train model from scratch
3. Evaluate on test set
4. Save metrics to CSV

**Aggregation**:
```python
metrics_df = pd.DataFrame({
    'seed': seeds,
    'AUPRC': auprc_values,
    'AUROC': auroc_values,
    ...
})

summary = {
    'AUPRC_mean': metrics_df['AUPRC'].mean(),
    'AUPRC_std': metrics_df['AUPRC'].std(),
    ...
}
```

### 6.3 Threshold Selection

**Validation Set**:
- Compute F1 at all thresholds
- Select threshold maximizing F1
- Record optimal threshold

**Test Set**:
- Apply fixed threshold from validation
- Compute all threshold-dependent metrics
- NO test-time tuning

---

## 7. Nohup & Logging Rules

### 7.1 Long Experiments

**Always use nohup**:
```bash
nohup python run_experiment.py --seed 42 > logs/training/seed42.log 2>&1 &
echo $! > logs/training/seed42.pid
```

### 7.2 Log Organization

```
logs/
├── training/
│   ├── seed42.log
│   ├── seed123.log
│   └── ...
├── evaluation/
│   ├── baselines.log
│   ├── metrics_seed42.csv
│   └── ...
└── testing/
    └── smoke_test.log
```

### 7.3 Log Content Requirements

**Must Log**:
- Timestamp for each major step
- Data statistics (sample counts, class balance)
- Hyperparameters
- Training loss per epoch
- Validation metrics per epoch
- Final test metrics
- Checkpoint paths

---

## 8. Output Requirements

### 8.1 Main Results Table

**Format**: CSV with columns
```
Model, AUPRC_mean, AUPRC_std, AUROC_mean, AUROC_std, F1_mean, F1_std, ATE_Bias_mean, ...
```

### 8.2 Supplementary Tables

- Threshold-dependent metrics (Precision, Recall, Specificity, NPV)
- Calibration metrics (intercept, slope, Brier)
- Efficiency metrics (Params, Inference time)
- TSTR/TRTR results (for generative baselines)

### 8.3 Figures (Must Generate)

1. **Figure 1**: ROC curves (all models)
2. **Figure 2**: PR curves (all models)
3. **Figure 3**: Calibration plots (all models)
4. **Figure 4**: Decision curves (all models)
5. **Figure 5**: Confusion matrices (best model)
6. **Figure 6**: Risk trajectory examples (Causal-TabDiff)

---

## 9. Smoke Test Protocol

**Purpose**: Verify pipeline integrity before long experiments

**Procedure**:
```bash
python run_experiment.py --debug_mode --epochs 2
python run_baselines.py --debug_mode --max_samples 100
```

**Checks**:
- [ ] Data loads without errors
- [ ] Shapes are correct
- [ ] No NaN in predictions
- [ ] Metrics compute without errors
- [ ] Plots generate successfully

---

## 10. Experiment Checklist

### Before Training
- [ ] Master modeling table constructed
- [ ] Leakage validation passed
- [ ] Data statistics logged
- [ ] Train/val/test splits verified
- [ ] Smoke test passed

### During Training
- [ ] Losses decreasing
- [ ] Validation metrics improving
- [ ] No NaN/Inf in gradients
- [ ] Checkpoints saving correctly

### After Training
- [ ] Best checkpoint identified
- [ ] Test evaluation completed
- [ ] All metrics computed
- [ ] All plots generated
- [ ] Results logged

### Multi-Seed
- [ ] All 5 seeds completed
- [ ] Metrics aggregated
- [ ] Mean ± std reported
- [ ] Seed variance acceptable

---

## 11. Reproducibility Requirements

**Must Document**:
- Python version
- PyTorch version
- CUDA version (if GPU used)
- All package versions (requirements.txt)
- Random seeds
- Hardware specs (CPU/GPU)
- Training time per seed

**Must Provide**:
- Trained model checkpoints
- Evaluation scripts
- Plotting scripts
- Raw metric CSVs

---

## 12. Failure Recovery

**If training crashes**:
1. Check logs for error
2. Verify data integrity
3. Reduce batch size if OOM
4. Resume from last checkpoint if available

**If metrics are poor**:
1. Check data leakage
2. Verify label construction
3. Inspect predictions (are they all 0 or 1?)
4. Check class imbalance handling

---

## 13. Timeline Estimate

**Data Construction**: 1-2 hours  
**Single Model Training**: 2-4 hours (GPU)  
**5-Seed Training**: 10-20 hours  
**Baseline Adaptation**: 1-2 days  
**Full Evaluation**: 1 day  

**Total**: ~1 week for complete experimental cycle

---

## 14. Revision History

- **v1.0 (2026-03-10)**: Initial protocol defining complete workflow

## 15. B1-2: Model Interface Alignment (COMPLETED)

**Status**: ✅ Completed (2026-03-11)  
**Report**: See `docs/reboot/B1_2_SMOKE_TEST_REPORT.md`

### 15.1 Data Module Update

**实现**: `src/data/data_module_landmark.py` (完全重写)

**核心改进**:
- 直接使用 `unified_person_landmark_table.pkl`
- 真实短历史提取（无 padding 伪造）
- Pid-level split (60%/20%/20%)
- 变长序列 collate function

**API**:
```python
from src.data.data_module_landmark import get_dataloader

loader = get_dataloader(
    table_path='data/landmark_tables/unified_person_landmark_table.pkl',
    split='train',
    batch_size=32,
    seed=42,
    debug_n_persons=None  # None = full dataset
)
```

**Batch 结构**:
```python
{
    'x': (batch, 3, 15),  # padded to max_time=3
    'y_2year': (batch, 1),
    'trajectory_target': (batch, 7),
    'trajectory_valid_mask': (batch, 7),
    'landmark': (batch, 1),
    'history_length': (batch, 1),
    'pid': list[int]
}
```

### 15.2 Model Update

**实现**: `src/models/causal_tabdiff_trajectory.py`

**修复**:
- 添加 `cond_dim=1` 参数传递
- 修复特征提取（使用 `block1` 而非不存在的 `encoder`）
- 保持 trajectory + 2-year risk 双输出

**输出**:
```python
{
    'diff_loss': scalar,
    'disc_loss': scalar,
    'trajectory': (batch, 7),
    'risk_2year': (batch, 1)
}
```

### 15.3 Loss 定义

```python
total_loss = diff_loss + 0.5 * disc_loss + loss_traj + loss_2year

# 其中:
# loss_traj = BCE(trajectory * mask, target * mask)
# loss_2year = BCE(risk_2year, y_2year)
```

### 15.4 Smoke Test

**命令**:
```bash
conda activate causal_tabdiff
python smoke_test_b1_2.py
```

**验证项**:
- ✅ 数据加载
- ✅ 模型前向
- ✅ Loss 计算
- ✅ 反向传播
- ✅ 2 Epoch 训练
- ✅ Validation
- ✅ 最小指标输出

### 15.5 当前主入口

**训练**: `run_experiment_landmark.py`  
**测试**: `smoke_test_b1_2.py`  
**数据**: `src/data/data_module_landmark.py`  
**模型**: `src/models/causal_tabdiff_trajectory.py`

**Legacy 入口**: 见 `docs/reboot/LEGACY_ENTRYPOINTS.md`

---

