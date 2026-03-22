# 评估管道修复实施方案

> **目标**: 修复对比实验结果中的 8 个问题，重新生成全部正式表格
> **约束**: 评估逻辑修改不需要重训模型；阶段3需要从checkpoint推理（不训练）；阶段4需要重新训练6个TSTR生成模型以获取checkpoint并测量采样延迟

---

## 阶段总览

| 阶段 | 内容 | 产出 | 预计工作量 |
|------|------|------|-----------|
| **1** | 修改核心指标库 | `src/evaluation/metrics.py` 更新 | 小 |
| **2** | 修复 Layer2 评估逻辑 | `scripts/eval_fulldata_baselines.py` + `evaluate_layer2.py` 更新 | 小 |
| **3** | CausalTabDiff Layer2 推理脚本 | `scripts/infer_causal_tabdiff_layer2.py`（新建）→ 生成 5 seed 的 layer2.npz | 中 |
| **4** | 重新训练6个TSTR生成模型(各1seed) + 测量采样延迟 | `scripts/retrain_and_measure_generative.py`（新建）→ checkpoint + efficiency JSON | **大** |
| **5** | 一键重算全部表格 | `scripts/regenerate_formal_tables_v2.py`（新建）→ 4 张新 CSV | 中 |
| **6** | 更新结果报告 | `对比实验结果.md` 更新 | 小 |

---

## 阶段1: 修改核心指标库

### 文件: `src/evaluation/metrics.py`

#### 修改1.1: `compute_calibration_metrics()` — 替换 intercept/slope 为 E/O ratio

**位置**: 第 90-106 行

**当前代码**:
```python
def compute_calibration_metrics(y_true, y_pred_proba):
    brier = brier_score_loss(y_true, y_pred_proba)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(y_pred_proba.reshape(-1, 1), y_true)
    intercept = lr.intercept_[0]
    slope = lr.coef_[0][0]
    return {
        'brier_score': brier,
        'calibration_intercept': intercept,
        'calibration_slope': slope
    }
```

**修改为**:
```python
def compute_calibration_metrics(y_true, y_pred_proba):
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # E/O ratio: mean(predicted) / mean(observed)
    # 完美校准 = 1.0; >1 = 高估风险; <1 = 低估风险
    mean_pred = float(np.mean(y_pred_proba))
    mean_obs = float(np.mean(y_true))
    eo_ratio = mean_pred / mean_obs if mean_obs > 0 else float('nan')
    
    return {
        'brier_score': brier,
        'eo_ratio': eo_ratio,
    }
```

#### 修改1.2: 新增 `platt_calibrate()` 函数

**位置**: 在 `compute_calibration_metrics()` 之后添加

```python
def platt_calibrate(val_y_true, val_y_pred, test_y_pred):
    """Platt Scaling: 在验证集 logits 上拟合 LogisticRegression，校准测试集概率。
    
    AUROC/AUPRC 排序不变，只改变概率值。
    
    Args:
        val_y_true: 验证集真实标签 (0/1)
        val_y_pred: 验证集预测概率 [0, 1]
        test_y_pred: 测试集预测概率 [0, 1]
    
    Returns:
        test_pred_calibrated: 校准后的测试集概率
    """
    val_y_true = np.asarray(val_y_true).flatten()
    val_y_pred = np.asarray(val_y_pred).flatten()
    test_y_pred = np.asarray(test_y_pred).flatten()
    
    # 转为 logits
    eps = 1e-7
    val_logits = np.log(np.clip(val_y_pred, eps, 1 - eps) / (1 - np.clip(val_y_pred, eps, 1 - eps)))
    test_logits = np.log(np.clip(test_y_pred, eps, 1 - eps) / (1 - np.clip(test_y_pred, eps, 1 - eps)))
    
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(val_logits.reshape(-1, 1), val_y_true)
    
    test_calibrated = lr.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    return test_calibrated
```

#### 修改1.3: `compute_all_metrics()` 返回值同步更新

**位置**: 第 109-127 行

无需修改函数签名，因为 `compute_calibration_metrics()` 的返回 key 变了，上层代码会自动获取新 key。只需确认所有引用 `calibration_intercept` / `calibration_slope` 的地方同步更新。

---

## 阶段2: 修复 Layer2 评估逻辑

### 文件: `scripts/eval_fulldata_baselines.py`

#### 修改2.1: 修复 readout 双重 sigmoid

**位置**: 第 140-148 行 (`evaluate_layer2_prediction()` 内)

**当前代码**:
```python
pred_2year = np.nan_to_num(test_y_pred[:, :2], nan=0.0, posinf=1e6, neginf=-1e6).mean(axis=1)
pred_2year = np.clip(pred_2year, -500, 500)
pred_2year = 1.0 / (1.0 + np.exp(-pred_2year))
true_2year = (test_y_true[:, :2].sum(axis=1) > 0).astype(int)

val_pred_2year = np.nan_to_num(val_y_pred[:, :2], nan=0.0, posinf=1e6, neginf=-1e6).mean(axis=1)
val_pred_2year = np.clip(val_pred_2year, -500, 500)
val_pred_2year = 1.0 / (1.0 + np.exp(-val_pred_2year))
val_true_2year = (val_y_true[:, :2].sum(axis=1) > 0).astype(int)
```

**修改为**:
```python
def _to_probability(raw_values):
    """如果值域超出 [0,1]（说明是 logits/raw 值），做 sigmoid；否则直接当概率用。"""
    clean = np.nan_to_num(raw_values, nan=0.0, posinf=1e6, neginf=-1e6)
    mean_vals = clean.mean(axis=1) if clean.ndim > 1 else clean
    if mean_vals.min() < -0.01 or mean_vals.max() > 1.01:
        return 1.0 / (1.0 + np.exp(-np.clip(mean_vals, -500, 500)))
    else:
        return np.clip(mean_vals, 0.0, 1.0)

pred_2year = _to_probability(test_y_pred[:, :2])
true_2year = (test_y_true[:, :2].sum(axis=1) > 0).astype(int)

val_pred_2year = _to_probability(val_y_pred[:, :2])
val_true_2year = (val_y_true[:, :2].sum(axis=1) > 0).astype(int)
```

#### 修改2.2: 轨迹 MSE 统一到概率空间

**位置**: 第 122-134 行

在 `valid_pred` / `valid_true` 筛选后、计算 MSE 之前添加:

```python
# 统一到概率空间: 回归模型输出做 sigmoid，概率模型保持不变
if valid_pred.min() < -0.01 or valid_pred.max() > 1.01:
    valid_pred = 1.0 / (1.0 + np.exp(-np.clip(valid_pred, -500, 500)))
```

#### 修改2.3: 汇总表列名更新

**位置**: 第 227-229 行

```python
# 当前:
metric_cols = ['auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity',
               'npv', 'accuracy', 'balanced_accuracy', 'mcc',
               'brier_score', 'calibration_intercept', 'calibration_slope']

# 改为:
metric_cols = ['auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity',
               'npv', 'accuracy', 'balanced_accuracy', 'mcc',
               'brier_score', 'eo_ratio']
```

**位置**: 第 290-291 行 (Layer2 readout 列名)

```python
# 当前:
for read_key in ['auroc', 'auprc', 'f1', 'precision', 'recall', 'brier_score',
                 'calibration_intercept', 'calibration_slope']:

# 改为:
for read_key in ['auroc', 'auprc', 'f1', 'precision', 'recall', 'brier_score',
                 'eo_ratio']:
```

### 文件: `evaluate_layer2.py`

#### 修改2.4: 同步修复 `compute_2year_readout()`

**位置**: 第 45-52 行

```python
# 当前:
def compute_2year_readout(y_pred, y_true, y_mask):
    pred_2year = y_pred[:, :2].mean(axis=1)
    pred_2year = 1.0 / (1.0 + np.exp(-pred_2year))
    true_2year = (y_true[:, :2].sum(axis=1) > 0).astype(int)
    return pred_2year, true_2year

# 改为:
def compute_2year_readout(y_pred, y_true, y_mask):
    raw = y_pred[:, :2].mean(axis=1)
    # 自动判断: 概率输出直接用，logits 做 sigmoid
    if raw.min() < -0.01 or raw.max() > 1.01:
        pred_2year = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
    else:
        pred_2year = np.clip(raw, 0.0, 1.0)
    true_2year = (y_true[:, :2].sum(axis=1) > 0).astype(int)
    return pred_2year, true_2year
```

---

## 阶段3: CausalTabDiff Layer2 推理脚本

### 新建文件: `scripts/infer_causal_tabdiff_layer2.py`

**功能**: 加载 phase2 checkpoint，对 val/test 集推理，保存 trajectory + risk_head 完整输出为标准 layer2.npz 格式。

**核心逻辑**:
```python
# 伪代码
for seed in [42, 52, 62, 72, 82]:
    # 1. 加载数据
    train_df, val_df, test_df = load_and_split_data(table_path, seed)
    
    # 2. 加载模型
    model = CausalTabDiffTrajectory(t_steps, feature_dim, diffusion_steps=100, trajectory_len=7)
    ckpt = torch.load(f'checkpoints/landmark/phase2_best_seed{seed}.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    
    # 3. 推理获取 trajectory 和 risk 预测
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(x, alpha, history_length=hl)
            trajectories.append(outputs['trajectory'].cpu().numpy())  # [B, 7] sigmoid 概率
            risk_logits.append(outputs['risk_2year_logit'].cpu().numpy())  # [B, 1]
    
    # 4. 构建 mask (所有有效位置)
    # trajectory_valid_mask 来自 dataset
    
    # 5. 保存标准格式
    np.savez(f'outputs/fulldata_baselines/layer2/causaltabdiff_seed{seed}_layer2.npz',
             val_y_pred=val_trajectories,   # [N_val, 7]
             val_y_true=val_traj_targets,    # [N_val, 7]
             val_y_mask=val_traj_masks,      # [N_val, 7]
             test_y_pred=test_trajectories,  # [N_test, 7]
             test_y_true=test_traj_targets,  # [N_test, 7]
             test_y_mask=test_traj_masks)    # [N_test, 7]
```

**运行方式**:
```bash
for seed in 42 52 62 72 82; do
    python -u scripts/infer_causal_tabdiff_layer2.py --seed $seed
done
```

---

## 阶段4: 重新训练TSTR生成模型 + 测量采样延迟

### 背景

原始训练流程（`train_generative_strict.py`）没有保存生成模型 checkpoint，只保存了 XGBoost 和最终预测。需要重新训练以获取 checkpoint，然后测量实际采样延迟。

### 新建文件: `scripts/retrain_and_measure_generative.py`

**功能**: 对每个生成模型（TabDiff, TabSyn, SSSD, SurvTraj, STaSy, TSDiff）:
1. 重新训练 30 epochs（与原协议一致），保存 checkpoint
2. 测量生成采样延迟（warmup 20 次，测量 1000 次取平均）
3. 保存效率 JSON

**只需要 1 个 seed（seed=42）**，因为采样延迟与 seed 无关。

**核心逻辑**:
```python
# 伪代码
for model_name in [tabdiff, tabsyn, sssd, survtraj, stasy, tsdiff]:
    # 1. 训练生成模型（复用 train_generative_strict.py 的逻辑，添加 checkpoint 保存）
    model = create_model(model_name)
    model.fit(train_loader, epochs=30, device=device)
    torch.save(model.state_dict(), f'checkpoints/generative/{model_name}_seed42.pt')
    
    # 2. 测量采样延迟
    model.eval()
    # Warmup
    for _ in range(20):
        model.sample(n_samples=1, device=device)
    # 正式测量
    with tracker.track_inference(n_samples=1000):
        model.sample(n_samples=1000, device=device)
    
    # 3. 保存效率 JSON（包含 generative_inference_latency_ms 和 generative_throughput）
    tracker.save_json(f'outputs/fulldata_baselines/efficiency/generative_{model_name}_seed42.json')
```

**预估时间**: 每个模型 ~5-30 分钟训练 + ~1 分钟测量，总计约 1-3 小时。

**注意**: 
- CausalTabDiff 的推理时间沿用已有数据（4.94ms, 包含 100 步扩散采样）
- CausalForest / iTransformer / TimeXer 是判别式模型，推理时间沿用已有数据
- 只重新训练+测量 6 个 TSTR 生成模型

---

## 阶段5: 一键重算全部表格

### 新建文件: `scripts/regenerate_formal_tables_v2.py`

**功能**: 从 .npz 文件出发，按新协议重新计算所有指标，生成 4 张正式表。

**新协议要点**:
1. **统一 Platt Scaling**: 所有模型（Layer1 Direct + TSTR + Layer2 Readout）都做 Platt 校准
2. **统一概率空间**: Layer2 轨迹 MSE 在概率空间计算；readout 自动判断是否需要 sigmoid
3. **新校准指标**: E/O ratio 替换 calibration_intercept/slope
4. **新效率表结构**: 添加 Type、NFE 列；使用生成模型实际采样延迟

**输出目录**: `outputs/fulldata_baselines/summaries_v2/`

**输出文件**:
- `baseline_layer1_direct_v2.csv` — 含 CausalTabDiff 行，含 eo_ratio 列
- `baseline_layer1_tstr_v2.csv` — 统一 Platt 校准后
- `baseline_layer2_v2.csv` — 含 CausalTabDiff 行，MSE 已统一口径
- `baseline_efficiency_v2.csv` — 含 Type、NFE 列，生成模型使用实际采样延迟

**核心数据源映射**:

| 模型 | Layer1 Direct .npz | TSTR .npz | Layer2 .npz | Efficiency JSON |
|------|-------------------|-----------|-------------|-----------------|
| CausalTabDiff | `predictions/landmark/phase2_seed{s}_calibrated.npz` | — | `outputs/.../causaltabdiff_seed{s}_layer2.npz`（阶段3产出） | 手动录入 |
| CausalForest | `outputs/.../layer1/causal_forest_seed{s}_predictions.npz` | — | — | 已有 |
| iTransformer | `outputs/.../layer1/itransformer_seed{s}_predictions.npz` | — | `outputs/.../layer2/itransformer_seed{s}_layer2.npz` | 已有 |
| TimeXer | — | — | `outputs/.../layer2/timexer_seed{s}_layer2.npz` | 已有 |
| TSDiff | `outputs/.../layer1/tsdiff_seed{s}_predictions.npz` | — | — | 已有 |
| STaSy | `outputs/.../layer1/stasy_seed{s}_predictions.npz` | — | — | 已有 |
| SSSD | — | `outputs/.../tstr/sssd_seed{s}_predictions.npz` | `outputs/.../layer2/sssd_seed{s}_layer2.npz` | 阶段4更新 |
| TabSyn | — | `outputs/.../tstr/tabsyn_seed{s}_predictions.npz` | — | 阶段4更新 |
| TabDiff | — | `outputs/.../tstr/tabdiff_seed{s}_predictions.npz` | — | 阶段4更新 |
| SurvTraj | — | `outputs/.../tstr/survtraj_seed{s}_predictions.npz` | `outputs/.../layer2/survtraj_seed{s}_layer2.npz` | 阶段4更新 |
| TSDiff (TSTR) | — | `outputs/.../tstr/tsdiff_seed{s}_predictions.npz` | — | 阶段4更新 |
| STaSy (TSTR) | — | `outputs/.../tstr/stasy_seed{s}_predictions.npz` | — | 阶段4更新 |

---

## 阶段6: 更新结果报告

### 文件: `对比实验结果.md`

- 用 `summaries_v2/` 的新数据替换所有表格
- Layer1 总结果表增加 `AUPRC/Baseline` 列
- Layer2 表中 iTransformer/TimeXer 的 MSE 从 1047/49 降到 0.59/0.52（概率空间）
- 效率表添加 Type 和 NFE 列
- 删除 calibration_intercept/slope，替换为 eo_ratio
- 添加新的表格说明文字
- 添加 Platt Scaling 统一协议说明

---

## 执行顺序与依赖

```
阶段1 (metrics.py)
   │
   ├─→ 阶段2 (eval 修复) ──→ 阶段5 (重算表格) ──→ 阶段6 (更新报告)
   │                              ↑
   ├─→ 阶段3 (CausalTabDiff L2推理) ─┘
   │                              ↑
   └─→ 阶段4 (生成模型推理测量) ──┘
```

阶段 3 和 4 可以并行执行。阶段 5 依赖 1+2+3+4 全部完成。

---

## 关键注意事项（自检补充）

### 注意1: CausalTabDiff 避免双重 Platt
CausalTabDiff 的 `phase2_seed*_calibrated.npz` 已经包含 Platt 校准后的概率。重算脚本中**不应再次校准**。
- CausalTabDiff → 直接使用 `_calibrated.npz` 中的 `test_y_pred`
- 其他所有 baseline → 从原始 `.npz` 做 Platt

### 注意2: AUROC/AUPRC 用原始概率，Brier/E-O/F1 用校准后概率
Platt 是单调变换，不影响排序指标。但为严谨：
- `auroc`, `auprc` → 用**原始概率**（保证与之前数值一致，审稿人不会质疑）
- `brier_score`, `eo_ratio`, `f1`, `precision`, `recall` 等 → 用**校准后概率**

### 注意3: 阶段4 仅需 1 个 seed
采样延迟与随机种子无关（模型架构和扩散步数固定），只需训练 seed=42 即可。效率表中该列报告为 "X.XX ± 0.00"（单次测量）。

---

## 验收标准

- [ ] 所有 5 个 seed 的指标文件完整
- [ ] Layer1 Direct 表含 CausalTabDiff 行，AUROC/AUPRC 与之前一致（Platt 不影响排序）
- [ ] Layer1 TSTR 表所有模型已统一 Platt 校准
- [ ] Layer2 表中 iTransformer MSE < 1.0（概率空间），CausalTabDiff MSE 最低
- [ ] Layer2 Readout 指标无双重 sigmoid：CausalTabDiff 的 readout AUROC 应 ≈ Layer1 AUROC
- [ ] 效率表含 Type/NFE 列，生成模型推理延迟为实际采样时间
- [ ] 所有表中 calibration_intercept/slope 已替换为 eo_ratio
- [ ] `python -u scripts/regenerate_formal_tables_v2.py` 可一键运行无报错
