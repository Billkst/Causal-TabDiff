# B1-3 纠偏报告

## 执行时间
2026-03-11

## 目标
修复 B1-3 的 4 个阻塞点，使仓库真正达到可进入 B2 的状态。

## 4 个阻塞点及修复情况

### 阻塞点 1: evaluate_model.py 随机采样 alpha_target

**问题**:
- 评估时使用 `torch.rand()` 随机采样 alpha_target
- 导致验证集和测试集指标不稳定、不可复现
- 不符合正式评估要求

**修复**:
✅ **已修复**

**修复方案**:
- **训练时**: 随机采样 alpha_target ∈ [0.1, 0.9]
- **评估时**: 固定使用 alpha_target = 0.5 (中性条件，可复现)

**实现位置**: `evaluate_model.py::get_ours_predictions()`
```python
alpha_target = torch.full((x.shape[0], 1), alpha_target_value, device=device)
```

**可复现性保证**:
- 评估时使用固定值 0.5
- 可通过 `--alpha_target` 参数指定其他固定值
- 不再依赖随机数生成器

---

### 阻塞点 2: evaluate_model.py hardcode 到 ours

**问题**:
- 评估入口 hardcode 到 `CausalTabDiffTrajectory`
- 不是真正的统一评估入口
- baseline 无法使用

**修复**:
✅ **已修复**

**修复方案**:
重写 `evaluate_model.py` 为真正统一的评估入口，支持两种模式：

**模式 1: Ours 模型评估**
```bash
python evaluate_model.py --model_type ours --model_path checkpoints/model.pt
```

**模式 2: Baseline 模型评估**
```bash
python evaluate_model.py --model_type baseline --predictions_file baseline_predictions.npz
```

**核心设计**:
- `evaluate_from_predictions()` - 通用评估函数，接收预测结果
- `get_ours_predictions()` - Ours 模型预测生成
- Baseline 通过 `.npz` 文件提供预测结果

**共享逻辑**:
- ✅ 相同指标函数 (`src/evaluation/metrics.py`)
- ✅ 相同阈值选择逻辑 (验证集 F1 最大化)
- ✅ 相同图表输出逻辑 (`src/evaluation/plots.py`)

---

### 阻塞点 3: ATE_Bias / Wasserstein / CMD 未接入新框架

**问题**:
- 因果和分布指标只存在于旧 legacy 入口
- 在新评估框架中没有明确位置

**修复**:
✅ **已修复**

**修复方案**:
创建独立的因果评估模块

**新增文件**: `src/evaluation/causal_metrics.py`

**实现指标**:
- ✅ `compute_ate_bias()` - 平均处理效应偏差
- ✅ `compute_wasserstein()` - Wasserstein 距离
- ✅ `compute_cmd()` - Central Moment Discrepancy
- ✅ `evaluate_causal_and_distribution()` - 统一评估接口

**在新框架中的位置**:
- **主任务评估**: `evaluate_model.py` (AUPRC, AUROC, F1, Brier, Calibration)
- **因果诊断评估**: `src/evaluation/causal_metrics.py` (ATE_Bias, Wasserstein, CMD)

**使用方式**:
```python
from src.evaluation.causal_metrics import evaluate_causal_and_distribution
metrics = evaluate_causal_and_distribution(model, dataloader, device, output_dir)
```

---

### 阻塞点 4: Decision Curve 未实现

**问题**:
- Decision Curve / Net Benefit 标记为 TODO
- 但声称 B1-3 已完成

**修复**:
✅ **已修复**

**修复方案**:
完整实现 Decision Curve

**新增文件**: `src/evaluation/decision_curve.py`

**实现功能**:
- ✅ `compute_net_benefit()` - 计算净收益
- ✅ `compute_decision_curve()` - 计算决策曲线数据
- ✅ `plot_decision_curve()` - 绘制决策曲线

**集成到评估流程**:
- 已集成到 `src/evaluation/plots.py::generate_all_plots()`
- 自动生成 `decision_curve.png`

**数学公式**:
```
Net Benefit = (TP/N) - (FP/N) × (threshold / (1 - threshold))
```

---

## 修复总结

### 已完成的修复
1. ✅ alpha_target 评估来源 - 固定为 0.5 (可复现)
2. ✅ 统一评估入口 - 支持 ours 和 baseline
3. ✅ 因果指标接入 - 独立模块 `causal_metrics.py`
4. ✅ Decision Curve - 完整实现

### 新增/修改文件清单

**新增文件**:
- `src/evaluation/causal_metrics.py` - 因果和分布指标
- `src/evaluation/decision_curve.py` - 决策曲线
- `docs/reboot/B1_3_CORRECTION_REPORT.md` - 本文件

**修改文件**:
- `evaluate_model.py` - 重写为统一评估入口
- `src/evaluation/plots.py` - 集成 decision curve

---

## 当前状态

### 无阻塞性问题
所有 4 个阻塞点已全部修复。

### 评估体系完整性
**主任务指标** (已实现):
- AUROC, AUPRC, F1, Precision, Recall, Specificity, NPV
- Accuracy, Balanced Accuracy, MCC
- Brier Score, Calibration Intercept, Calibration Slope
- Confusion Matrix

**图表** (已实现):
- ROC Curve, PR Curve
- Confusion Matrix (raw + normalized)
- Calibration Plot
- Decision Curve

**因果诊断指标** (已实现):
- ATE Bias, Wasserstein, CMD

### 评估协议明确性
**alpha_target 来源**:
- 训练: 随机 ∈ [0.1, 0.9]
- 评估: 固定 = 0.5

**阈值选择**:
- 验证集 F1 最大化
- 测试集使用固定阈值

**统一评估口径**:
- Baseline 和 ours 共享相同指标、阈值、图表逻辑

---

## 是否具备进入 B2 的条件

✅ **是的**，满足所有条件：

1. ✅ 评估可复现 (alpha_target 固定)
2. ✅ 统一评估入口 (支持 baseline 和 ours)
3. ✅ 因果指标有明确位置
4. ✅ Decision Curve 已实现
5. ✅ 无剩余阻塞点

**可以进入 B2: 正式实验准备阶段。**
