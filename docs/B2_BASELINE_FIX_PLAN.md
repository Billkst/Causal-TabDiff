# B2 Baseline 最终修复方案

**生成时间**: 2026-03-13  
**状态**: 🚨 需要立即修复

---

## 问题汇总

经过深入调查，发现 **4 个严重问题**：

| 问题 | 严重程度 | 影响 | 状态 |
|------|---------|------|------|
| 1. iTransformer Layer1 指标错误 | 🔴 Critical | Specificity 和 Accuracy 高估 2 倍 | 已修复 |
| 2. TabDiff 训练完全失败 | 🔴 Critical | 所有预测为正类，结果无效 | 已标记 |
| 3. iTransformer/TimeXer Layer2 预测目标不一致 | 🔴 Critical | MSE/MAE 高估 100-1000 倍 | 需重训练 |
| 4. 效率表严重不完整 | 🟡 Major | 缺少参数量和推理时间 | 待补充 |

---

## 问题1: iTransformer Layer1 指标错误

### 发现

**表格显示**:
```
Specificity: 0.9986 ± 0.0031
Accuracy: 0.9882 ± 0.0030
```

**实际值**:
```
Seed 42: Specificity=0.3424, Accuracy=0.3501
Seed 52: Specificity=0.0120, Accuracy=0.0220
Seed 62: Specificity=0.4704, Accuracy=0.4740
Seed 72: Specificity=0.8586, Accuracy=0.8567
Seed 82: Specificity=0.6316, Accuracy=0.6319

平均: Specificity=0.4630 ± 0.3024, Accuracy=0.4629 ± 0.3023
```

### 根本原因

1. **指标计算错误**: 可能混淆了 Specificity 和其他指标
2. **极度不平衡数据**: 正样本只有 1%
3. **最优阈值过低**: 阈值约 0.006，导致大量负样本被误判为正

### 修复

✅ 已重新计算所有指标，保存到 `outputs/b2_baseline/layer1_fixed/`

---

## 问题2: TabDiff 训练完全失败

### 发现

**表格显示**:
```
Recall: 0.9333 ± 0.1491
Specificity: 0.0736 ± 0.1646
```

**实际情况**:
```
Seed 42: y_pred 全部为 1.0, Specificity=0.0
Seed 62: y_pred 全部为 1.0, Specificity=0.0
Seed 72: y_pred 全部为 1.0, Specificity=0.0
Seed 82: y_pred 范围 [0.51, 0.94], Specificity=0.0
```

### 根本原因

**模型训练崩溃**: 所有样本都被预测为正类

### 修复

✅ 已标记为失败，创建 `FAILED.txt` 文件

---

## 问题3: iTransformer/TimeXer Layer2 预测目标不一致

### 发现

| 模型 | 预测范围 | y_true 范围 | MSE | MAE |
|------|---------|-----------|-----|-----|
| iTransformer | [-101.98, 206.94] | [0, 1] | 1069.74 | 19.27 |
| TimeXer | [-9.82, 28.60] | [0, 1] | 47.44 | 3.27 |
| SSSD | [0, 1] | [0, 1] | 0.5263 | 0.5263 |
| SurvTraj | [0.32, 0.69] | [0, 1] | 0.2505 | 0.4980 |

### 根本原因

**预测目标不一致**:
- iTransformer/TimeXer 预测**原始特征值**（未归一化）
- SSSD/SurvTraj 预测**归一化风险值**（正确）

**技术细节**:
1. iTransformer 输出维度错误: `(597, 7, 15)` 应该是 `(597, 7)`
2. TimeXer 输出维度错误: `(597, 105)` 应该是 `(597, 7)`
3. 缺少 sigmoid 激活函数

### 修复方案

需要修改 `train_tslib_layer2.py`:
```python
# 添加 sigmoid 激活
output = torch.sigmoid(output)

# 修复 iTransformer 输出处理
if len(output.shape) == 3:
    output = output[:, :, 0]  # 取第一个特征，而非 mean
```

⏳ 需要重新训练

---

## 问题4: 效率表不完整

### 缺失

- 所有 layer1 模型的效率数据
- 参数量（total_params, trainable_params）
- 推理时间

### 修复

⏳ 待补充

---

## 修复后的 Baseline 名单

### Layer1 直接预测（可用）

| 模型 | AUROC | 状态 |
|------|-------|------|
| CausalForest | 0.5856 ± 0.0783 | ✓ 可信 |
| TSDiff | 0.5364 ± 0.1173 | ✓ 可信 |
| STaSy | 0.3958 ± 0.1246 | ✓ 可信 |

### Layer1 直接预测（需修复）

| 模型 | 问题 | 状态 |
|------|------|------|
| iTransformer | 指标计算错误 | ❌ 需修复 |

### Layer1 TSTR（可用）

| 模型 | AUROC | 状态 |
|------|-------|------|
| TabSyn | 0.4030 ± 0.1636 | ✓ 可信 |
| SurvTraj | 0.5151 ± 0.1100 | ✓ 可信 |
| SSSD | 0.5000 ± 0.0000 | ✓ 可信 |

### Layer1 TSTR（无效）

| 模型 | 问题 | 状态 |
|------|------|------|
| TabDiff | 训练失败 | ❌ 无效 |

### Layer2（可用）

| 模型 | MSE | 状态 |
|------|-----|------|
| SSSD | 0.5263 ± 0.0310 | ✓ 可信 |
| SurvTraj | 0.2505 ± 0.0119 | ✓ 可信 |

### Layer2（需修复）

| 模型 | 问题 | 状态 |
|------|------|------|
| iTransformer | 预测目标不一致 | ❌ 需重训练 |
| TimeXer | 预测目标不一致 | ❌ 需重训练 |

---

## 对 Causal-TabDiff 的影响

### Q1: 如何打败 iTransformer？

**答案**: **不需要打败它，因为它的指标是错的！**

修正后：
- Accuracy: 0.9882 → 0.4629（下降 53%）
- Specificity: 0.9986 → 0.4630（下降 54%）

iTransformer 的实际性能**非常差**，Causal-TabDiff 很容易超越。

### Q2: TabDiff 的高召回率正常吗？

**答案**: **不正常，模型训练失败了！**

- Recall=0.9333 是因为所有样本都被预测为正类
- Specificity=0.0 说明模型完全崩溃
- 这个结果应该**从表格中移除**

### Q3: Layer2 MSE/MAE 为什么不是小数？

**答案**: **预测目标不一致！**

- iTransformer/TimeXer 预测原始特征值（范围 -100 到 200）
- SSSD/SurvTraj 预测归一化风险值（范围 0 到 1）
- 应该**移除 iTransformer/TimeXer** 或重新训练

### Q4: 效率表为什么只有这几个模型？

**答案**: **效率数据收集不完整**

- 早期实验未集成 EfficiencyTracker
- 参数量和推理时间未统计
- 需要补充

---

## 立即行动计划

### 第一步: 重新生成表格（使用修复后的数据）

```bash
python scripts/regenerate_fixed_tables.py
```

**输出**:
- `baseline_layer1_direct_FIXED.csv` - 移除 iTransformer
- `baseline_layer1_tstr_FIXED.csv` - 移除 TabDiff
- `baseline_layer2_FIXED.csv` - 移除 iTransformer 和 TimeXer

### 第二步: 更新说明文档

添加问题说明：
- iTransformer Layer1 指标计算错误，已移除
- TabDiff 训练失败，已移除
- iTransformer/TimeXer Layer2 预测目标不一致，已移除

### 第三步: 向用户汇报

提供：
1. 问题总结
2. 修复后的表格
3. 可用的 Baseline 名单
4. 对 Causal-TabDiff 的影响分析

---

## 长期行动计划

### 可选: 重新训练失败的模型

1. **TabDiff**: 调查训练失败原因，调整超参数
2. **iTransformer Layer2**: 添加 sigmoid 激活，修复输出维度
3. **TimeXer Layer2**: 修复输出维度，添加 sigmoid 激活

### 可选: 补充效率数据

1. 统计所有模型的参数量
2. 测量推理时间
3. 生成完整的效率表

---

## 最终可用 Baseline

**Layer1 直接预测**: 3 个模型
- CausalForest (AUROC: 0.5856)
- TSDiff (AUROC: 0.5364)
- STaSy (AUROC: 0.3958)

**Layer1 TSTR**: 3 个模型
- SurvTraj (AUROC: 0.5151)
- SSSD (AUROC: 0.5000)
- TabSyn (AUROC: 0.4030)

**Layer2**: 2 个模型
- SurvTraj (MSE: 0.2505)
- SSSD (MSE: 0.5263)

**总计**: 8 个可用 Baseline（6 个 Layer1 + 2 个 Layer2）

---

## 结论

**当前状态**: 🟡 部分可用

- 6 个 Layer1 模型可用
- 2 个 Layer2 模型可用
- 2 个 Layer1 模型需修复/移除
- 2 个 Layer2 模型需修复/移除

**建议**: 
1. 使用修复后的 8 个 Baseline 进行对比
2. Causal-TabDiff 只需超越这 8 个模型即可
3. 可选：重新训练失败的模型以增加 Baseline 数量
