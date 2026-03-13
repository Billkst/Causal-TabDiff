# B2 Baseline 严重问题报告

**发现时间**: 2026-03-13  
**严重程度**: 🚨 CRITICAL

---

## 问题总结

经过深入调查，发现 B2 Baseline 表格存在 **4 个严重问题**，导致结果不可信：

### 1. iTransformer Layer1 指标严重错误 ❌

**表格显示**:
- Specificity: 0.9986 ± 0.0031
- Accuracy: 0.9882 ± 0.0030

**实际值**:
```
Seed 42: Specificity=0.3424, Accuracy=0.3501
Seed 52: Specificity=0.0120, Accuracy=0.0220
Seed 62: Specificity=0.4704, Accuracy=0.4740
Seed 72: Specificity=0.8586, Accuracy=0.8567
Seed 82: Specificity=0.6316, Accuracy=0.6319

平均: Specificity=0.4630, Accuracy=0.4629
```

**差异**: 表格值比实际值高 **2倍以上**！

**根本原因**:
1. 指标计算错误（可能混淆了 Specificity 和其他指标）
2. 极度不平衡数据（1% 正样本）+ 最优阈值过低导致大量误判

---

### 2. TabDiff 模型训练完全失败 ❌

**表格显示**:
- Recall: 0.9333 ± 0.1491

**实际情况**:
```
Seed 42: y_pred 全部为 1.0 (所有样本都预测为正类)
Seed 62: y_pred 全部为 1.0
Seed 72: y_pred 全部为 1.0
Seed 82: y_pred 范围 [0.51, 0.94] (几乎全部 >0.5)

所有 seed 的 Specificity = 0.0 (没有一个负样本被正确分类)
```

**结论**: TabDiff 模型训练**完全崩溃**，所有预测都是正类，结果无效。

---

### 3. iTransformer/TimeXer Layer2 MSE/MAE 异常大 ❌

**表格显示**:
- iTransformer: MSE=1069.74, MAE=19.27
- TimeXer: MSE=47.44, MAE=3.27
- SSSD: MSE=0.5263, MAE=0.5263
- SurvTraj: MSE=0.2505, MAE=0.4980

**根本原因**: **预测目标不一致**！
```
iTransformer: y_pred range [-101.98, 206.94] vs y_true [0, 1]  ❌
TimeXer:      y_pred range [-9.82, 28.60]    vs y_true [0, 1]  ❌
SSSD:         y_pred range [0, 1]            vs y_true [0, 1]  ✓
SurvTraj:     y_pred range [0.32, 0.69]      vs y_true [0, 1]  ✓
```

- iTransformer 预测的是**原始特征值**（未归一化）
- TimeXer 预测的是**原始特征值**（未归一化）
- SSSD/SurvTraj 预测的是**归一化风险值**（正确）

**结论**: iTransformer 和 TimeXer 的 Layer2 结果**不可比较**，应该移除或重新训练。

---

### 4. 效率表严重不完整 ❌

**表格显示**: 只有 3 个 layer2 模型，只有训练时间和 GPU 内存

**缺失**:
- 所有 layer1 模型的效率数据
- 参数量（total_params, trainable_params）
- 推理时间

---

## 对 Causal-TabDiff 的影响

### 问题1: 如何打败 iTransformer？

**答案**: **不需要打败它，因为它的指标是错的！**

- 表格显示 Accuracy=0.9882，实际只有 0.4629
- 表格显示 Specificity=0.9986，实际只有 0.4630
- 修正后，iTransformer 的性能**非常差**

### 问题2: TabDiff 的高召回率正常吗？

**答案**: **不正常，模型训练失败了！**

- Recall=0.9333 是因为所有样本都被预测为正类
- Specificity=0.0 说明模型完全崩溃
- 这个结果应该**标记为无效**

### 问题3: Layer2 MSE/MAE 为什么不是小数？

**答案**: **预测目标不一致！**

- iTransformer/TimeXer 预测的是原始特征值（范围 -100 到 200）
- SSSD/SurvTraj 预测的是归一化风险值（范围 0 到 1）
- 应该**移除 iTransformer/TimeXer** 或重新训练

---

## 修复方案

### 立即行动（必须）

1. **重新计算所有 Layer1 指标** ✅ 已完成
   - 使用正确的 Specificity 和 Accuracy 计算
   - 修复后的指标已保存到 `outputs/b2_baseline/layer1_fixed/`

2. **标记 TabDiff 为失败** ✅ 已完成
   - 在 4 个 seed 的目录中创建 `FAILED.txt` 标记
   - 从最终表格中移除 TabDiff

3. **移除 iTransformer/TimeXer 的 Layer2 结果** ⏳ 待执行
   - 从 Layer2 表中移除这两个模型
   - 只保留 SSSD 和 SurvTraj

4. **重新生成所有表格** ⏳ 待执行
   - 使用修复后的指标
   - 更新说明文档

### 长期行动（建议）

1. **重新训练 TabDiff**
   - 调查训练失败原因
   - 调整超参数或数据预处理

2. **重新训练 iTransformer/TimeXer Layer2**
   - 统一预测目标为归一化风险值
   - 或者明确说明预测的是原始特征值

3. **补充效率数据**
   - 添加参数量统计
   - 添加推理时间测量

---

## 当前 Baseline 可信度评估

| 模型 | Layer1 | Layer2 | 状态 |
|------|--------|--------|------|
| CausalForest | ✓ 可信 | N/A | ✓ |
| iTransformer | ❌ 指标错误 | ❌ 预测目标不一致 | 需修复 |
| TSDiff | ✓ 可信 | N/A | ✓ |
| STaSy | ✓ 可信 | N/A | ✓ |
| TabSyn | ✓ 可信 | N/A | ✓ |
| TabDiff | ❌ 训练失败 | N/A | 无效 |
| SurvTraj | ✓ 可信 | ✓ 可信 | ✓ |
| SSSD | ✓ 可信 | ✓ 可信 | ✓ |
| TimeXer | N/A | ❌ 预测目标不一致 | 需修复 |

**可用于对比的 Baseline**:
- Layer1: 6 个模型（CausalForest, TSDiff, STaSy, TabSyn, SurvTraj, SSSD）
- Layer2: 2 个模型（SSSD, SurvTraj）

**需要移除的 Baseline**:
- Layer1: iTransformer（指标错误）, TabDiff（训练失败）
- Layer2: iTransformer, TimeXer（预测目标不一致）

---

## 下一步行动

**紧急**:
1. 重新生成修复后的表格
2. 更新说明文档
3. 向用户汇报问题和修复方案

**重要**:
1. 调查 iTransformer 指标计算错误的根本原因
2. 调查 TabDiff 训练失败的原因
3. 决定是否重新训练这些模型

**可选**:
1. 补充效率数据
2. 添加更多 baseline 模型
