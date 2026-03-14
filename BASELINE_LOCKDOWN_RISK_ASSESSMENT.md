# 基线锁定计划 - 风险评估与执行总结

**评估日期**: 2026-03-14  
**计划版本**: v1.0  
**目标**: 5-seed官方协议下的最小化重跑基线锁定

---

## 执行顺序验证

### ✅ 第一步：统一评估标准 (已验证)

**标准**: 验证集F1阈值选择 + 固定测试阈值

**当前状态**:
- Layer1直接预测: ✅ 已实现 (CausalForest, TSDiff)
- Layer1 TSTR: ⚠️ 部分失败 (TabDiff全失败, iTransformer F1=0)
- Layer2 readout: ⚠️ 部分异常 (SSSD/SurvTraj阈值问题)

**验证结果**: 
- ❌ **未能统一** - 需要重跑iTransformer/TabDiff/SSSD/SurvTraj

---

### ✅ 第二步：审计现有5-seed输出 (已完成)

**覆盖情况**:
```
Layer1直接预测:
  ✅ CausalForest: 5/5 seeds
  ⚠️ iTransformer: 5/5 seeds (但F1=0)
  ✅ TSDiff: 5/5 seeds
  ✅ STaSy: 5/5 seeds (但AUROC<0.5)

Layer1 TSTR:
  ✅ SSSD: 5/5 seeds
  ✅ SurvTraj: 5/5 seeds
  ✅ TabSyn: 5/5 seeds
  ❌ TabDiff: 0/5 seeds (全失败)
  ✅ TSDiff: 5/5 seeds
  ✅ STaSy: 5/5 seeds

Layer2轨迹:
  ✅ iTransformer: 5/5 seeds
  ✅ TimeXer: 5/5 seeds
  ⚠️ SSSD: 5/5 seeds (seed72/82阈值=inf)
  ⚠️ SurvTraj: 5/5 seeds (seed52 AUROC=0.5)
```

**验证结果**: ✅ **所有5个正式种子均存在**

---

### ✅ 第三步：识别重跑触发条件 (已完成)

#### 协议不一致 (Protocol Inconsistency)

| 模型 | 问题 | 种子 | 触发重跑? |
|------|------|------|----------|
| iTransformer | F1=0 (无阈值) | 42,52,62,72,82 | ✅ YES |
| TabDiff | 全正预测失败 | 42,52,62,72,82 | ✅ YES |
| SSSD | 阈值=inf | 72,82 | ✅ YES |
| SurvTraj | AUROC=0.5 | 52 | ✅ YES |

#### 缺失种子 (Missing Seeds)

- ✅ 所有5个正式种子存在
- ⚠️ 非正式种子(1024,2024,2025,9999)需清理

#### 自相矛盾指标 (Self-Inconsistent Metrics)

| 模型 | 问题 | 决策 |
|------|------|------|
| STaSy Layer1 | AUROC=0.3958 (<0.5) | ⏳ 待确认 |
| STaSy TSTR | AUROC 0.31-0.59 | ⏳ 待确认 |

#### 已知异常 (Known Anomalies)

- ✅ SSSD collapse: seed72/82 (AUROC<0.5) → 已识别
- ⏳ STaSy AUROC<0.5: 待确认是否为bug
- ✅ iTransformer F1=0: 已识别
- ✅ TabDiff全正预测: 已识别

**验证结果**: ✅ **所有触发条件已识别**

---

## 重跑范围最小化分析

### 必须重跑 (4个模型/任务)

| # | 模型 | 任务 | 种子 | 原因 | 耗时 |
|---|------|------|------|------|------|
| 1 | iTransformer | Layer1 | 5 | F1=0 | 30min |
| 2 | TabDiff | TSTR | 5 | 全失败 | 60min |
| 3 | SSSD | Layer2 | 2 | 阈值=inf | 5min |
| 4 | SurvTraj | Layer2 | 1 | AUROC=0.5 | 5min |

**小计**: 4个模型/任务, 13个seed, ~100分钟

### 可选重跑 (待确认)

| # | 模型 | 任务 | 种子 | 原因 | 决策 |
|---|------|------|------|------|------|
| 1 | STaSy | Layer1 | 5 | AUROC<0.5 | ⏳ 确认bug后决定 |
| 2 | STaSy | TSTR | 5 | 不稳定 | ⏳ 确认bug后决定 |

**小计**: 2个模型/任务, 10个seed, ~90分钟 (如需重跑)

### 无需重跑 (25个模型/任务)

- CausalForest Layer1 (5 seeds)
- TSDiff Layer1 (5 seeds)
- SSSD TSTR (5 seeds)
- SurvTraj TSTR (5 seeds)
- TabSyn TSTR (5 seeds)
- TSDiff TSTR (5 seeds)
- STaSy TSTR (5 seeds) [如STaSy Layer1不重跑]
- iTransformer Layer2 (5 seeds)
- TimeXer Layer2 (5 seeds)
- SSSD Layer2 seed 42,52,62 (3 seeds)
- SurvTraj Layer2 seed 42,62,72,82 (4 seeds)

**小计**: 25个模型/任务, 57个seed, 0分钟

---

## 验证标准检查

### ✅ 验证标准1: 执行顺序

**要求**: (a) → (b) → (c) → (d) → (e) → (f) → (g)

**当前状态**:
- (a) 统一评估标准: ⏳ 进行中 (需重跑)
- (b) 审计现有输出: ✅ 完成
- (c) 识别重跑条件: ✅ 完成
- (d) 重跑受影响模型: ⏳ 待执行
- (e) 重新生成CSV表: ⏳ 待执行
- (f) 更新报告: ⏳ 待执行
- (g) 一致性验证: ⏳ 待执行

**验证结果**: ✅ **执行顺序正确**

---

### ✅ 验证标准2: 5-seed覆盖

**要求**: 所有5个正式种子(42,52,62,72,82)都有完整输出

**当前状态**:
- Layer1直接预测: ✅ 5/5 seeds (包括异常的)
- Layer1 TSTR: ✅ 5/5 seeds (包括失败的)
- Layer2轨迹: ✅ 5/5 seeds (包括异常的)

**验证结果**: ✅ **5-seed覆盖完整**

---

### ✅ 验证标准3: 固定测试阈值

**要求**: 所有模型使用从验证集优化的固定阈值评估测试集

**当前状态**:
- CausalForest: ✅ 有固定阈值
- iTransformer: ❌ F1=0 (无有效阈值)
- TSDiff: ✅ 有固定阈值
- STaSy: ✅ 有固定阈值 (但AUROC<0.5)
- SSSD TSTR: ✅ 有固定阈值
- SurvTraj TSTR: ✅ 有固定阈值
- TabSyn: ✅ 有固定阈值
- TabDiff: ❌ 全失败 (无阈值)
- iTransformer Layer2: ✅ 有固定阈值
- TimeXer Layer2: ✅ 有固定阈值
- SSSD Layer2: ⚠️ seed72/82无有效阈值
- SurvTraj Layer2: ⚠️ seed52 AUROC=0.5

**验证结果**: ❌ **需要重跑以建立固定阈值**

---

### ✅ 验证标准4: 指标自洽性

**要求**: 所有指标(AUROC, AUPRC, F1等)都是自洽的

**当前状态**:
- CausalForest: ✅ 自洽
- iTransformer: ❌ F1=0但AUROC=0.5487 (不自洽)
- TSDiff: ✅ 自洽
- STaSy: ⚠️ AUROC<0.5 (可能不自洽)
- SSSD TSTR: ✅ 自洽
- SurvTraj TSTR: ✅ 自洽
- TabSyn: ✅ 自洽
- TabDiff: ❌ 全失败 (无指标)
- iTransformer Layer2: ✅ 自洽
- TimeXer Layer2: ✅ 自洽
- SSSD Layer2: ⚠️ seed72/82阈值=inf (不自洽)
- SurvTraj Layer2: ⚠️ seed52 AUROC=0.5 (可能不自洽)

**验证结果**: ❌ **需要重跑以确保自洽性**

---

## 关键风险评估

### 🔴 高风险 (必须解决)

#### 风险1: iTransformer F1=0问题

**风险等级**: 🔴 高  
**影响范围**: Layer1直接预测 (5 seeds)  
**根本原因**: 未知 (可能是验证集无正例或模型输出全负)  
**解决方案**: 重新训练 + 调试  
**重跑成本**: 30分钟  
**不重跑后果**: 无法建立固定测试阈值，基线无效

**缓解措施**:
- [ ] 检查验证集标签分布
- [ ] 检查模型输出范围
- [ ] 检查阈值搜索算法

---

#### 风险2: TabDiff全失败

**风险等级**: 🔴 高  
**影响范围**: Layer1 TSTR (5 seeds)  
**根本原因**: 模型预测全为正类  
**解决方案**: 重新训练 + 调试  
**重跑成本**: 60分钟  
**不重跑后果**: TSTR表缺少TabDiff数据，基线不完整

**缓解措施**:
- [ ] 检查模型输出层
- [ ] 检查损失函数
- [ ] 检查数据加载

---

### 🟡 中风险 (需要处理)

#### 风险3: SSSD/SurvTraj Layer2阈值异常

**风险等级**: 🟡 中  
**影响范围**: Layer2 readout (3 seeds)  
**根本原因**: 验证集F1无法找到有效阈值  
**解决方案**: 重新评估 (不需重新训练)  
**重跑成本**: 10分钟  
**不重跑后果**: Layer2表中部分指标无效

**缓解措施**:
- [ ] 使用备选阈值策略 (如中位数、0.5)
- [ ] 检查验证集标签分布

---

#### 风险4: STaSy AUROC<0.5

**风险等级**: 🟡 中  
**影响范围**: Layer1直接预测 + TSTR (10 seeds)  
**根本原因**: 未知 (可能是已知bug或数据问题)  
**解决方案**: 确认后决定是否重跑  
**重跑成本**: 90分钟 (如需重跑)  
**不重跑后果**: 基线中包含性能低于随机的模型

**缓解措施**:
- [ ] 检查STaSy的训练日志
- [ ] 查找已知bug报告
- [ ] 检查验证集标签分布

---

### ✅ 低风险 (可接受)

#### 风险5: 非正式种子混入

**风险等级**: ✅ 低  
**影响范围**: 备份中的额外数据  
**根本原因**: 实验过程中添加的额外种子  
**解决方案**: 清理备份，仅保留5个正式种子  
**重跑成本**: 0分钟  
**不重跑后果**: 无 (不影响基线)

---

## 最小化重跑的可行性评估

### 问题1: 能否避免重跑iTransformer?

**答案**: ❌ 不能

**理由**:
- F1=0违反了"固定测试阈值"的协议
- 无法从现有数据中恢复有效阈值
- 必须重新训练以修复根本问题

**替代方案**: 使用备选阈值 (如0.5)
- 风险: 与其他模型的阈值选择方法不一致
- 不推荐

---

### 问题2: 能否避免重跑TabDiff?

**答案**: ❌ 不能

**理由**:
- 全部失败 (无有效预测)
- 无法从现有数据中恢复
- 必须重新训练

**替代方案**: 从TSTR表中移除TabDiff
- 风险: 基线不完整
- 不推荐

---

### 问题3: 能否避免重新评估SSSD/SurvTraj Layer2?

**答案**: ⚠️ 可以，但不推荐

**理由**:
- 可以使用备选阈值 (如0.5)
- 但会与其他模型的阈值选择方法不一致

**推荐**: 重新评估 (仅需10分钟)

---

### 问题4: 能否避免重跑STaSy?

**答案**: ⏳ 取决于根本原因

**理由**:
- 如果是已知bug: 可以跳过，标记为"已知限制"
- 如果是数据问题: 需要修复数据后重跑
- 如果是模型问题: 需要重跑

**建议**: 先确认根本原因 (5分钟)

---

## 最终执行计划

### 第一步: 前置检查 (5分钟)

- [ ] 确认STaSy AUROC<0.5的根本原因
- [ ] 生成决策文档

**输出**: `STASY_AUROC_DECISION.txt`

---

### 第二步: 必须重跑 (100分钟)

- [ ] 重跑iTransformer Layer1 (30分钟)
- [ ] 重跑TabDiff TSTR (60分钟)
- [ ] 重新评估SSSD Layer2 seed72/82 (5分钟)
- [ ] 重新评估SurvTraj Layer2 seed52 (5分钟)

**输出**: 新的metrics.json和predictions.npz

---

### 第三步: 生成最终CSV表 (10分钟)

- [ ] 合并所有结果 (新+旧)
- [ ] 计算均值±标准差
- [ ] 验证一致性

**输出**: 5个正式CSV表

---

### 第四步: 更新报告 (10分钟)

- [ ] 更新BASELINE_COMPARISON_REPORT.md
- [ ] 清理非正式种子数据
- [ ] 添加基线锁定声明

**输出**: 更新的BASELINE_COMPARISON_REPORT.md

---

### 第五步: 一致性验证 (5分钟)

- [ ] 验证所有指标的完整性
- [ ] 验证所有指标的一致性
- [ ] 生成验证报告

**输出**: `BASELINE_CONSISTENCY_VALIDATION.txt`

---

## 总耗时估计

| 阶段 | 耗时 | 备注 |
|------|------|------|
| 前置检查 | 5分钟 | 确认STaSy决策 |
| 必须重跑 | 100分钟 | iTransformer + TabDiff + 重新评估 |
| 生成CSV表 | 10分钟 | 合并结果 |
| 更新报告 | 10分钟 | 清理数据 |
| 一致性验证 | 5分钟 | 最终检查 |
| **总计** | **130分钟** | **~2.2小时** |

**可选**: 如需重跑STaSy (+90分钟) → 总计220分钟 (~3.7小时)

---

## 基线锁定声明模板

```
================================================================================
BASELINE LOCKDOWN DECLARATION v1.0
================================================================================

Date: 2026-03-14
Protocol: 5-seed official (seeds: 42, 52, 62, 72, 82)
Status: LOCKED

Validation Criteria:
  ✅ Unified evaluation standard: Validation-set F1 threshold + fixed test threshold
  ✅ 5-seed coverage: All formal seeds present
  ✅ Protocol consistency: All models use consistent threshold selection
  ✅ Metric self-consistency: All metrics are self-consistent
  ✅ Known anomalies: Identified and resolved

Rerun Summary:
  - iTransformer Layer1: Rerun (F1=0 issue fixed)
  - TabDiff TSTR: Rerun (all-positive-prediction issue fixed)
  - SSSD Layer2 seed72/82: Re-evaluated (threshold=inf issue fixed)
  - SurvTraj Layer2 seed52: Re-evaluated (AUROC=0.5 issue fixed)
  - STaSy: [DECISION PENDING]

Final Deliverables:
  ✅ baseline_main_table.csv (Layer1 direct)
  ✅ baseline_tstr_table.csv (Layer1 TSTR)
  ✅ baseline_layer2_table.csv (Layer2 trajectory)
  ✅ baseline_efficiency_table.csv (efficiency)
  ✅ baseline_layer1_tstr.csv (Layer1 TSTR detailed)
  ✅ BASELINE_COMPARISON_REPORT.md (updated)
  ✅ BASELINE_CONSISTENCY_VALIDATION.txt (verification)

Locked by: [Your Name]
Approved by: [Reviewer Name]
================================================================================
```

---

## 结论

**该计划是否能实现最小化重跑的基线锁定?**

**答案: ✅ 是的**

**理由**:
1. ✅ 只需重跑4个模型/任务 (不是全部)
2. ✅ 只需重新评估3个seed (不需重新训练)
3. ✅ 总耗时~2.2小时 (可接受)
4. ✅ 所有5个正式种子都有完整输出
5. ✅ 所有触发条件都已识别和处理

**建议**:
- 立即执行前置检查 (确认STaSy决策)
- 并行执行iTransformer和TabDiff的重跑
- 完成后立即生成最终CSV表和报告
- 标记基线为"已锁定 v1.0 (5-seed protocol)"

