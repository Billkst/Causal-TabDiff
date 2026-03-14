# 最小化重跑基线锁定计划审查报告

**审查日期**: 2026-03-14  
**备份源**: `outputs/b2_baseline_backup_20260313/`  
**正式种子**: 42, 52, 62, 72, 82 (5-seed protocol)

---

## 执行顺序与验证标准审查

### (a) 统一评估标准：验证集F1阈值选择 + 固定测试阈值

**当前状态**:
- ✅ Layer1直接预测: 已使用验证集F1优化阈值
- ⚠️ **iTransformer Layer1**: F1=0.0000 (所有5个seed) → 阈值选择失败
  - 原因: 模型预测全为负类，无法计算有意义的F1
  - 影响: 无法建立固定测试阈值
- ✅ Layer2 readout: 已使用验证集F1优化阈值
- ⚠️ **SSSD/SurvTraj Layer2**: 部分seed阈值异常
  - SSSD seed72/82: optimal_threshold=inf (无有效阈值)
  - SurvTraj seed52: AUROC=0.5 (随机分类器)

**风险**: iTransformer的F1=0问题破坏了统一的阈值协议。需要确认是否为：
1. 验证集标签分布极度不平衡导致无正例
2. 模型输出范围问题
3. 阈值搜索算法bug

---

### (b) 审计现有5-seed输出

**Layer1直接预测** (5 models × 5 seeds):
```
✅ CausalForest: 5/5 seeds完整 (metrics + plots)
✅ iTransformer: 5/5 seeds完整 (metrics + plots) [但F1=0异常]
✅ TSDiff: 5/5 seeds完整 (metrics + plots)
✅ STaSy: 5/5 seeds完整 (metrics + plots)
```

**Layer1 TSTR生成式** (6 models × 5 seeds):
```
✅ SSSD: 5/5 seeds完整 (metrics + predictions)
✅ SurvTraj: 5/5 seeds完整 (metrics + predictions)
✅ TabSyn: 5/5 seeds完整 (metrics + predictions)
❌ TabDiff: 5/5 seeds失败 (FAILED标记)
  - 所有5个seed均标记: "all predictions are positive class"
  - 无有效metrics/predictions
✅ TSDiff: 5/5 seeds完整 (metrics)
✅ STaSy: 5/5 seeds完整 (metrics)
```

**Layer2轨迹预测** (4 models × 5 seeds):
```
✅ iTransformer: 5/5 seeds完整 (trajectory + readout metrics)
✅ TimeXer: 5/5 seeds完整 (trajectory + readout metrics)
⚠️ SSSD: 5/5 seeds readout metrics存在 (但seed72/82阈值=inf)
⚠️ SurvTraj: 5/5 seeds readout metrics存在 (但seed52 AUROC=0.5)
```

**汇总表** (5个CSV):
```
✅ baseline_main_table.csv (Layer1直接预测)
✅ baseline_tstr_table.csv (Layer1 TSTR)
✅ baseline_layer2_table.csv (Layer2轨迹)
✅ baseline_efficiency_table.csv (效率数据)
✅ baseline_layer1_tstr.csv (Layer1 TSTR详细)
```

---

### (c) 重跑触发条件分析

#### 1. **协议不一致** (Protocol Inconsistency)

| 问题 | 模型 | 种子 | 严重性 | 重跑? |
|------|------|------|--------|-------|
| F1=0 (无阈值) | iTransformer | 42,52,62,72,82 | 🔴 高 | **YES** |
| 阈值=inf | SSSD | 72,82 | 🟡 中 | **YES** |
| AUROC=0.5 | SurvTraj | 52 | 🟡 中 | **YES** |
| 全正预测失败 | TabDiff | 42,52,62,72,82 | 🔴 高 | **YES** |

#### 2. **缺失种子** (Missing Seeds)

- ✅ 所有5个正式种子(42,52,62,72,82)均存在
- ⚠️ 备份中包含非正式种子(1024,2024,2025,9999) → 需清理

#### 3. **自相矛盾指标** (Self-Inconsistent Metrics)

| 模型 | 问题 | 影响 |
|------|------|------|
| STaSy Layer1 | AUROC=0.3958 (< 0.5) | 需验证是否为已知bug |
| STaSy TSTR | AUROC范围0.31-0.59 (极度不稳定) | 需验证训练稳定性 |
| iTransformer | F1=0但AUROC=0.5487 | 阈值选择算法bug |

#### 4. **已知异常** (Known Anomalies)

| 异常 | 状态 | 重跑? |
|------|------|-------|
| SSSD collapse (AUROC<0.5) | seed72/82存在 | **YES** |
| STaSy AUROC<0.5 | Layer1全部 | 需确认是否为bug |
| iTransformer F1=0 | Layer1全部 | **YES** |
| TabDiff全正预测 | TSTR全部 | **YES** |

---

### (d) 重跑范围 (最小化)

**必须重跑**:
1. **iTransformer Layer1** (5 seeds)
   - 原因: F1=0 → 无法建立固定测试阈值
   - 操作: 重新训练 + 验证集F1优化

2. **TabDiff TSTR** (5 seeds)
   - 原因: 全部失败 (全正预测)
   - 操作: 重新训练 + 验证预测分布

3. **SSSD Layer2** (seed 72, 82)
   - 原因: 阈值=inf (无有效F1)
   - 操作: 重新评估 + 重新计算readout指标

4. **SurvTraj Layer2** (seed 52)
   - 原因: AUROC=0.5 (随机分类)
   - 操作: 重新评估 + 重新计算readout指标

**可选重跑** (需确认):
- STaSy Layer1 (AUROC<0.5是否为已知bug?)
- STaSy TSTR (极度不稳定的AUROC)

**无需重跑**:
- CausalForest, TSDiff (Layer1)
- SSSD, SurvTraj, TabSyn, TSDiff, STaSy (TSTR, 除TabDiff外)
- iTransformer, TimeXer (Layer2)
- SSSD Layer2 seed 42,52,62
- SurvTraj Layer2 seed 42,62,72,82

---

### (e) 重新生成5个正式CSV表

**操作流程**:
1. 从备份中提取5-seed数据 (排除1024,2024,2025,9999)
2. 对重跑的模型/种子，使用新结果替换
3. 重新计算均值±标准差
4. 验证所有指标的一致性

**表格清单**:
- `baseline_main_table.csv` (Layer1直接预测)
- `baseline_tstr_table.csv` (Layer1 TSTR)
- `baseline_layer2_table.csv` (Layer2轨迹)
- `baseline_efficiency_table.csv` (效率数据)
- `baseline_layer1_tstr.csv` (Layer1 TSTR详细)

---

### (f) 更新BASELINE_COMPARISON_REPORT.md

**需更新内容**:
1. 移除非正式种子(1024,2024,2025,9999)的数据
2. 更新iTransformer Layer1的F1和阈值信息
3. 更新TabDiff TSTR的状态 (从失败→成功或移除)
4. 更新SSSD/SurvTraj Layer2的阈值信息
5. 添加"基线锁定日期"和"协议版本"标记

---

### (g) 一致性验证

**验证清单**:
- [ ] 所有5个CSV表中的模型列表一致
- [ ] 所有指标的均值±标准差格式一致
- [ ] 没有NaN或inf值
- [ ] 所有5个种子的数据都被包含
- [ ] 没有非正式种子的数据
- [ ] F1阈值都是从验证集优化得出
- [ ] 测试集使用固定阈值评估

---

## 关键风险与建议

### 🔴 高风险

1. **iTransformer F1=0问题**
   - 根本原因: 验证集可能无正例或模型输出全为负类
   - 建议: 检查验证集标签分布，确认模型输出范围
   - 重跑前必须解决

2. **TabDiff全失败**
   - 根本原因: 模型预测全为正类
   - 建议: 检查模型输出层、损失函数、数据加载
   - 需要完整重新训练

### 🟡 中风险

3. **SSSD/SurvTraj Layer2阈值异常**
   - 根本原因: 验证集F1无法找到有效阈值
   - 建议: 使用备选阈值策略 (如中位数、0.5)
   - 仅需重新评估，不需重新训练

4. **STaSy AUROC<0.5**
   - 根本原因: 未知 (可能是已知bug或数据问题)
   - 建议: 确认是否为已知问题，决定是否重跑
   - 暂不重跑，等待确认

### ✅ 低风险

5. **非正式种子混入**
   - 根本原因: 实验过程中添加的额外种子
   - 建议: 清理备份，仅保留5个正式种子
   - 不影响基线锁定

---

## 最小化重跑计划总结

### 必须执行的步骤

1. **确认STaSy AUROC<0.5是否为已知bug** (5分钟)
   - 如是: 跳过重跑
   - 如否: 添加到重跑列表

2. **重跑iTransformer Layer1** (30分钟)
   - 5个种子，修复F1=0问题

3. **重跑TabDiff TSTR** (60分钟)
   - 5个种子，修复全正预测问题

4. **重新评估SSSD/SurvTraj Layer2** (15分钟)
   - seed 72,82 (SSSD) + seed 52 (SurvTraj)
   - 仅需重新计算readout指标，不需重新训练

5. **重新生成5个CSV表** (10分钟)
   - 合并所有结果，计算统计量

6. **更新BASELINE_COMPARISON_REPORT.md** (10分钟)
   - 清理非正式种子，更新指标

7. **运行一致性验证脚本** (5分钟)
   - 验证所有指标的完整性和一致性

### 总耗时估计

- **最坏情况** (包括STaSy重跑): ~120分钟
- **最好情况** (STaSy跳过): ~130分钟
- **实际预期**: ~2小时

### 最终交付物

- ✅ 5个正式CSV表 (5-seed only)
- ✅ 更新的BASELINE_COMPARISON_REPORT.md
- ✅ 一致性验证报告
- ✅ 基线锁定声明 (包含日期、协议版本、验证状态)

---

## 审查结论

**该计划是否能实现最小化重跑的基线锁定?**

**答案: 是的，但需要解决4个关键问题**

1. ✅ 协议一致性: 通过重跑iTransformer/TabDiff/SSSD/SurvTraj可实现
2. ✅ 5-seed覆盖: 所有正式种子均存在
3. ✅ 指标自洽性: 通过重新评估可修复
4. ⚠️ 已知异常: STaSy AUROC<0.5需确认

**建议**: 
- 立即确认STaSy AUROC<0.5的根本原因
- 并行执行iTransformer和TabDiff的重跑
- 完成后立即生成最终CSV表和报告
- 标记基线为"已锁定 v1.0 (5-seed protocol)"

