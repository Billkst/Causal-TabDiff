# B1 Completion Checklist

**Date**: 2026-03-10  
**Status**: IN PROGRESS - NOT COMPLETE

---

## 硬性要求检查

### A. 主入口切换 ❌ 未完成

**当前状态**:
- ❌ 旧入口仍然存在且未标记 legacy
  - `src/data/data_module.py` - 旧伪时间逻辑，仍在使用
  - `run_experiment.py` - 旧扩散训练入口
  - `run_baselines.py` - 旧 TSTR/Wasserstein 主逻辑
  
- ⚠️ 新入口已创建但未成为默认
  - `src/data/data_module_landmark.py` - 存在但不是主入口
  - `run_experiment_landmark.py` - 存在但不是主入口
  - `run_baselines_landmark.py` - 存在但不是主入口

**问题**: 仓库同时存在新旧两套管线，未明确哪个是正式入口

**需要做**:
- [ ] 明确标记旧文件为 `_legacy.py` 或移入 `archive/`
- [ ] 将新 landmark 文件重命名为主入口名称
- [ ] 更新所有文档指向新入口
- [ ] 在 README 中明确说明正式入口

---

### B. 5表整合 ❌ 未完成

**当前状态**:
- ❌ 只读取了 `prsn` 表
- ❌ `screen/ctab/ctabc/canc` 未整合
- ❌ 只使用 5 个 baseline 特征（age, gender, bmi, cigsmok, copd）
- ❌ 未构建 person-year 中间表

**数据表情况**:
- prsn: 53,453 行
- screen: 75,139 行（有 study_yr）
- ctab: 177,488 行（结节特征）
- ctabc: 31,047 行（结节变化）
- canc: 2,151 行（结局）

**需要做**:
- [ ] 构建 person-year screening summary（从 screen 表）
- [ ] 构建 person-year abnormality summary（从 ctab 表）
- [ ] 构建 person-year change summary（从 ctabc 表）
- [ ] 整合 canc 表获取 cancyr
- [ ] 生成最终 person-landmark 主建模表
- [ ] 输出数据统计报告

---

### C. Trajectory 定义 ⚠️ 部分完成

**当前状态**:
- ✅ 文档中有概念定义
- ❌ 代码实现不严谨
- ❌ 不同 landmark 下的处理逻辑不清晰

**当前代码问题**:
```python
# 当前实现（不严谨）
def _construct_risk_trajectory(self, cancyr, landmark, T_max=7):
    traj_len = T_max  # 固定长度 7
    hazard = np.zeros(traj_len, dtype=np.float32)
    
    if cancyr > 0 and cancyr > landmark:
        offset = int(cancyr - 1)  # 这个逻辑有问题
        if 0 <= offset < traj_len:
            hazard[offset] = 1.0
    
    return hazard
```

**问题**:
1. 对所有 landmark 都返回长度 7，但含义不同
2. `offset = cancyr - 1` 不考虑 landmark 位置
3. 未明确是 yearly hazard 还是 cumulative risk
4. 2-year risk 导出公式未实现

**需要做**:
- [ ] 明确定义：使用 yearly hazard
- [ ] 修正 offset 计算：`offset = cancyr - landmark - 1`
- [ ] 为不同 landmark 添加 mask
- [ ] 实现 2-year risk 导出函数
- [ ] 在文档中写清楚数学定义

---

### D. 端到端 Smoke Test ❌ 未完成

**当前状态**:
- ✅ Data-level shape check 通过
- ❌ 模型前向失败（维度不匹配）
- ❌ 未跑通完整训练循环
- ❌ 未输出任何真实指标

**Smoke test 结果**:
```
✅ Data loading: 178 samples from 60 persons
✅ Shapes: x=[4,3,5], trajectory=[4,7], y_2year=[4,1]
✅ Value ranges valid
✅ Splits loaded
❌ Model forward: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x1 and 10x64)
```

**问题**: 模型期望输入维度与实际数据不匹配

**需要做**:
- [ ] 修复模型输入维度
- [ ] 跑通前向传播
- [ ] 跑通一次完整训练迭代
- [ ] 跑通一次验证
- [ ] 输出至少一组指标（AUPRC, AUROC, F1）

---

### E. 新指标体系落地 ❌ 未完成

**当前状态**:
- ✅ 文档中定义了完整指标体系
- ❌ 代码中未实现
- ❌ 新 baseline 脚本只有 3 个基础指标

**当前实现**:
```python
# run_baselines_landmark.py 只有：
- AUROC
- AUPRC  
- F1
```

**缺失**:
- Precision, Recall, Specificity, NPV, Accuracy, Balanced Accuracy, MCC
- Brier score, Calibration intercept/slope
- Confusion matrix
- ROC curve, PR curve, Calibration plot, Decision curve
- ATE_Bias, Wasserstein, CMD

**需要做**:
- [ ] 实现完整指标计算函数
- [ ] 实现图表生成函数
- [ ] 集成到 baseline 评估脚本
- [ ] 输出格式化结果表

---

### F. 旧伪时间主链退出 ❌ 未完成

**当前状态**:
- ❌ 旧文件仍在主目录，未标记 legacy
- ❌ 未明确哪个是正式入口
- ❌ 可能导致后续实验混乱

**旧文件列表**:
- `src/data/data_module.py` - 伪时间复制逻辑
- `run_experiment.py` - 旧训练入口
- `run_baselines.py` - 旧评估入口

**需要做**:
- [ ] 重命名旧文件为 `*_legacy.py`
- [ ] 或移入 `archive/legacy_20260310/`
- [ ] 将新 landmark 文件重命名为主入口名
- [ ] 更新所有文档引用

---

## 文档完成度

| 文档 | 状态 | 备注 |
|------|------|------|
| TASK_CHARTER.md | ✅ | 概念清晰 |
| TRAJECTORY_FORMULATION.md | ⚠️ | 需补充严格数学定义 |
| EXPERIMENT_PROTOCOL.md | ✅ | 流程清晰 |
| METRIC_RATIONALE.md | ✅ | 指标完整 |
| CAUSAL_EVAL_PROTOCOL.md | ✅ | 方法明确 |
| BASELINE_COMPATIBILITY_REPORT.md | ✅ | 策略清晰 |
| DATA_LEAKAGE_BLACKLIST.md | ✅ | 规则明确 |
| B1_COMPLETION_CHECKLIST.md | ✅ | 本文档 |

---

## 代码完成度

| 组件 | 状态 | 备注 |
|------|------|------|
| Landmark 数据模块 | ⚠️ | 雏形存在，但只用 prsn 表 |
| 5表整合 | ❌ | 未完成 |
| Trajectory 构建 | ⚠️ | 逻辑不严谨 |
| 模型接口 | ❌ | 维度不匹配 |
| 训练脚本 | ⚠️ | 存在但未跑通 |
| 评估脚本 | ⚠️ | 只有基础指标 |
| Smoke test | ⚠️ | Data-level 通过，end-to-end 失败 |

---

## 总体完成度评估

**文档**: 90% ✅  
**数据管线**: 30% ❌  
**模型集成**: 10% ❌  
**评估体系**: 20% ❌  
**端到端验证**: 0% ❌  

**总体**: **约 30%** ❌

---

## B1 完成的真实标准

只有满足以下所有条件，才能声称 B1 完成：

- [ ] 主入口明确且唯一（旧入口已标记 legacy）
- [ ] 5表真实整合完成
- [ ] Trajectory 数学定义严格且代码正确
- [ ] 端到端 smoke test 跑通（包括训练、验证、指标输出）
- [ ] 新指标体系代码落地（至少主要指标）
- [ ] 数据统计报告输出（样本数、患者数、阳性率等）

**当前状态**: ❌ **未满足任何一项硬性标准**

---

## 下一步行动

### 优先级 1（必须立即完成）
1. 完成 5表整合
2. 修正 trajectory 构建逻辑
3. 修复模型维度
4. 跑通端到端 smoke test

### 优先级 2（B1 完成前必须）
5. 实现完整指标体系
6. 标记旧入口为 legacy
7. 输出数据统计报告

### 优先级 3（B2 阶段）
8. Baseline 适配
9. 5-seed 实验
10. 结果生成

---

## 修订历史

- **v1.0 (2026-03-10)**: 诚实的 B1 完成度评估 - 约 30% 完成
