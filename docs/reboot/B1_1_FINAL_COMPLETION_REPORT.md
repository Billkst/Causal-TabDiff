# B1-1 Final Completion Report

**Version**: 1.1 (收尾修补完成)  
**Date**: 2026-03-11  
**Status**: B1-1 完成,可进入 B1-2

---

## 1. unified_person_landmark_table 最终包含的字段

### 1.1 字段清单 (43 个字段)

**标识与标签** (4):
- pid, landmark, y_2year, cancyr

**Baseline 特征** (4):
- baseline_age, baseline_gender, baseline_race, baseline_cigsmok

**Screen 特征** (12 = 4 字段 × 3 时间点):
- screen_t{0,1,2}_ctdxqual
- screen_t{0,1,2}_kvp
- screen_t{0,1,2}_ma
- screen_t{0,1,2}_fov

**Abnormality 特征** (12 = 4 字段 × 3 时间点):
- abn_t{0,1,2}_count
- abn_t{0,1,2}_max_long_dia
- abn_t{0,1,2}_max_perp_dia
- abn_t{0,1,2}_has_spiculated

**Change 特征** (9 = 3 字段 × 3 时间点):
- change_t{0,1,2}_has_growth
- change_t{0,1,2}_has_attn_change
- change_t{0,1,2}_change_count

**Trajectory 目标** (2):
- trajectory_target (array[7])
- trajectory_valid_mask (array[7])

### 1.2 特征维度统计

| Landmark | Baseline | Temporal | Total | 说明 |
|---|---|---|---|---|
| T0 | 4 | 11 | 15 | 只有 T0 时间点特征 |
| T1 | 4 | 22 | 26 | T0 + T1 时间点特征 |
| T2 | 4 | 33 | 37 | T0 + T1 + T2 时间点特征 |

**注**: 不同 landmark 的特征维度不同,体现真实短历史。

---

## 2. has_spiculated 最终采用的编码规则

### 2.1 数据字典定义

**来源**: dictionary_idc_ctab_idc-20210527.md (第 110 行)

**sct_margins 完整编码表**:
```
1 = "Spiculated (Stellate)" - 毛刺状/星芒状
2 = "Smooth" - 光滑
3 = "Poorly defined" - 边界不清
9 = "Unable to determine" - 无法判断
.N = "Not applicable" - 不适用
```

### 2.2 聚合规则

```python
has_spiculated = ('sct_margins', lambda x: (x == 1).any() if len(x) > 0 else False)
```

**语义**: 如果该 person-year 的任一病灶具有毛刺状边缘 (sct_margins=1),则 has_spiculated=True。

### 2.3 修正前后对比

| 项目 | 修前 | 修后 |
|---|---|---|
| 判断条件 | sct_margins == 4 | sct_margins == 1 |
| 依据 | 错误推测 | 数据字典确认 |
| 结果 | 永远为 False (数据中无 4) | 正确识别毛刺状病灶 |

---

## 3. trajectory_target 与 trajectory_valid_mask 的定义

### 3.1 trajectory_target

**类型**: Future yearly event indicator sequence

**维度**: 7 (float32)

**编码**:
- 0: 该年未发生事件
- 1: 该年发生事件

**构建逻辑**:
```python
future_event_years = np.zeros(7, dtype=np.float32)
if pd.notna(cancyr) and cancyr > landmark:
    event_offset = int(cancyr - landmark - 1)
    if 0 <= event_offset < valid_length:
        future_event_years[event_offset] = 1.0
```

**位置含义**:
- trajectory_target[0]: landmark + 1 年是否发生事件
- trajectory_target[1]: landmark + 2 年是否发生事件
- ...
- trajectory_target[6]: landmark + 7 年是否发生事件

### 3.2 trajectory_valid_mask

**类型**: Valid position indicator

**维度**: 7 (float32)

**编码**:
- 1: 该位置在观察期内,有效
- 0: 该位置超出观察期,无效

**构建逻辑**:
```python
trajectory_valid_mask = np.ones(7, dtype=np.float32)
max_observable_year = 7
valid_length = min(7, max_observable_year - landmark)
if valid_length < 7:
    trajectory_valid_mask[valid_length:] = 0
```

**不同 landmark 的 mask**:
- T0: [1,1,1,1,1,1,1] (全部 7 年可观察)
- T1: [1,1,1,1,1,1,0] (只能观察到 T7,T8 超出)
- T2: [1,1,1,1,1,0,0] (只能观察到 T7,T8-T9 超出)

### 3.3 配合使用

**训练时**:
```python
# 只计算有效位置的 loss
valid_pred = pred * mask
valid_target = target * mask
loss = criterion(valid_pred, valid_target)

# 或使用 mask 作为 weight
loss = criterion(pred, target, weight=mask)
```

**推断时**:
```python
# 只使用有效位置的预测
valid_pred = pred * mask
risk_2year = valid_pred[:2].max()  # 未来 2 年最大风险
```

### 3.4 与 y_2year 的关系

```python
y_2year = (trajectory_target[0] | trajectory_target[1])  # 未来 2 年内任一年发生
```

**验证**: y_2year 可由 trajectory_target 推导,保证一致性。

---

## 4. 修补后 B1-1 是否真的达到可进入 B1-2 的状态?

### 4.1 B1-1 完成度: 100%

**核心目标达成**:
- ✅ 5 表整合完成
- ✅ Leakage 防护完成
- ✅ 真实短历史实现
- ✅ 排除逻辑正确
- ✅ 2-year label 正确
- ✅ Trajectory target + mask 实现
- ✅ 所有中间表特征写入主表
- ✅ has_spiculated 编码修正
- ✅ 字段清单完整

**数据质量**:
- 总患者数: 991
- 总样本数: 2,957
- 字段数: 43
- 特征维度: T0=15, T1=26, T2=37

### 4.2 是的,已具备进入 B1-2 的条件

**B1-2 的唯一目标**:
1. 更新 `data_module_landmark.py` 使用新的 `unified_person_landmark_table.pkl`
2. 修复模型输入维度不匹配
3. 运行端到端 smoke test
4. 输出真实指标 (AUPRC, AUROC, F1)

**禁止事项**:
- 禁止改变数据表结构
- 禁止添加新特征
- 禁止运行完整训练
- 禁止运行 baseline

---

## 5. 修补前后对比

| 指标 | 修补前 | 修补后 | 说明 |
|---|---|---|---|
| 主表字段数 | ~20 | 43 | 补全所有中间表特征 |
| has_spiculated 编码 | sct_margins==4 | sct_margins==1 | 修正为正确编码 |
| trajectory_target | 有 | 有 | 保持不变 |
| trajectory_valid_mask | 无 | 有 | 新增,标记有效长度 |
| Screen 特征 | 1 字段 | 4 字段 | 补全 kvp, ma, fov |
| Abnormality 特征 | 2 字段 | 4 字段 | 补全 max_perp_dia, has_spiculated |
| Change 特征 | 1 字段 | 3 字段 | 补全 has_attn_change, change_count |

---

## 6. 输出文件

**代码**:
- `src/data/build_landmark_tables.py` (已修补)

**数据**:
- `data/landmark_tables/unified_person_landmark_table.pkl` (2,957 samples, 43 fields)
- `data/landmark_tables/statistics_report.json`

**文档**:
- `docs/reboot/B1_1_FIXES_REPORT.md` (已更新)
- `docs/reboot/B1_1_FINAL_COMPLETION_REPORT.md` (新建,本文档)

---

**B1-1 收尾修补完成,可以进入 B1-2。**
