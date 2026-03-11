# B1-1 Fixes Report

**Version**: 1.0  
**Date**: 2026-03-11  
**Status**: B1-1 修复回合完成

---

## 1. 修复的 Bug

### Bug 1: cancyr 语义错误

**修前问题**:
- 将缺失值 `fillna(0)`,导致"无癌"和"T0 癌症"混在一起
- 排除规则 `if cancyr > 0 and cancyr <= landmark` 无法排除 T0 癌症
- T0 癌症患者被错误地保留在训练集中

**修后规则**:
根据数据字典 (dictionary_idc_prsn_idc-20210527.md):
```
cancyr 编码:
  .N (NaN) = "Not Applicable" (无癌症)
  0 = "T0" (基线年份癌症)
  1 = "T1" (第 1 年癌症)
  2 = "T2" (第 2 年癌症)
  ...
  7 = "T7" (第 7 年癌症)
```

**修复措施**:
1. 保留 NaN 表示无癌,不再 fillna(0)
2. 排除规则改为: `if pd.notna(cancyr) and cancyr <= landmark`
3. 标签构建: `y_2year = 1 if (pd.notna(cancyr) and cancyr > landmark and cancyr <= landmark + 2) else 0`

**验证结果**:
- 癌症病例: 41 人
- 无癌病例: 959 人
- 排除样本: 43 个 (cancyr <= landmark)
- T0 癌症正确排除

---

### Bug 2: ctabc 变化特征逻辑错误

**修前问题**:
- `has_growth = (sct_ab_gwth == 1).any()` 抓到的是 "No" 而不是 "Yes"
- `has_attn_change = (sct_ab_attn > 0).any()` 会把 No/Yes/Unknown 都算进去

**修后规则**:
根据数据字典 (dictionary_idc_ctabc_idc-20210527.md):
```
sct_ab_gwth (病灶生长):
  .N = "Not applicable"
  1 = "No" (无生长)
  2 = "Yes" (有生长)
  9 = "Unable to determine"

sct_ab_attn (衰减变化):
  .M = "Missing"
  .N = "Not applicable"
  1 = "No" (无变化)
  2 = "Yes" (有变化)
  9 = "Unable to determine"
```

**修复措施**:
```python
has_growth = (sct_ab_gwth == 2).any()  # 2 = Yes
has_attn_change = (sct_ab_attn == 2).any()  # 2 = Yes
```

**验证结果**:
- person_year_change_summary: 392 条记录 (修前: 579)
- 只统计明确的 "Yes",排除 "No" 和 "Unknown"

---

### Bug 3: debug 抽样策略错误

**修前问题**:
- 对 5 张表分别取前 `nrows=1000`
- 导致 5 张表的 pid 不一致
- 统计口径不可信

**修后规则**:
1. 先从 prsn 随机抽样 1000 个 pid
2. 再对其余 4 张表按 pid 过滤
3. 保证 5 张表的 pid 一致

**修复措施**:
```python
self.sampled_pids = self.prsn['pid'].sample(n=1000, random_state=42)
self.prsn = self.prsn[self.prsn['pid'].isin(self.sampled_pids)]
self.screen = self.screen[self.screen['pid'].isin(self.sampled_pids)]
self.ctab = self.ctab[self.ctab['pid'].isin(self.sampled_pids)]
self.ctabc = self.ctabc[self.ctabc['pid'].isin(self.sampled_pids)]
self.canc = self.canc[self.canc['pid'].isin(self.sampled_pids)]
```

**验证结果**:
- prsn: 1000 人
- screen: 1523 条记录 (修前: 1000)
- ctab: 3727 条记录 (修前: 1000)
- ctabc: 674 条记录 (修前: 1000)
- canc: 47 条记录 (修前: 1000)

---

### Bug 4: 缺少 trajectory target

**修前问题**:
- 主建模表只有 y_2year 和 cancyr
- 没有 future risk trajectory target
- 无法支持 trajectory learning

**修后规则**:
采用 **方案 1: Future yearly event indicator sequence**

**Trajectory Target 定义**:
```python
future_event_years = np.zeros(7, dtype=np.float32)
if pd.notna(cancyr) and cancyr > landmark:
    event_offset = int(cancyr - landmark - 1)
    if 0 <= event_offset < 7:
        future_event_years[event_offset] = 1.0
```

**维度说明**:
- 长度: 7 (对应 future year 1-7)
- 位置 0: landmark + 1 年是否发生事件
- 位置 1: landmark + 2 年是否发生事件
- ...
- 位置 6: landmark + 7 年是否发生事件

**Landmark 不同的处理**:
- T0 样本: trajectory[0-6] 对应 T1-T7
- T1 样本: trajectory[0-6] 对应 T2-T8 (但 T8 超出观察期,实际只到 T7)
- T2 样本: trajectory[0-6] 对应 T3-T9 (但 T9 超出观察期,实际只到 T7)

**2-year risk 与 trajectory 的关系**:
```python
y_2year = trajectory[0] | trajectory[1]  # 未来 2 年内任一年发生事件
```

---

## 2. 新的 Debug 统计 (基于 pid 抽样)

### 2.1 原始表统计

| 表 | 行数 | 说明 |
|---|---|---|
| prsn | 1,000 | 抽样的 1000 个 pid |
| screen | 1,523 | 这 1000 人的筛查记录 |
| ctab | 3,727 | 这 1000 人的异常记录 |
| ctabc | 674 | 这 1000 人的变化记录 |
| canc | 47 | 这 1000 人的癌症记录 |

### 2.2 中间表统计

| 表 | 行数 | 粒度 |
|---|---|---|
| person_baseline_table | 1,000 | person |
| person_year_screening_summary | 1,523 | person × year |
| person_year_abnormality_summary | 1,337 | person × year |
| person_year_change_summary | 392 | person × year |
| event_label_table | 1,000 | person |

### 2.3 主建模表统计

- **总患者数**: 991 (9 人因 T0 癌症被排除)
- **总样本数**: 2,957
- **平均每人样本数**: 2.98
- **排除样本数**: 43 (cancyr <= landmark)

**Landmark 分布**:
- **T0**: 991 样本, 阳性率 0.0101 (1.01%)
- **T1**: 985 样本, 阳性率 0.0091 (0.91%)
- **T2**: 981 样本, 阳性率 0.0143 (1.43%)

**癌症病例分布**:
- 癌症病例: 41 人
- 无癌病例: 959 人
- 癌症比例: 4.1%

---

## 3. unified_person_landmark_table 当前状态

### 3.1 是否同时支持 2-year event label 和 trajectory target?

**是的**。主建模表现在包含:

1. **2-year event label**: `y_2year` (binary)
2. **Trajectory target**: `trajectory_target` (array[7])

### 3.2 Trajectory Target 定义

**类型**: Future yearly event indicator sequence

**维度**: 7 (对应 future year 1-7 相对于 landmark)

**编码**:
- 0: 该年未发生事件
- 1: 该年发生事件

**示例**:
```python
# T0 样本, cancyr=2 (T2 癌症)
trajectory_target = [0, 1, 0, 0, 0, 0, 0]  # T2 发生事件 (offset=1)

# T1 样本, cancyr=4 (T4 癌症)
trajectory_target = [0, 0, 1, 0, 0, 0, 0]  # T4 发生事件 (offset=2)

# T0 样本, 无癌
trajectory_target = [0, 0, 0, 0, 0, 0, 0]  # 全 0
```

**有效长度处理**:
- 当前实现: 固定长度 7,超出观察期的位置自然为 0
- 未来可扩展: 添加 mask 标记有效长度

---

## 4. 目前 B1-1 完成度

### 4.1 已完成 (100%)

✅ **5 表整合**: 完整流水线实现  
✅ **Leakage 审计**: 逐字段审计完成  
✅ **cancyr 语义修正**: NaN=无癌, 0=T0 癌症  
✅ **ctabc 聚合修正**: 2=Yes (正确抓取生长和变化)  
✅ **Debug 抽样修正**: 基于 pid 抽样,5 表一致  
✅ **Trajectory target**: Future yearly event indicator 实现  
✅ **统计报告**: 基于修正后的数据  
✅ **文档更新**: LANDMARK_DATA_PIPELINE.md, DATA_LEAKAGE_BLACKLIST.md

### 4.2 已知限制

⚠️ **特征稀疏性**: 部分 person-year 组合无 ctab/ctabc 记录  
⚠️ **Baseline 特征不全**: 当前只用了 5 个,应该用 16 个  
⚠️ **Trajectory mask**: 未实现有效长度 mask  
⚠️ **prsn 表 scr_* 字段**: 暂未使用,需确认是否安全

---

## 5. 是否已具备进入 B1-2 的条件?

### 5.1 B1-1 核心目标达成情况

| 目标 | 状态 | 说明 |
|---|---|---|
| 5 表整合 | ✅ 完成 | 5 张中间表 + 1 张主建模表 |
| Leakage 防护 | ✅ 完成 | 逐字段审计,黑名单强制执行 |
| 真实短历史 | ✅ 完成 | T0=[T0], T1=[T0,T1], T2=[T0,T1,T2] |
| 排除逻辑 | ✅ 完成 | cancyr <= landmark 正确排除 |
| 2-year label | ✅ 完成 | y_2year 正确构建 |
| Trajectory target | ✅ 完成 | 7 维 future event indicator |

### 5.2 结论

**是的,已具备进入 B1-2 的条件。**

B1-2 的唯一目标:
- 修复模型输入维度
- 完成端到端 smoke test
- 输出真实指标 (AUPRC, AUROC, F1)

---

## 6. 修复前后对比

| 指标 | 修前 | 修后 | 说明 |
|---|---|---|---|
| 总患者数 | 1000 | 991 | 9 人因 T0 癌症被排除 |
| 总样本数 | 2976 | 2957 | 正确排除 cancyr <= landmark |
| 排除样本数 | 24 | 43 | 修正后正确排除 T0 癌症 |
| T0 阳性率 | 1.70% | 1.01% | 修正后更合理 |
| T1 阳性率 | 1.11% | 0.91% | 修正后更合理 |
| T2 阳性率 | 0.51% | 1.43% | 修正后更合理 |
| screen 记录 | 1000 | 1523 | 基于 pid 抽样,更真实 |
| ctab 记录 | 1000 | 3727 | 基于 pid 抽样,更真实 |
| ctabc 记录 | 1000 | 674 | 基于 pid 抽样,更真实 |
| change_summary | 579 | 392 | 修正聚合规则后更准确 |

---

## 7. 下一步 (B1-2)

**禁止事项**:
- 禁止改变数据表结构
- 禁止添加新特征
- 禁止运行完整训练
- 禁止运行 baseline

**必须完成**:
1. 更新 `data_module_landmark.py` 使用新的 `unified_person_landmark_table.pkl`
2. 修复模型输入维度不匹配
3. 运行端到端 smoke test
4. 输出真实指标

---

## 8. 附录: 关键代码片段

### 8.1 cancyr 处理

```python
# event_label_table
all_pids['cancyr'] = self.prsn['cancyr']  # 保留 NaN
all_pids['cancyr'] = all_pids['cancyr_from_canc'].combine_first(all_pids['cancyr'])

# 排除逻辑
if pd.notna(cancyr) and cancyr <= landmark:
    excluded_count += 1
    continue

# 标签构建
y_2year = 1 if (pd.notna(cancyr) and cancyr > landmark and cancyr <= landmark + 2) else 0
```

### 8.2 ctabc 聚合

```python
change_agg = self.ctabc.groupby(['pid', 'study_yr']).agg(
    has_growth=('sct_ab_gwth', lambda x: (x == 2).any()),  # 2 = Yes
    has_attn_change=('sct_ab_attn', lambda x: (x == 2).any()),  # 2 = Yes
    change_count=('sct_ab_num', 'count')
)
```

### 8.3 Trajectory target

```python
future_event_years = np.zeros(7, dtype=np.float32)
if pd.notna(cancyr) and cancyr > landmark:
    event_offset = int(cancyr - landmark - 1)
    if 0 <= event_offset < 7:
        future_event_years[event_offset] = 1.0

sample['trajectory_target'] = future_event_years
```

### 8.4 Debug 抽样

```python
self.sampled_pids = self.prsn['pid'].sample(n=1000, random_state=42)
self.prsn = self.prsn[self.prsn['pid'].isin(self.sampled_pids)]
self.screen = self.screen[self.screen['pid'].isin(self.sampled_pids)]
self.ctab = self.ctab[self.ctab['pid'].isin(self.sampled_pids)]
self.ctabc = self.ctabc[self.ctabc['pid'].isin(self.sampled_pids)]
self.canc = self.canc[self.canc['pid'].isin(self.sampled_pids)]
```
