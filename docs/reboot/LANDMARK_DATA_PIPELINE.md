# Landmark Data Pipeline

**Version**: 1.0  
**Date**: 2026-03-10  
**Status**: B1-1 Complete - 5-Table Integration Implemented

---

## 1. Overview

本文档描述从 NLST 原始 5 表到统一 landmark 主建模表的完整数据流水线。

**核心原则**:
- 一行 = 一个 (person, landmark) 样本
- 只使用截至 landmark t 可见的信息
- 真实短历史: T0=[T0], T1=[T0,T1], T2=[T0,T1,T2]
- 严格排除 cancyr ≤ t 的样本
- 零伪时间复制

---

## 2. Input Tables

### 2.1 Raw Tables

| Table | Rows | Grain | Key Fields |
|-------|------|-------|------------|
| prsn | 53,453 | person | pid |
| screen | 75,139 | person × year | pid, study_yr |
| ctab | 177,488 | lesion | pid, study_yr, sct_ab_num |
| ctabc | 31,047 | lesion change | pid, study_yr, sct_ab_num |
| canc | 2,151 | cancer event | pid, study_yr |

### 2.2 Table Relationships

```
prsn (pid)
  ├─ screen (pid, study_yr) → 每人每年筛查记录
  ├─ ctab (pid, study_yr, sct_ab_num) → 病灶级异常
  ├─ ctabc (pid, study_yr, sct_ab_num) → 病灶级变化
  └─ canc (pid, study_yr) → 癌症事件
```

---

## 3. Intermediate Tables

### 3.1 person_baseline_table

**来源**: prsn  
**粒度**: person  
**主键**: pid

**字段**:
- Demographics: age, gender, race, ethnic, bmi
- Smoking: cigsmok, smokeage, smokeyr, cigsperday, smokeday, smokequit
- Medical history: copd, emphysema, chronic_bronchitis, fhx_lung_cancer, prior_cancer

**剔除字段**: cancyr, candx_days, can_scr, canc_rpt_link, lesionsize, de_*, loc*

**输出**: `person_baseline_table.pkl`

---

### 3.2 person_year_screening_summary

**来源**: screen  
**粒度**: person × year  
**主键**: (pid, study_yr)

**聚合规则**:
```python
screen.groupby(['pid', 'study_yr']).agg({
    'ctdxqual': 'first',
    'techpara_kvp': 'mean',
    'techpara_ma': 'mean',
    'techpara_fov': 'mean'
})
```

**输出**: `person_year_screening_summary.pkl`

---

### 3.3 person_year_abnormality_summary

**来源**: ctab  
**粒度**: person × year  
**主键**: (pid, study_yr)

**聚合规则**:
```python
ctab.groupby(['pid', 'study_yr']).agg(
    abnormality_count=('sct_ab_num', 'count'),
    max_long_dia=('sct_long_dia', 'max'),
    max_perp_dia=('sct_perp_dia', 'max'),
    has_spiculated=('sct_margins', lambda x: (x == 4).any())
)
```

**说明**:
- abnormality_count: 该年发现的异常数量
- max_long_dia/max_perp_dia: 最大病灶直径
- has_spiculated: 是否存在毛刺边缘 (sct_margins=4)

**输出**: `person_year_abnormality_summary.pkl`

---

### 3.4 person_year_change_summary

**来源**: ctabc  
**粒度**: person × year  
**主键**: (pid, study_yr)

**聚合规则**:
```python
ctabc.groupby(['pid', 'study_yr']).agg(
    has_growth=('sct_ab_gwth', lambda x: (x == 1).any()),
    has_attn_change=('sct_ab_attn', lambda x: (x > 0).any()),
    change_count=('sct_ab_num', 'count')
)
```

**说明**:
- has_growth: 是否存在间隔生长
- has_attn_change: 是否存在衰减变化
- change_count: 变化记录数

**输出**: `person_year_change_summary.pkl`

---

### 3.5 event_label_table

**来源**: prsn + canc  
**粒度**: person  
**主键**: pid

**构建逻辑**:
1. 从 prsn 提取 pid 和 cancyr
2. 从 canc 提取首次癌症事件的 study_yr
3. 合并: 优先使用 canc 表的事件时间
4. cancyr=0 表示无癌症

**输出**: `event_label_table.pkl`

---

## 4. Final Master Table

### 4.1 unified_person_landmark_table

**粒度**: person × landmark  
**主键**: (pid, landmark)

**构建逻辑**:

```python
for person in baseline:
    for landmark in [0, 1, 2]:
        # 排除规则
        if cancyr > 0 and cancyr <= landmark:
            continue
        
        # 2-year label
        y_2year = 1 if (cancyr > landmark and cancyr <= landmark + 2) else 0
        
        # 时间过滤
        screen_hist = screen[study_yr <= landmark]
        abn_hist = abnormality[study_yr <= landmark]
        change_hist = change[study_yr <= landmark]
        
        # 构建样本
        sample = {
            'pid': pid,
            'landmark': landmark,
            'y_2year': y_2year,
            'cancyr': cancyr,  # bookkeeping only
            'baseline_*': baseline features,
            'screen_t0_*': screen features at T0,
            'screen_t1_*': screen features at T1 (if landmark >= 1),
            'abn_t0_*': abnormality features at T0,
            'abn_t1_*': abnormality features at T1 (if landmark >= 1),
            'change_t0_*': change features at T0,
            'change_t1_*': change features at T1 (if landmark >= 1),
            ...
        }
```

**真实短历史**:
- T0 样本: 只有 T0 特征
- T1 样本: T0 + T1 特征
- T2 样本: T0 + T1 + T2 特征

**输出**: `unified_person_landmark_table.pkl`

---

## 5. Field Inventory

### 5.1 Baseline Features (16 fields from prsn)

age, gender, race, ethnic, bmi, cigsmok, smokeage, smokeyr, cigsperday, smokeday, smokequit, copd, emphysema, chronic_bronchitis, fhx_lung_cancer, prior_cancer

### 5.2 Temporal Features (per time point)

**From screen** (4 fields × 3 time points = 12):
- ctdxqual, techpara_kvp, techpara_ma, techpara_fov

**From abnormality** (4 fields × 3 time points = 12):
- abnormality_count, max_long_dia, max_perp_dia, has_spiculated

**From change** (3 fields × 3 time points = 9):
- has_growth, has_attn_change, change_count

### 5.3 Label Fields

- y_2year: binary (0/1)
- cancyr: bookkeeping (not feature)

### 5.4 Total Feature Dimensions

- Baseline: 16
- Temporal: 12 + 12 + 9 = 33 (但实际维度取决于 landmark)
- T0 样本: 16 + 11 = 27
- T1 样本: 16 + 22 = 38
- T2 样本: 16 + 33 = 49

---

## 6. Data Statistics (Debug Mode: nrows=1000)

### 6.1 Intermediate Tables

| Table | Rows | Columns |
|-------|------|---------|
| person_baseline_table | 1,000 | 5 |
| person_year_screening_summary | 1,000 | 6 |
| person_year_abnormality_summary | 388 | 6 |
| person_year_change_summary | 579 | 5 |
| event_label_table | 1,000 | 2 |

### 6.2 Final Master Table

- **总患者数**: 1,000
- **总样本数**: 2,976
- **平均每人样本数**: 2.98
- **排除样本数**: 24 (cancyr ≤ t)

**Landmark 分布**:
- T0: 1,000 样本, 阳性率 0.0170
- T1: 993 样本, 阳性率 0.0111
- T2: 983 样本, 阳性率 0.0051

---

## 7. Leakage Prevention

### 7.1 Blacklist Enforcement

代码中强制检查:
```python
BLACKLIST_EXACT = [
    'cancyr', 'candx_days', 'can_scr', 'canc_rpt_link',
    'clinical_stag', 'path_stag', 'histology', 'grade',
    'lesionsize', 'vital_status', 'fup_days'
]

BLACKLIST_PATTERNS = [r'^canc_.*', r'^de_.*', r'^loc.*']
```

### 7.2 Temporal Filtering

所有时间序列特征必须满足: `study_yr <= landmark`

### 7.3 Exclusion Logic

样本排除条件: `cancyr > 0 and cancyr <= landmark`

---

## 8. Known Issues & Limitations

### 8.1 当前实现的简化

1. **特征稀疏性**: 部分 person-year 组合在 ctab/ctabc 中无记录,导致特征缺失
2. **聚合规则**: 当前使用简单的 max/count,未考虑更复杂的时序模式
3. **prsn 表中的 scr_* 字段**: 暂未使用,因为不确定是否与 cancyr 独立

### 8.2 未决问题

⚠️ **需要人工判断的字段**:
- `scr_days0/1/2`: 筛查时间距随机化天数
- `scr_iso0/1/2`: 筛查隔离读结果
- `scr_res0/1/2`: 筛查最终结果

**建议**: 暂时不使用 prsn 表中的 scr_* 字段,改用 screen 表。

### 8.3 下一步改进 (B1-2+)

1. 处理特征缺失值
2. 优化聚合规则 (如加权平均、时序趋势)
3. 增加更多 baseline 特征 (如 BMI、家族史等)
4. 验证 scr_* 字段的安全性

---

## 9. Usage

### 9.1 运行流水线

```bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff
conda activate causal_tabdiff
python src/data/build_landmark_tables.py
```

### 9.2 加载主建模表

```python
import pandas as pd

unified = pd.read_pickle('data/landmark_tables/unified_person_landmark_table.pkl')
print(f"Samples: {len(unified)}")
print(f"Persons: {unified['pid'].nunique()}")
print(f"Positive rate: {unified['y_2year'].mean():.4f}")
```

---

## 10. References

- Data dictionaries: `docs/dataset/dictionaries/`
- Leakage blacklist: `docs/reboot/DATA_LEAKAGE_BLACKLIST.md`
- Implementation: `src/data/build_landmark_tables.py`
