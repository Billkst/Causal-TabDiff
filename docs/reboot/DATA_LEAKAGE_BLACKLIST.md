# Data Leakage Blacklist

**Version**: 1.0  
**Date**: 2026-03-10  
**Status**: Active - MANDATORY ENFORCEMENT

---

## 1. Critical Rule

**ANY field that contains information about the outcome (cancer diagnosis) or is only known AFTER diagnosis MUST be excluded from input features.**

---

## 2. Definitive Blacklist

### 2.1 Outcome Variables (NEVER use as features)

**From canc table**:
- `cancyr` - Year of cancer diagnosis (THIS IS THE LABEL SOURCE)
- `candx_days` - Days from randomization to diagnosis
- `can_scr` - Cancer detected by screening
- `canc_rpt_link` - Cancer report linkage
- `clinical_stag` - Clinical stage (only known after diagnosis)
- `path_stag` - Pathological stage (only known after diagnosis)
- `histology` - Histology type (only known after diagnosis)
- `grade` - Tumor grade (only known after diagnosis)

**From prsn table (lung cancer section)**:
- `cancyr` - Duplicate of outcome
- Any field prefixed with `canc_*`
- Any field prefixed with `de_*` (death-related, occurs after diagnosis)

### 2.2 Diagnosis-Derived Features

**Lesion characteristics (only measured after diagnosis)**:
- `lesionsize` - Tumor size at diagnosis
- `loc*` fields - Tumor location (loclhil, loclung, locoth, etc.)
- Any field describing confirmed malignancy

### 2.3 Post-Diagnosis Events

**Death information**:
- `de_*` - All death-related fields
- `fup_days` - Follow-up days (contains censoring info)
- `vital_status` - Vital status (leaks survival info)

### 2.4 Future Information

**Temporal leakage**:
- Any feature from study_yr > landmark
- Any screening result from T > landmark
- Any nodule measurement from T > landmark

---

## 3. Allowed Features (Safe to Use)

### 3.1 Baseline Demographics (prsn)
✅ `age` - Age at randomization  
✅ `gender` - Sex  
✅ `race` - Race/ethnicity  
✅ `ethnic` - Hispanic ethnicity  
✅ `bmi` - Body mass index at baseline  

### 3.2 Smoking History (prsn)
✅ `cigsmok` - Smoking status  
✅ `smokeage` - Age started smoking  
✅ `smokeyr` - Years smoked  
✅ `cigsperday` - Cigarettes per day  
✅ `packyears` - Pack-years (derived)  
✅ `smokeday` - Current smoking frequency  
✅ `smokequit` - Quit smoking status  

### 3.3 Medical History (prsn)
✅ `copd` - COPD diagnosis  
✅ `emphysema` - Emphysema  
✅ `chronic_bronchitis` - Chronic bronchitis  
✅ `fhx_lung_cancer` - Family history of lung cancer  
✅ `prior_cancer` - Prior cancer (non-lung)  

### 3.4 Screening Results (screen, ctab, ctabc)
✅ Screening occurred at landmark or before  
✅ Nodule findings at landmark or before  
✅ Nodule changes between visits ≤ landmark  
✅ CT quality metrics  

**CRITICAL**: Only use data from study_yr ≤ landmark

---

## 4. Validation Checklist

Before using ANY feature, verify:

- [ ] Is this field available at the landmark time point?
- [ ] Does this field contain outcome information?
- [ ] Is this field only measurable after diagnosis?
- [ ] Does this field leak future information?

**If ANY answer is "yes" or "uncertain" → EXCLUDE**

---

## 5. Common Mistakes to Avoid

❌ **Using cancyr as a feature** (it's the label!)  
❌ **Including T2 data in T0 samples** (temporal leakage)  
❌ **Using clinical_stag or path_stag** (only known after diagnosis)  
❌ **Including lesionsize** (measured at diagnosis, not screening)  
❌ **Using de_* fields** (death info leaks survival)  

---

## 6. Enforcement

**Code-Level Checks**:
```python
BLACKLIST = [
    'cancyr', 'candx_days', 'can_scr', 'canc_rpt_link',
    'clinical_stag', 'path_stag', 'histology', 'grade',
    'lesionsize', 'vital_status', 'fup_days'
]

# Plus any field matching patterns:
BLACKLIST_PATTERNS = [
    r'^canc_.*',   # Cancer-related
    r'^de_.*',     # Death-related  
    r'^loc.*',     # Location (diagnosis-time)
]

def validate_features(feature_list):
    for feat in feature_list:
        if feat in BLACKLIST:
            raise ValueError(f"LEAKAGE: {feat} is blacklisted!")
        for pattern in BLACKLIST_PATTERNS:
            if re.match(pattern, feat):
                raise ValueError(f"LEAKAGE: {feat} matches blacklist pattern!")
```

**Data Module Integration**:
- Implement validation in `_load_data()`
- Raise exception if blacklisted field detected
- Log all features used for audit trail

---

## 7. B1-1 Field Audit Report

### 7.1 PRSN Table (39 fields)

| Field | Decision | Reason |
|-------|----------|--------|
| pid | ✅ KEEP | Identifier |
| age | ✅ KEEP | Baseline demographic |
| gender | ✅ KEEP | Baseline demographic |
| race | ✅ KEEP | Baseline demographic |
| cigsmok | ✅ KEEP | Baseline smoking status |
| cancyr | ❌ REMOVE | Outcome variable (label source) |
| candx_days | ❌ REMOVE | Diagnosis timing (leakage) |
| can_scr | ❌ REMOVE | Cancer detection method (post-diagnosis) |
| canc_rpt_link | ❌ REMOVE | Cancer report linkage (post-diagnosis) |
| canc_free_days | ❌ REMOVE | Survival info (leakage) |
| lesionsize | ❌ REMOVE | Tumor size at diagnosis (leakage) |
| de_type | ❌ REMOVE | Death type (post-diagnosis) |
| de_grade | ❌ REMOVE | Tumor grade (post-diagnosis) |
| de_stag | ❌ REMOVE | Disease stage (post-diagnosis) |
| de_stag_7thed | ❌ REMOVE | TNM stage (post-diagnosis) |
| loclhil | ❌ REMOVE | Tumor location (post-diagnosis) |
| locllow | ❌ REMOVE | Tumor location (post-diagnosis) |
| loclup | ❌ REMOVE | Tumor location (post-diagnosis) |
| locrhil | ❌ REMOVE | Tumor location (post-diagnosis) |
| locrlow | ❌ REMOVE | Tumor location (post-diagnosis) |
| locrmid | ❌ REMOVE | Tumor location (post-diagnosis) |
| locrmsb | ❌ REMOVE | Tumor location (post-diagnosis) |
| locrup | ❌ REMOVE | Tumor location (post-diagnosis) |
| locunk | ❌ REMOVE | Tumor location (post-diagnosis) |
| locoth | ❌ REMOVE | Tumor location (post-diagnosis) |
| locmed | ❌ REMOVE | Tumor location (post-diagnosis) |
| loclmsb | ❌ REMOVE | Tumor location (post-diagnosis) |
| loccar | ❌ REMOVE | Tumor location (post-diagnosis) |
| loclin | ❌ REMOVE | Tumor location (post-diagnosis) |
| scr_days0 | ⚠️ UNCERTAIN | Screening timing - may leak study design |
| scr_days1 | ⚠️ UNCERTAIN | Screening timing - may leak study design |
| scr_days2 | ⚠️ UNCERTAIN | Screening timing - may leak study design |
| scr_iso0 | ⚠️ UNCERTAIN | Screening result - need temporal filtering |
| scr_iso1 | ⚠️ UNCERTAIN | Screening result - need temporal filtering |
| scr_iso2 | ⚠️ UNCERTAIN | Screening result - need temporal filtering |
| scr_res0 | ⚠️ UNCERTAIN | Screening result - need temporal filtering |
| scr_res1 | ⚠️ UNCERTAIN | Screening result - need temporal filtering |
| scr_res2 | ⚠️ UNCERTAIN | Screening result - need temporal filtering |
| dataset_version | ✅ KEEP | Metadata (non-predictive) |

**PRSN Summary**: 5 safe baseline features kept, 23 leakage fields removed, 9 uncertain fields need review.

### 7.2 SCREEN Table (20 fields)

| Field | Decision | Reason |
|-------|----------|--------|
| pid | ✅ KEEP | Identifier |
| study_yr | ✅ KEEP | Time dimension (with temporal filtering) |
| ctdxqual | ✅ KEEP | CT quality (safe if study_yr ≤ landmark) |
| techpara_kvp | ✅ KEEP | Technical parameter (safe) |
| techpara_ma | ✅ KEEP | Technical parameter (safe) |
| techpara_fov | ✅ KEEP | Technical parameter (safe) |
| techpara_effmas | ✅ KEEP | Technical parameter (safe) |
| ct_recon_filter* | ✅ KEEP | Reconstruction parameters (safe) |
| ctdxqual_* | ✅ KEEP | Quality indicators (safe) |
| dataset_version | ✅ KEEP | Metadata |

**SCREEN Summary**: All fields safe with temporal filtering (study_yr ≤ landmark).

### 7.3 CTAB Table (12 fields)

| Field | Decision | Reason |
|-------|----------|--------|
| pid | ✅ KEEP | Identifier |
| study_yr | ✅ KEEP | Time dimension |
| sct_ab_num | ✅ KEEP | Abnormality ID |
| sct_ab_desc | ✅ KEEP | Abnormality description (safe if study_yr ≤ landmark) |
| sct_long_dia | ✅ KEEP | Nodule diameter (safe with temporal filtering) |
| sct_perp_dia | ✅ KEEP | Nodule diameter (safe with temporal filtering) |
| sct_margins | ✅ KEEP | Nodule margins (safe with temporal filtering) |
| sct_pre_att | ✅ KEEP | Nodule attenuation (safe with temporal filtering) |
| sct_epi_loc | ✅ KEEP | Anatomical location (safe - screening finding, not diagnosis) |
| sct_slice_num | ✅ KEEP | CT slice number (safe) |
| sct_found_after_comp | ✅ KEEP | Discovery timing (safe) |
| dataset_version | ✅ KEEP | Metadata |

**CTAB Summary**: All fields safe with temporal filtering.

### 7.4 CTABC Table (10 fields)

| Field | Decision | Reason |
|-------|----------|--------|
| pid | ✅ KEEP | Identifier |
| study_yr | ✅ KEEP | Time dimension |
| sct_ab_num | ✅ KEEP | Abnormality ID |
| sct_ab_code | ✅ KEEP | Abnormality code (safe with temporal filtering) |
| sct_ab_gwth | ✅ KEEP | Growth indicator (safe with temporal filtering) |
| sct_ab_attn | ✅ KEEP | Attenuation change (safe with temporal filtering) |
| sct_ab_invg | ✅ KEEP | Investigation flag (safe with temporal filtering) |
| sct_ab_preexist | ✅ KEEP | Pre-existing flag (safe) |
| visible_days | ✅ KEEP | Visibility duration (safe) |
| dataset_version | ✅ KEEP | Metadata |

**CTABC Summary**: All fields safe with temporal filtering.

### 7.5 CANC Table (34 fields)

| Field | Decision | Reason |
|-------|----------|--------|
| pid | ✅ KEEP | Identifier (for label construction only) |
| study_yr | ✅ KEEP | Event timing (for label construction only) |
| candx_days | ❌ REMOVE | Diagnosis timing (leakage) |
| clinical_stag | ❌ REMOVE | Clinical stage (post-diagnosis) |
| path_stag | ❌ REMOVE | Pathological stage (post-diagnosis) |
| de_stag | ❌ REMOVE | Disease stage (post-diagnosis) |
| de_stag_7thed | ❌ REMOVE | TNM stage (post-diagnosis) |
| de_grade | ❌ REMOVE | Tumor grade (post-diagnosis) |
| de_type | ❌ REMOVE | Death type (post-diagnosis) |
| clinical_t/n/m_7thed | ❌ REMOVE | TNM components (post-diagnosis) |
| path_t/n/m_7thed | ❌ REMOVE | TNM components (post-diagnosis) |
| lesionsize | ❌ REMOVE | Tumor size (post-diagnosis) |
| lc_topog | ❌ REMOVE | Tumor topography (post-diagnosis) |
| lc_morph | ❌ REMOVE | Tumor morphology (post-diagnosis) |
| histology | ❌ REMOVE | Histology type (post-diagnosis) |
| All other fields | ❌ REMOVE | Post-diagnosis information |

**CANC Summary**: Only pid and study_yr used for label construction. All clinical fields removed.

---

## 8. Implementation Status

### 8.1 Current Implementation (build_landmark_tables.py)

**Baseline features (from prsn)**: age, gender, race, ethnic, bmi, cigsmok, smokeage, smokeyr, cigsperday, smokeday, smokequit, copd, emphysema, chronic_bronchitis, fhx_lung_cancer, prior_cancer

**Temporal features (with study_yr ≤ landmark filtering)**:
- Screen: ctdxqual, techpara_kvp, techpara_ma, techpara_fov
- Abnormality: abnormality_count, max_long_dia, max_perp_dia, has_spiculated
- Change: has_growth, has_attn_change, change_count

**Label construction**: cancyr from prsn/canc (bookkeeping only, not feature)

### 8.2 Unresolved Issues

⚠️ **UNCERTAIN FIELDS** (需要人工判断):
1. `scr_days0/1/2` - 筛查时间距随机化天数,可能泄露研究设计信息
2. `scr_iso0/1/2` - 筛查隔离读结果,需要确认是否与 cancyr 独立
3. `scr_res0/1/2` - 筛查最终结果,需要确认是否与 cancyr 独立

**建议**: 暂时不使用 prsn 表中的 scr_* 字段,改用 screen 表中的对应字段(已有时间维度过滤)。

---

## 9. Revision History

- **v1.0 (2026-03-10)**: Initial blacklist based on NLST data dictionaries
- **v1.1 (2026-03-10)**: B1-1 field audit - 5 表逐字段审计完成
