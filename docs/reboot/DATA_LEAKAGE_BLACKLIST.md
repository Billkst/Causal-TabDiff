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

## 7. Revision History

- **v1.0 (2026-03-10)**: Initial blacklist based on NLST data dictionaries
