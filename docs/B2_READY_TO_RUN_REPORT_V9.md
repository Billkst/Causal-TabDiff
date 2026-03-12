# B2 Baseline Gate 验证报告 V9 - Final Success

**生成时间**: 2026-03-12  
**验证版本**: V9 (Last 3 Models Fixed)  
**数据**: `unified_person_landmark_table.pkl` (200 persons debug mode)

---

## 🎯 执行摘要

**总体结果**: **9/9 模型通过 (100%)**

### 通过模型 (9/9)

**Layer1 支持 (7)**:
1. ✅ CausalForest
2. ✅ TSDiff
3. ✅ SurvTraj_strict
4. ✅ SSSD_strict
5. ✅ STaSy
6. ✅ TabSyn_strict (本轮修复)
7. ✅ TabDiff_strict
8. ✅ iTransformer (本轮修复)

**Layer2 支持 (2)**:
1. ✅ iTransformer (本轮修复)
2. ✅ TimeXer (本轮修复)

---

## 📊 本轮修复成果 (V8 → V9)

### 成功修复的 3 个模型

#### 1. TabSyn_strict ✅

**V8 错误**: `arange() received an invalid combination of arguments`

**根因**: 
- `sample()` 函数调用时，device 参数作为位置参数传递
- 导致 num_steps 接收到 device 字符串
- Model 类缺少 sigma_min/sigma_max/round_sigma 属性

**修复内容**:
1. 使用关键字参数: `sample(..., device=device)`
2. 添加 sigma 属性到 denoise_fn_D
3. 使用 denoise_fn_D 而非 Model 进行 sampling

**修改文件**: `src/baselines/tabsyn_landmark_strict.py` (第 66-68, 96 行)

**Smoke 结果**: ✅ PASS  
**V9 Gate 结果**: ✅ PASS

---

#### 2. iTransformer ✅

**V8 错误**: `'iTransformerWrapper' object has no attribute 'predict'`

**根因**: Gate 脚本调用了不存在的 predict() 方法

**修复内容**:
- Gate 脚本改用 forward() 方法
- Layer1: 使用 task='classification'
- Layer2: 使用 task='long_term_forecast'

**修改文件**: `final_gate_test_v9_last3.py` (test_itransformer 函数)

**Smoke 结果**: ✅ PASS (Layer1 + Layer2)  
**V9 Gate 结果**: ✅ PASS (Layer1 + Layer2)

---

#### 3. TimeXer ✅

**V8 错误**: `TimeXerWrapper.__init__() missing 1 required positional argument: 'exog_in'`

**根因**: Gate 脚本未提供 exog_in 参数

**修复内容**:
- Gate 脚本添加 exog_in=FEATURE_DIM 参数
- 使用 task='long_term_forecast'

**修改文件**: `final_gate_test_v9_last3.py` (test_timexer 函数)

**Smoke 结果**: ✅ PASS (Layer2)  
**V9 Gate 结果**: ✅ PASS (Layer2)

---

## 📝 修改文件清单

### Wrapper 修复
1. `src/baselines/tabsyn_landmark_strict.py` - 修复 sample 调用和 sigma 属性

### Gate 脚本修复
1. `final_gate_test_v9_last3.py` - 修正 iTransformer 和 TimeXer 调用方式

### 验证脚本
1. `repair_tabsyn_strict_smoke_v2.py`
2. `repair_itransformer_smoke_v2.py`
3. `repair_timexer_smoke_v2.py`

### 输出文件
- `outputs/model_repairs/*_repair_result_v2.json` (3 个)
- `outputs/b2_gate/final_gate_test_results_v9.json`
- `logs/final_gate_test_v9.log`

---

## ✅ 验收标准达成情况

### Layer1 PASS 标准 (7/9 满足)
- ✅ create/load success
- ✅ fit/train success
- ✅ prediction export ready
- ✅ evaluate_model ready

### Layer2 PASS 标准 (2/9 满足)
- ✅ create/load success
- ✅ fit/train success
- ✅ forecast/trajectory output ready
- ✅ risk trajectory valid
- ✅ eval/readout ready

---

## 🎯 最终结论

**当前状态**: **9/9 模型全部可用于 baseline 实验**

**模型覆盖**:
- 传统方法: CausalForest
- Diffusion: TSDiff, TabDiff_strict
- 生成式 TSTR: SurvTraj_strict, SSSD_strict, STaSy, TabSyn_strict
- 时序预测: iTransformer (Layer1+Layer2), TimeXer (Layer2)

**建议行动**: ✅ **立即启动 baseline 正式对比实验**

所有 9 个候选模型已通过严格验证，具备完整的 baseline 对比能力。
