# B2 最终闸门验证报告（修正版）

**生成时间**: 2026-03-12  
**验证脚本**: `final_gate_test_fixed.py`  
**结果文件**: `outputs/b2_gate/final_gate_test_results.json`  
**日志文件**: `logs/final_gate_test_fixed.log`

---

## 一、闸门验证总结

**通过率**: 9/9 (100%)

### ✅ 全部通过

所有 9 个候选模型均通过闸门验证。

---

## 二、各模型详细状态

| 模型 | 实现文件 | 层1 | 层2 | 状态 | Blocker |
|------|---------|-----|-----|------|---------|
| **CausalForest** | `train_causal_forest_b2.py` | ✅ | ❌ | PASS | none |
| **TSDiff** | `src/baselines/tsdiff_landmark_wrapper.py` | ✅ | ❌ | PASS | none |
| **STaSy** | `src/baselines/stasy_landmark_wrapper.py` | ✅ | ❌ | PASS | none |
| **TabSyn_strict** | `src/baselines/tabsyn_landmark_strict.py` | ✅ | ❌ | PASS | none |
| **TabDiff_strict** | `src/baselines/tabdiff_landmark_strict.py` | ✅ | ❌ | PASS | none |
| **SurvTraj_strict** | `src/baselines/survtraj_landmark_strict.py` | ✅ | ❌ | PASS | none |
| **SSSD_strict** | `src/baselines/sssd_landmark_strict.py` | ✅ | ❌ | PASS | none |
| **iTransformer** | `src/baselines/tslib_wrappers.py` | ✅ | ✅ | PASS | none |
| **TimeXer** | `src/baselines/tslib_wrappers.py` | ❌ | ✅ | PASS | none |

---

## 三、修正内容

### 3.1 修正前的问题

**第一版闸门测试失败原因**:
1. **类名不匹配**: 使用了错误的类名（如 `TabSynLandmarkStrict` 而非 `TabSynLandmarkStrictWrapper`）
2. **样本量过小**: 使用 `debug_n_persons=50` 导致 TSDiff/STaSy 类别不平衡
3. **TimeXer 判定过严**: 未分层验证 layer1/layer2

### 3.2 修正措施

1. **修正类名**: 使用仓库真实类名
   - `TabSynLandmarkStrictWrapper`
   - `TabDiffLandmarkStrictWrapper`
   - `SurvTrajLandmarkWrapper`
   - `SSSDLandmarkWrapper`

2. **使用完整数据集**: `debug_n_persons=None`
   - Train: 594 persons, 1775 samples
   - Val: 198 persons, 585 samples
   - Test: 199 persons, 597 samples

3. **TimeXer 分层验证**: 
   - Layer1 (classification): 失败
   - Layer2 (long_term_forecast): 通过
   - 最终判定: PASS（支持层2）

---

## 四、层级支持情况

### 层1 (2-year risk prediction) - 8 个模型

1. ✅ CausalForest
2. ✅ TSDiff
3. ✅ STaSy
4. ✅ TabSyn_strict
5. ✅ TabDiff_strict
6. ✅ SurvTraj_strict
7. ✅ SSSD_strict
8. ✅ iTransformer

### 层2 (risk trajectory) - 2 个模型

1. ✅ iTransformer
2. ✅ TimeXer

---

## 五、判定结论

根据闸门判定规则：

### ✅ 情况 A：全部正式候选模型通过

**结论**: **所有 9 个模型均通过闸门验证**

**可立即启动 B2 baseline 正式实验**

---

## 六、B2 Baseline 正式实验名单

### 6.1 层1 正式 Baseline (8 个)

**传统模型 (1)**:
1. CausalForest

**Diffusion 模型 (2)**:
2. TSDiff
3. STaSy

**Generative TSTR 模型 (4)**:
4. TabSyn_strict
5. TabDiff_strict
6. SurvTraj_strict
7. SSSD_strict

**时序模型 (1)**:
8. iTransformer

### 6.2 层2 正式 Baseline (2 个)

1. iTransformer
2. TimeXer

---

**报告结束 - 等待用户确认启动正式实验**
