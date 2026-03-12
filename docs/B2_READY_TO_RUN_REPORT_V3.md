# B2 最终闸门验证报告（V3 - 修复后）

**生成时间**: 2026-03-12  
**验证脚本**: `final_gate_test_v3_after_fixes.py`  
**结果文件**: `outputs/b2_gate/final_gate_test_results_v3.json`  
**日志文件**: `logs/final_gate_test_v3.log`

---

## 一、V3 闸门验证总结

**通过率**: 3/9 (33.3%)

### ✅ PASS (3)
1. CausalForest
2. SurvTraj_strict
3. SSSD_strict

### ❌ FAIL (6)
1. TSDiff
2. STaSy
3. TabSyn_strict
4. TabDiff_strict
5. iTransformer
6. TimeXer

---

## 二、V2 → V3 修复效果

| 修复项 | 状态 | 效果 |
|--------|------|------|
| 维度修正 (45→15) | ✅ 完成 | SurvTraj_strict, SSSD_strict 通过 |
| sys.path 添加 src/baselines | ✅ 完成 | TabSyn/TabDiff 可导入，但内部接口不匹配 |
| sys.path 添加 external/TSLib | ❌ 未生效 | iTransformer/TimeXer 仍无法导入 |

---

## 三、剩余失败详细分析

### 3.1 TSDiff
**错误**: `'TSDiffLandmarkWrapper' object has no attribute 'predict'`

**根因**: **wrapper_interface_bug**

**说明**: Wrapper 缺少 `predict()` 方法，只有 `sample()` 方法

---

### 3.2 STaSy
**错误**: `'ncsnpp_tabular'`

**根因**: **dependency_or_path_bug**

**说明**: stasy_core 内部模块导入仍失败，需要检查 stasy_core/models/__init__.py

---

### 3.3 TabSyn_strict
**错误**: `Model_VAE.__init__() got an unexpected keyword argument 'num_continuous'`

**根因**: **wrapper_interface_bug**

**说明**: Wrapper 调用 TabSyn VAE 的接口参数不匹配

---

### 3.4 TabDiff_strict
**错误**: `UniModMLP.__init__() missing 4 required positional arguments`

**根因**: **wrapper_interface_bug**

**说明**: Wrapper 调用 TabDiff 模型的接口参数不匹配

---

### 3.5 iTransformer
**错误**: `No module named 'models.iTransformer'`

**根因**: **dependency_or_path_bug**

**说明**: 虽然添加了 external/TSLib 到 sys.path，但导入仍失败。需要检查 TSLib 内部结构

---

### 3.6 TimeXer
**错误**: `No module named 'models.iTransformer'`

**根因**: **dependency_or_path_bug**

**说明**: 同 iTransformer

---

## 四、根因分类汇总（V3 后）

### 已修复 (3)
- ✅ SurvTraj_strict: gate_adapter_bug (维度) → 已修复
- ✅ SSSD_strict: gate_adapter_bug (维度) → 已修复
- ✅ CausalForest: 无问题

### 待修复 (6)

**wrapper_interface_bug (3)**:
- TSDiff: 缺少 predict() 方法
- TabSyn_strict: VAE 接口参数不匹配
- TabDiff_strict: 模型接口参数不匹配

**dependency_or_path_bug (2)**:
- iTransformer: TSLib 导入路径问题
- TimeXer: TSLib 导入路径问题

**dependency_or_path_bug (1)**:
- STaSy: stasy_core 内部模块导入问题

---

## 五、判定结论

### ❌ 尚不具备进入 baseline 正式对比实验的条件

**理由**:
1. 仅 3/9 模型通过
2. 4 个 strict 模型中仅 2 个通过
3. 所有 TSLib 模型失败
4. TSDiff/STaSy 仍失败

**但进展显著**:
- V2: 1/9 通过 (11%)
- V3: 3/9 通过 (33%)
- 维度修复有效

---

## 六、下一步修复建议

### 高优先级（可快速修复）
1. **TSDiff**: 添加 predict() 方法到 wrapper
2. **TSLib 模型**: 修正 external/TSLib 导入路径

### 中优先级（需要接口适配）
3. **TabSyn_strict**: 修正 VAE 初始化参数
4. **TabDiff_strict**: 修正模型初始化参数
5. **STaSy**: 修正 stasy_core 内部导入

---

**报告结束 - 等待用户确认下一步行动**
