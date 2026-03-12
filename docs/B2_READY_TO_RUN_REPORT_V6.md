# B2 最终闸门验证报告（V6 - Deep Repair）

**生成时间**: 2026-03-12  
**验证脚本**: `final_gate_test_v6_deeprepair.py`  
**结果文件**: `outputs/b2_gate/final_gate_test_results_v6.json`  
**日志文件**: `logs/final_gate_test_v6.log`

---

## 一、V6 闸门验证总结

**通过率**: 4/9 (44.4%)

### ✅ PASS (4)
1. CausalForest
2. TSDiff
3. SurvTraj_strict
4. SSSD_strict

### ❌ FAIL (5)
1. STaSy - 新错误：'num_scales'
2. TabSyn_strict - 新错误：'NoneType' object is not iterable
3. TabDiff_strict - 新错误：object of type 'NoneType' has no len()
4. iTransformer - TSLib 导入失败
5. TimeXer - TSLib 导入失败

---

## 二、V6 修复进展

### ✅ 已修复的接口问题
1. **TabSyn_strict**: VAE 参数名已修正
   - 从 `num_continuous` → `d_numerical`
   - 从 `num_categories` → `categories`
   - 新错误说明已通过参数检查，进入内部逻辑

2. **TabDiff_strict**: UniModMLP 参数已修正
   - 从错误参数名 → 正确参数 `d_numerical, categories, num_layers, d_token`
   - 新错误说明已通过参数检查，进入内部逻辑

3. **STaSy**: 添加 activation 配置
   - 新错误：'num_scales' 说明需要继续补齐配置

---

## 三、实际修改文件

1. ✅ `src/baselines/tabsyn_landmark_strict.py`
   - 修正 Model_VAE 参数名

2. ✅ `src/baselines/tabdiff_landmark_strict.py`
   - 修正 UniModMLP 参数名

3. ✅ `src/baselines/stasy_landmark_wrapper.py`
   - 添加 activation 配置

---

## 四、剩余问题分析

### STaSy - 配置链未完成
- 当前错误：'num_scales'
- 根因：配置字段仍不完整
- 类型：wrapper_config_bug（可继续修复）

### TabSyn_strict - categories=None 导致问题
- 当前错误：'NoneType' object is not iterable
- 根因：内部代码期望 categories 是列表而非 None
- 类型：wrapper_interface_bug（可继续修复）

### TabDiff_strict - categories=None 导致问题
- 当前错误：object of type 'NoneType' has no len()
- 根因：内部代码期望 categories 有长度
- 类型：wrapper_interface_bug（可继续修复）

### iTransformer/TimeXer - 路径问题未解决
- 当前错误：TSLib 导入失败
- 根因：路径冲突未修复
- 类型：dependency_or_path_bug（可继续修复）

---

## 五、判定结论

### ❌ 不具备进入 baseline 正式对比实验的条件

**理由**: 仅 4/9 通过 (44%)

**但有实质性进展**:
- 3 个模型的接口参数已修正
- 错误已从"参数不匹配"推进到"内部逻辑问题"
- 无 true_code_blocker 证据

---

**报告结束**
