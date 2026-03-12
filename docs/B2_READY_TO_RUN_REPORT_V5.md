# B2 最终闸门验证报告（V5 - Focused Sprint）

**生成时间**: 2026-03-12  
**验证脚本**: `final_gate_test_v5_focused.py`  
**结果文件**: `outputs/b2_gate/final_gate_test_results_v5.json`  
**日志文件**: `logs/final_gate_test_v5.log`

---

## 一、V5 闸门验证总结

**通过率**: 4/9 (44.4%)

### ✅ PASS (4)
1. CausalForest
2. TSDiff
3. SurvTraj_strict
4. SSSD_strict

### ❌ FAIL (5)
1. STaSy - 'activation' 配置错误
2. TabSyn_strict - Model_VAE 参数不匹配
3. TabDiff_strict - UniModMLP 参数不匹配
4. iTransformer - TSLib 导入失败
5. TimeXer - TSLib 导入失败

---

## 二、V5 vs V4 对比

**结果**: V5 = V4 (4/9 通过)

**原因**: V5 focused sprint 未能完成剩余 5 个模型的深度接口修复

---

## 三、剩余 5 个失败的根因

### wrapper_interface_bug (2)
1. **TabSyn_strict**: VAE 参数名不匹配
   - 需要: `Model_VAE(num_layers, d_numerical, categories, d_token, n_head, factor)`
   - 当前: `Model_VAE(num_continuous=..., num_categories=...)`

2. **TabDiff_strict**: 模型参数缺失
   - 需要: `UniModMLP(d_numerical, categories, num_layers, d_token, ...)`
   - 当前: 缺少必要参数

### dependency_or_path_bug (2)
3. **iTransformer**: TSLib 导入路径冲突
4. **TimeXer**: TSLib 导入路径冲突

### wrapper_config_bug (1)
5. **STaSy**: 配置字段缺失 ('activation')

---

## 四、判定结论

### ❌ 不具备进入 baseline 正式对比实验的条件

**理由**:
1. 仅 4/9 模型通过 (44%)
2. 剩余 5 个失败需要深度接口适配
3. 所有 TSLib 模型失败
4. 2 个 strict 模型失败

---

**报告结束**
