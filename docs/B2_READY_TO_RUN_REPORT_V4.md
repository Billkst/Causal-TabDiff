# B2 最终闸门验证报告（V4 - Targeted Fix）

**生成时间**: 2026-03-12  
**验证脚本**: `final_gate_test_v4_targeted.py`  
**结果文件**: `outputs/b2_gate/final_gate_test_results_v4.json`  
**日志文件**: `logs/final_gate_test_v4.log`

---

## 一、V4 闸门验证总结

**通过率**: 4/9 (44.4%)

### ✅ PASS (4)
1. CausalForest
2. **TSDiff** ← V4 新通过
3. SurvTraj_strict
4. SSSD_strict

### ❌ FAIL (5)
1. STaSy - 新错误：'activation'
2. TabSyn_strict - 参数不匹配
3. TabDiff_strict - 参数不匹配
4. iTransformer - TSLib 导入仍失败
5. TimeXer - TSLib 导入仍失败

---

## 二、进展对比

| 版本 | 通过 | 通过率 | 新增通过 |
|------|------|--------|---------|
| V2 | 1/9 | 11% | - |
| V3 | 3/9 | 33% | +2 (SurvTraj, SSSD) |
| V4 | 4/9 | 44% | +1 (TSDiff) |

---

## 三、V4 修复效果

### ✅ 成功修复 (1)
**TSDiff**
- 问题：缺少 predict() 方法
- 修复：添加 predict() 方法
- 文件：src/baselines/tsdiff_landmark_wrapper.py
- 状态：✅ 通过

### ⚠️ 部分修复 (1)
**STaSy**
- 问题：ncsnpp_tabular 导入失败
- 修复：添加模块注册到 __init__.py
- 新错误：'activation' (配置问题)
- 文件：src/baselines/stasy_core/models/__init__.py
- 状态：❌ 仍失败，但错误已变化

### ❌ 未修复 (4)
**TabSyn_strict**
- 问题：Model_VAE 参数不匹配
- 原因：wrapper 使用错误参数名
- 需要：改用正确签名

**TabDiff_strict**
- 问题：UniModMLP 参数不匹配
- 原因：wrapper 缺少必要参数
- 需要：补齐参数

**iTransformer**
- 问题：TSLib 导入失败
- 原因：gate 脚本中 sys.path 设置与 wrapper 内部冲突
- 需要：统一路径处理

**TimeXer**
- 问题：TSLib 导入失败
- 原因：同 iTransformer
- 需要：统一路径处理

---

## 四、剩余 5 个失败的根因分类

### wrapper_interface_bug (2)
1. TabSyn_strict - VAE 接口参数不匹配
2. TabDiff_strict - 模型接口参数不匹配

### dependency_or_path_bug (2)
3. iTransformer - TSLib 路径冲突
4. TimeXer - TSLib 路径冲突

### wrapper_config_bug (1)
5. STaSy - 配置参数缺失 ('activation')

---

## 五、判定结论

### ❌ 尚不具备进入 baseline 正式对比实验的条件

**理由**:
1. 仅 4/9 模型通过 (44%)
2. 4 个 strict 模型中仅 2 个通过
3. 所有 TSLib 模型失败
4. STaSy 仍失败

**但进展显著**:
- V3 → V4: +1 通过 (TSDiff)
- 已修复 3 个模型 (SurvTraj, SSSD, TSDiff)
- 剩余 5 个失败都是可修复的接口/配置问题

---

## 六、实际修改文件

1. ✅ `src/baselines/tsdiff_landmark_wrapper.py`
   - 添加 predict() 方法

2. ✅ `src/baselines/stasy_core/models/__init__.py`
   - 添加模块注册 (from . import ncsnpp_tabular, utils, ema)

---

**报告结束 - 等待用户确认下一步行动**
