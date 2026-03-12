# B2 最终闸门验证报告（严格版 V2）

**生成时间**: 2026-03-12  
**验证脚本**: `final_gate_test_strict_v2.py`  
**结果文件**: `outputs/b2_gate/final_gate_test_results_v2.json`  
**日志文件**: `logs/final_gate_test_strict_v2.log`

---

## 一、闸门验证总结

**通过率**: 1/9 (11.1%)

### ✅ PASS (1)
- CausalForest

### ❌ FAIL (8)
- TSDiff
- STaSy
- TabSyn_strict
- TabDiff_strict
- SurvTraj_strict
- SSSD_strict
- iTransformer
- TimeXer

---

## 二、各模型详细状态

| 模型 | Layer1 | Layer2 | Final | Blocker | 错误信息 |
|------|--------|--------|-------|---------|---------|
| **CausalForest** | PASS | N/A | PASS | none | - |
| **TSDiff** | FAIL | FAIL | FAIL | code_blocker | tensor size mismatch (46 vs 136) |
| **STaSy** | FAIL | FAIL | FAIL | code_blocker | 'ncsnpp_tabular' module missing |
| **TabSyn_strict** | FAIL | FAIL | FAIL | code_blocker | No module 'tabsyn_core' |
| **TabDiff_strict** | FAIL | FAIL | FAIL | code_blocker | No module 'tabdiff_core' |
| **SurvTraj_strict** | FAIL | FAIL | FAIL | code_blocker | mat1/mat2 shape mismatch (45 vs 135) |
| **SSSD_strict** | FAIL | FAIL | FAIL | code_blocker | mat1/mat2 shape mismatch (46 vs 136) |
| **iTransformer** | FAIL | FAIL | FAIL | code_blocker | No module 'models.iTransformer' |
| **TimeXer** | FAIL | FAIL | FAIL | code_blocker | No module 'models.iTransformer' |

---

## 三、失败原因分析

### 3.1 TSDiff
**错误**: `The size of tensor a (46) must match the size of tensor b (136) at non-singleton dimension 1`

**原因**: 维度不匹配，wrapper 实现与数据维度不兼容

**Blocker 类型**: code_blocker

---

### 3.2 STaSy
**错误**: `'ncsnpp_tabular'`

**原因**: 缺少 stasy_core 内部模块导入

**Blocker 类型**: code_blocker

---

### 3.3 TabSyn_strict
**错误**: `No module named 'tabsyn_core'`

**原因**: tabsyn_core 模块路径未正确添加到 sys.path

**Blocker 类型**: code_blocker

---

### 3.4 TabDiff_strict
**错误**: `No module named 'tabdiff_core'`

**原因**: tabdiff_core 模块路径未正确添加到 sys.path

**Blocker 类型**: code_blocker

---

### 3.5 SurvTraj_strict
**错误**: `mat1 and mat2 shapes cannot be multiplied (32x45 and 135x128)`

**原因**: 输入维度与模型期望维度不匹配

**Blocker 类型**: code_blocker

---

### 3.6 SSSD_strict
**错误**: `mat1 and mat2 shapes cannot be multiplied (32x46 and 136x256)`

**原因**: 输入维度与模型期望维度不匹配

**Blocker 类型**: code_blocker

---

### 3.7 iTransformer
**错误**: `No module named 'models.iTransformer'`

**原因**: TSLib 外部库路径未正确配置

**Blocker 类型**: code_blocker

---

### 3.8 TimeXer
**错误**: `No module named 'models.iTransformer'`

**原因**: TSLib 外部库路径未正确配置

**Blocker 类型**: code_blocker

---

## 四、判定结论

### ❌ 不具备进入 baseline 正式对比实验的条件

**理由**:
1. 仅 1/9 模型通过严格验证
2. 所有 strict 模型均失败（TabSyn/TabDiff/SurvTraj/SSSD）
3. 所有 diffusion 模型均失败（TSDiff/STaSy）
4. 所有 TSLib 模型均失败（iTransformer/TimeXer）
5. 所有失败均为 code_blocker，需要修复代码

---

## 五、V1 vs V2 对比

### V1 (宽松版) 结果
- 通过: 9/9 (100%)
- 判定标准: 能创建对象即 PASS

### V2 (严格版) 结果
- 通过: 1/9 (11.1%)
- 判定标准: 真实训练+预测+落盘

### 差异说明
V1 的"通过"是虚假的，因为：
- 只验证了 import 和实例化
- 未验证真实训练能力
- 未验证预测生成能力
- 未验证评估链路

V2 暴露了真实问题：
- 8 个模型存在代码级阻塞
- 需要修复后才能进入正式实验

---

**报告结束 - 等待用户确认下一步行动**
