# B2 Blocker Triage 报告

**生成时间**: 2026-03-12  
**基于**: final_gate_test_strict_v2.py 结果

---

## Triage 总表

| 模型 | V2 错误 | 根因分类 | 根因解释 | 修复方案 |
|------|---------|---------|---------|---------|
| TSDiff | tensor size mismatch (46 vs 136) | **gate_adapter_bug** | Gate 脚本传入 feature_dim=45，但真实数据是 15 维/timestep。Wrapper 期望 flattened dim (3×15+1=46)，但某处计算错误导致 136 | 修正 gate 脚本使用正确维度 15 |
| STaSy | 'ncsnpp_tabular' | **dependency_or_path_bug** | stasy_core 内部模块导入路径问题，需要正确设置 sys.path | 修正 sys.path 和 import 语句 |
| TabSyn_strict | No module 'tabsyn_core' | **dependency_or_path_bug** | Gate 脚本未正确添加 src/baselines 到 sys.path | 修正 gate 脚本 sys.path |
| TabDiff_strict | No module 'tabdiff_core' | **dependency_or_path_bug** | Gate 脚本未正确添加 src/baselines 到 sys.path | 修正 gate 脚本 sys.path |
| SurvTraj_strict | mat1/mat2 shape mismatch (45 vs 135) | **gate_adapter_bug** | Gate 脚本传入 feature_dim=45，但真实数据是 15 维/timestep | 修正 gate 脚本使用正确维度 15 |
| SSSD_strict | mat1/mat2 shape mismatch (46 vs 136) | **gate_adapter_bug** | Gate 脚本传入 feature_dim=45，但真实数据是 15 维/timestep | 修正 gate 脚本使用正确维度 15 |
| iTransformer | No module 'models.iTransformer' | **dependency_or_path_bug** | Gate 脚本未正确添加 external/TSLib 到 sys.path | 修正 gate 脚本 sys.path |
| TimeXer | No module 'models.iTransformer' | **dependency_or_path_bug** | Gate 脚本未正确添加 external/TSLib 到 sys.path | 修正 gate 脚本 sys.path |

---

## 根因分类统计

- **gate_adapter_bug**: 4 个 (TSDiff, SurvTraj_strict, SSSD_strict, 部分 TSDiff)
- **dependency_or_path_bug**: 5 个 (STaSy, TabSyn_strict, TabDiff_strict, iTransformer, TimeXer)
- **wrapper_interface_bug**: 0 个
- **true_code_blocker**: 0 个（待修复后验证）

---

## 关键发现

### 真实数据维度
```
x.shape = (batch, 3, 15)  # 3 timesteps, 15 features per timestep
y_2year.shape = (batch, 1)
trajectory_target.shape = (batch, 7)
```

### V2 Gate 的致命错误
1. **硬编码错误维度**: 所有测试函数使用 `feature_dim=45`
2. **正确维度应为**: `feature_dim=15` (per timestep)
3. **Flattened 维度**: 3×15+1=46 (用于某些 wrapper)

---

## 修复清单

### 1. 修正 gate 脚本维度
- 所有 wrapper 初始化改为 `feature_dim=15`
- 移除硬编码的 45

### 2. 修正 sys.path
- 添加 `src/baselines` 到 sys.path (for strict wrappers)
- 添加 `external/TSLib` 到 sys.path (for TSLib models)
- 确保 stasy_core 内部导入正确

### 3. 修正 wrapper 调用
- 确保 wrapper 接收正确的 batch 结构
- 确保 layer1/layer2 验证路径正确

---

**下一步**: 创建修复后的 V3 gate 脚本
