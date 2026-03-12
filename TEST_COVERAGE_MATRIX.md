# 测试覆盖矩阵分析

## 概览
- **总模型数**: 14
- **已覆盖**: 7 (50%)
- **未覆盖**: 7 (50%)

## 测试文件覆盖情况

### 1. test_all_landmark_wrappers.py
覆盖模型 (3个):
- ✓ TSDiffLandmarkWrapper (v2)
- ✓ TabSynLandmarkWrapper (v2)
- ✓ TabDiffLandmarkWrapper (v2)

### 2. test_landmark_wrappers.py
覆盖模型 (2个):
- ✓ TSDiffLandmarkWrapper (v2)
- ✓ STaSyLandmarkWrapper (v2)

### 3. test_new_wrappers.py
覆盖模型 (2个):
- ✓ TSDiffLandmarkWrapper (旧版)
- ✓ STaSyLandmarkWrapper (旧版)

### 4. tests/test_causal_tabdiff.py
覆盖模型 (1个):
- ✓ CausalTabDiffWrapper (间接测试)

### 5. tests/test_causal_tabdiff_blockers.py
覆盖模型 (1个):
- ✓ CausalTabDiffWrapper (间接测试)

## 已覆盖的模型清单

| 模型名称 | 文件位置 | 测试文件 |
|---------|---------|---------|
| TSDiffLandmarkWrapper | src/baselines/tsdiff_landmark_v2.py | test_all_landmark_wrappers.py, test_landmark_wrappers.py |
| STaSyLandmarkWrapper | src/baselines/stasy_landmark_v2.py | test_landmark_wrappers.py |
| TabSynLandmarkWrapper | src/baselines/tabsyn_landmark_v2.py | test_all_landmark_wrappers.py |
| TabDiffLandmarkWrapper | src/baselines/tabdiff_landmark_v2.py | test_all_landmark_wrappers.py |
| TSDiffLandmarkWrapper (旧) | src/baselines/tsdiff_landmark_wrapper.py | test_new_wrappers.py |
| STaSyLandmarkWrapper (旧) | src/baselines/stasy_landmark_wrapper.py | test_new_wrappers.py |
| CausalTabDiffWrapper | src/baselines/wrappers.py | tests/test_causal_tabdiff.py, tests/test_causal_tabdiff_blockers.py |

## 未覆盖的模型清单 (需要新测试)

| 模型名称 | 文件位置 | 类型 |
|---------|---------|------|
| CausalForestWrapper | src/baselines/wrappers.py | 标准 wrapper (NLST) |
| STaSyWrapper | src/baselines/wrappers.py | 标准 wrapper (NLST) |
| TSDiffWrapper | src/baselines/wrappers.py | 标准 wrapper (NLST) |
| TabSynWrapper | src/baselines/wrappers.py | 标准 wrapper (NLST) |
| TabDiffWrapper | src/baselines/wrappers.py | 标准 wrapper (NLST) |
| iTransformerWrapper | src/baselines/tslib_wrappers.py | 时间序列 wrapper |
| TimeXerWrapper | src/baselines/tslib_wrappers.py | 时间序列 wrapper |

## 关键发现

1. **Landmark wrappers (v2)** 覆盖较好: 4/4 已覆盖
2. **标准 wrappers (NLST)** 完全未覆盖: 0/6 已覆盖
3. **时间序列 wrappers** 完全未覆盖: 0/2 已覆盖
4. **重复测试**: TSDiffLandmarkWrapper 被测试了 2 次 (v2 和旧版)

## 建议

编写统一测试脚本应覆盖:
1. 所有 7 个未覆盖的模型
2. 统一的测试接口和验证逻辑
3. 支持不同数据源 (landmark vs NLST)
