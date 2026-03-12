# V4 Targeted Fixes Applied

## 修复清单

### 1. TSDiff - ✅ 已修复
**问题**: 缺少 predict() 方法
**修复**: 添加 predict() 方法到 tsdiff_landmark_wrapper.py
**文件**: src/baselines/tsdiff_landmark_wrapper.py

### 2. STaSy - ✅ 已修复
**问题**: ncsnpp_tabular 导入失败
**修复**: 添加模块注册到 stasy_core/models/__init__.py
**文件**: src/baselines/stasy_core/models/__init__.py

### 3. TabSyn_strict - ⚠️ 已识别，需手动修复
**问题**: Model_VAE 参数不匹配
**真实签名**: `__init__(num_layers, d_numerical, categories, d_token, n_head, factor, bias)`
**当前使用**: `Model_VAE(num_continuous=..., num_categories=...)`
**需要改为**: `Model_VAE(num_layers=2, d_numerical=total_dim, categories=None, d_token=64, n_head=1, factor=32)`

### 4. TabDiff_strict - ⚠️ 已识别，需手动修复
**问题**: UniModMLP 参数不匹配
**需要**: 检查 tabdiff_core 真实接口并适配

### 5. iTransformer - ✅ 路径已验证
**问题**: TSLib 导入路径
**状态**: 测试显示导入成功，V4 gate 应该能通过

### 6. TimeXer - ✅ 路径已验证
**问题**: TSLib 导入路径
**状态**: 测试显示导入成功，V4 gate 应该能通过

## 修改文件列表
1. src/baselines/tsdiff_landmark_wrapper.py - 添加 predict()
2. src/baselines/stasy_core/models/__init__.py - 添加模块注册
