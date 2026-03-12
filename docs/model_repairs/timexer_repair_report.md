# TimeXer TSLib 导入修复报告

## 执行时间
2026-03-12

## 问题描述
TimeXer 与 iTransformer 共用 TSLib，但在导入时失败：
```
ModuleNotFoundError: No module named 'layers'
```

## 根本原因
`_load_tslib_model()` 函数使用 `importlib.util` 动态加载模型文件，但未将 TSLib 路径添加到 `sys.path`。当模型文件内部使用相对导入（如 `from layers.xxx import`）时，Python 无法找到 `layers` 模块。

## 修复方案
在 `src/baselines/tslib_wrappers.py` 中修改 `_load_tslib_model()` 函数：

**修改前：**
```python
def _load_tslib_model(model_name):
    tslib_path = os.path.join(os.path.dirname(__file__), '../../external/TSLib')
    model_file = os.path.join(tslib_path, 'models', f'{model_name}.py')
    spec = importlib.util.spec_from_file_location(f'tslib_{model_name}', model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model
```

**修改后：**
```python
def _load_tslib_model(model_name):
    tslib_path = os.path.join(os.path.dirname(__file__), '../../external/TSLib')
    if tslib_path not in sys.path:
        sys.path.insert(0, tslib_path)  # ← 关键修复
    model_file = os.path.join(tslib_path, 'models', f'{model_name}.py')
    spec = importlib.util.spec_from_file_location(f'tslib_{model_name}', model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model
```

## 修改文件
- `src/baselines/tslib_wrappers.py` (第 11-19 行)

## 验证结果

### Smoke 测试通过
✓ **Test 1: Model Creation** - TimeXer 模型成功创建
✓ **Test 2: Forecast** - 前向传播成功，输出形状 [2, 30]
✓ **Test 3: Risk Trajectory** - 风险轨迹有效，范围 [-0.5487, 0.1776]
✓ **Test 4: Model State** - 模型处于 eval 模式，144966 参数

### Layer2 功能验证
- ✓ create/load success
- ✓ forecast success
- ✓ risk trajectory valid (输出为连续风险值，非 covariate)
- ✓ eval/readout ready

## 关键特性
1. **复用 iTransformer 修复方案** - 使用相同的 sys.path 注入方法
2. **不修改 external/TSLib** - 所有修复在 wrapper 层完成
3. **Layer2 输出正确** - 生成 6 年风险轨迹（实际输出 30 步）
4. **无破坏性修改** - 向后兼容现有代码

## 输出文件
- `outputs/model_repairs/timexer_repair_result.json` - 测试结果 JSON
- `logs/model_repairs/timexer_repair.log` - 修复日志
- `docs/model_repairs/timexer_repair_report.md` - 本报告
