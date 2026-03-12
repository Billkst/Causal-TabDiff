# iTransformer & TimeXer TSLib 导入修复报告

## 问题分析
- **根本原因**：`src/baselines/tslib_wrappers.py` 使用 `sys.path.insert()` 后直接导入，但 iTransformer.py 内部的 `from layers.Transformer_EncDec import ...` 找不到 layers 模块
- **症状**：`ModuleNotFoundError: No module named 'layers'`

## 修复方案
**方案选择**：importlib 动态加载 + sys.path 管理

### 关键改动
1. **文件**：`src/baselines/tslib_wrappers.py`
2. **方法**：
   - 使用 `importlib.util.spec_from_file_location()` 动态加载模型文件
   - 在执行模块前将 TSLib 路径加入 sys.path
   - 注册模块到 sys.modules 确保内部导入正确解析

### 代码变更
```python
def _load_tslib_model(model_name):
    """Load TSLib model dynamically using importlib."""
    tslib_path = os.path.join(os.path.dirname(__file__), '../../external/TSLib')
    if tslib_path not in sys.path:
        sys.path.insert(0, tslib_path)
    model_file = os.path.join(tslib_path, 'models', f'{model_name}.py')
    spec = importlib.util.spec_from_file_location(f'tslib_{model_name}', model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f'tslib_{model_name}'] = module
    spec.loader.exec_module(module)
    return module.Model
```

### TimeXer 适配
- TimeXer 不支持 `classification()` 方法，仅支持 `forecast()`
- 修改 TimeXerWrapper.forward()：使用 forecast 输出作为分类结果
- 添加 `self.num_class` 属性用于分类输出维度

## Smoke 测试结果
✅ **全部通过**

### iTransformer
- Layer 1 (classification): `[2, 24, 5] -> [2, 2]` ✓
- Layer 2 (forecast): `[2, 24, 5] -> [2, 6, 5]` ✓

### TimeXer
- Layer 1 (classification): `[2, 24, 5] -> [2, 2]` ✓
- Layer 2 (forecast): `[2, 24, 5] -> [2, 30]` ✓

## 修改文件清单
1. `src/baselines/tslib_wrappers.py` - 导入机制 + TimeXer 适配

## 验证
- 修复在 wrapper 内部完成，无需修改 external/TSLib
- 两个模型共用同一套导入机制
- 避免与其他 models 命名空间冲突
