# Legacy 入口文件说明

## 目的
标记项目中已被替代的旧入口文件，避免混淆。

## Legacy 文件列表

### 1. `run_experiment.py`
**状态**: Legacy（不再使用）  
**原因**: 使用伪时间逻辑，不支持真实 landmark 短历史  
**替代**: `run_experiment_landmark.py`

### 2. `run_baselines.py`
**状态**: Legacy（不再使用）  
**原因**: 基于旧数据模块  
**替代**: `run_baselines_landmark.py`（待 B2 后使用）

### 3. `smoke_test_landmark.py`
**状态**: Legacy（部分过时）  
**原因**: 使用旧的 `get_landmark_dataloader` API  
**替代**: `smoke_test_b1_2.py`

### 4. `src/data/data_module.py`
**状态**: Legacy（不再使用）  
**原因**: 伪时间逻辑，从原始表重新构建  
**替代**: `src/data/data_module_landmark.py`

## 当前正式入口

### 训练
- **主入口**: `run_experiment_landmark.py`
- **数据**: `src/data/data_module_landmark.py`
- **模型**: `src/models/causal_tabdiff_trajectory.py`

### 测试
- **Smoke Test**: `smoke_test_b1_2.py`

### Baseline（待 B2 后启用）
- **入口**: `run_baselines_landmark.py`

## 迁移指南

### 数据加载
```python
# 旧方式
from src.data.data_module import get_dataloader
loader = get_dataloader('data', split='train', batch_size=32)

# 新方式
from src.data.data_module_landmark import get_dataloader
loader = get_dataloader(
    'data/landmark_tables/unified_person_landmark_table.pkl',
    split='train', batch_size=32, seed=42
)
```

### Batch 结构变化
旧 batch: `x`, `y`, `alpha`  
新 batch: `x`, `y_2year`, `trajectory_target`, `trajectory_valid_mask`, `landmark`, `history_length`, `pid`

## 何时可以删除

在 B2 完成、Baseline 适配完成、5-seed 实验完成且用户批准后可删除。

**当前**: 保留但不使用。
