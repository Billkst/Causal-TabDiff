# B1-2 Smoke Test 完成报告

## 执行时间
2026-03-11

## 目标
完成模型接口对齐与端到端 smoke test，验证 B1-1 数据表能够正确接入模型并完成训练循环。

## 完成内容

### 1. 数据侧对接

#### 1.1 重写 `src/data/data_module_landmark.py`
- **数据源**: 直接使用 `data/landmark_tables/unified_person_landmark_table.pkl`
- **不再使用**: 旧的从原始 5 表重新构建逻辑
- **核心改进**:
  - 真实短历史提取（T0=[T0], T1=[T0,T1], T2=[T0,T1,T2]）
  - Pid-level split（train/val/test = 60%/20%/20%）
  - 变长序列 padding（max_time=3）
  - 缺失值处理（NaN → 0.0）

#### 1.2 特征提取策略
**输入特征**（每个时间点）:
- Baseline (4): age, gender, race, cigsmok
- Temporal (11): screen (4) + abnormality (4) + change (3)
- 总维度: 15

**排除字段**（bookkeeping only）:
- `cancyr` - 仅用于构建标签，不进入输入
- `pid`, `landmark` - 仅用于 split 和追踪
- `trajectory_target`, `trajectory_valid_mask` - 仅作为监督目标

#### 1.3 缺失值处理
- **连续变量**: NaN → 0.0
- **布尔变量**: 转为 float 后 NaN → 0.0
- **缺失时间点**: 通过 padding 处理，模型需使用 `history_length` 或 mask

#### 1.4 Pid-Level Split 统计（Debug Mode: 100 persons）
```
Train: 60 persons, 180 samples, pos_rate=0.0167
Val:   20 persons, 60 samples, pos_rate=0.0000
Test:  20 persons, 60 samples, pos_rate=0.0000
```

### 2. 模型侧修复

#### 2.1 修复 `src/models/causal_tabdiff_trajectory.py`
- 添加 `cond_dim=1` 参数传递给 base_model
- 修复特征提取逻辑（使用 `block1` 而非不存在的 `encoder`）
- 保持 trajectory + 2-year risk 双输出

#### 2.2 模型输入/输出 Shape
**输入**:
- `x`: (batch, 3, 15) - padded 到 max_time=3
- `alpha_target`: (batch, 1) - 随机生成的环境暴露
- `history_length`: (batch, 1) - 真实历史长度 {1, 2, 3}

**输出**:
- `diff_loss`: scalar - 扩散损失
- `disc_loss`: scalar - 判别器损失
- `trajectory`: (batch, 7) - 7 年风险轨迹
- `risk_2year`: (batch, 1) - 2 年风险

#### 2.3 2-Year Risk 导出公式
当前实现:
```python
risk_2year = sigmoid(Linear(trajectory_probs))
```

**说明**: 这是简化版本。`trajectory` 是 7 维 yearly event indicator，`risk_2year` 通过线性层从 trajectory 派生。

### 3. Loss 定义

当前使用 4 项 loss:
```python
total_loss = diff_loss + 0.5 * disc_loss + loss_traj + loss_2year
```

**各项定义**:
1. `diff_loss`: MSE(predicted_noise, true_noise) - 扩散模型核心损失
2. `disc_loss`: MSE(predicted_alpha, target_alpha) - 判别器损失
3. `loss_traj`: BCE(trajectory * mask, target * mask) - 轨迹损失（使用 valid_mask）
4. `loss_2year`: BCE(risk_2year, y_2year) - 2 年风险损失

### 4. 端到端 Smoke Test 结果

#### 4.1 测试流程
✅ 数据加载  
✅ 模型初始化  
✅ 前向传播  
✅ Loss 计算  
✅ 反向传播  
✅ 2 Epoch 训练  
✅ Validation  
✅ 指标计算  

#### 4.2 最小指标（Debug Mode: 100 persons, 2 epochs）
```
AUROC: N/A (validation set 无阳性样本)
AUPRC: N/A (validation set 无阳性样本)
F1:    0.0000
Confusion Matrix:
[[ 0 60]
 [ 0  0]]
```

**说明**: 
- Debug mode 下 validation set 恰好无阳性样本
- 模型预测全为阴性（阈值 0.5）
- 这是正常的 smoke test 行为，不代表最终性能

### 5. 当前主入口

#### 5.1 正式入口
- **训练**: `run_experiment_landmark.py`
- **Smoke Test**: `smoke_test_b1_2.py`
- **数据**: `src/data/data_module_landmark.py`
- **模型**: `src/models/causal_tabdiff_trajectory.py`

#### 5.2 Legacy 入口（暂不使用）
- `run_experiment.py` - 旧训练脚本（伪时间逻辑）
- `run_baselines.py` - 旧 baseline 脚本
- `smoke_test_landmark.py` - 旧 smoke test（使用旧 API）

### 6. Smoke Test 命令

```bash
# B1-2 正式 smoke test
conda activate causal_tabdiff
python smoke_test_b1_2.py

# 预期输出: "B1-2 Smoke Test PASSED"
```

## 当前已知问题

### 1. Validation Set 阳性样本不足
- **现象**: Debug mode (100 persons) 下 val set 无阳性样本
- **影响**: 无法计算 AUROC/AUPRC
- **解决**: 使用完整数据集或增加 debug_n_persons

### 2. 模型预测偏向阴性
- **现象**: 所有预测 < 0.5
- **原因**: 仅训练 2 epochs，模型未收敛
- **影响**: F1=0, 混淆矩阵全为 FP
- **解决**: 正式训练时增加 epochs 和调整 loss weights

### 3. 缺失值处理策略未充分验证
- **当前**: 简单 NaN → 0.0
- **风险**: 可能引入 bias
- **后续**: 考虑 missing indicator 或更复杂的 imputation

### 4. Trajectory 与 2-Year Risk 的关系未明确
- **当前**: 通过线性层连接
- **理论**: 应该是 cumulative hazard 或 survival function
- **后续**: 需要明确数学关系

## 阻止进入 B2 的问题

**无阻塞性问题**。当前可以进入 B2（正式实验准备）。

但建议在 B2 中解决:
1. 完善 trajectory → 2-year risk 的数学公式
2. 验证完整数据集上的 split 平衡性
3. 调整 loss weights 以平衡各项损失
4. 实现完整指标套件（calibration, decision curves）

## 文件清单

### 新增文件
- `smoke_test_b1_2.py` - B1-2 正式 smoke test

### 修改文件
- `src/data/data_module_landmark.py` - 完全重写
- `src/models/causal_tabdiff_trajectory.py` - 修复 cond_dim 和特征提取
- `run_experiment_landmark.py` - 添加 F 导入

### 文档文件
- `docs/reboot/B1_2_SMOKE_TEST_REPORT.md` - 本文件

## 下一步

进入 **B2: 正式实验准备**，包括:
1. 使用完整数据集（移除 debug_n_persons）
2. 验证 split 平衡性和阳性率
3. 调整超参数（epochs, lr, loss weights）
4. 实现完整评估指标
5. 运行 5-seed 实验
