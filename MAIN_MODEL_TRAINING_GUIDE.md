# 主模型训练启动指南

**生成时间**: 2026-03-14  
**仓库路径**: `/home/UserData/ljx/Project_2/Causal-TabDiff`  
**Conda 环境**: `causal_tabdiff`  
**GPU**: 单张 NVIDIA 3090

---

## 一、模型确认

### 主模型名称
**CausalTabDiff** (因果表型扩散模型)

### 实现位置
- **核心模型**: `src/models/causal_tabdiff.py` (410 行)
- **训练入口**: `run_experiment_landmark.py` (117 行，已修复)
- **Wrapper**: `src/baselines/wrappers.py::CausalTabDiffWrapper` (第 858-1108 行)

### 与开题报告的一致性

**开题报告核心设计** (`docs/proposal/方案二：开题报告.md`):
> 设计时序因果扩散生成模型, 结合因果约束的梯度引导与反事实推理机制, 以解决回顾性医疗数据异构稀疏、伴生疾病共病模式缺失及幸存者偏差导致模型不可信的问题。

**当前实现对齐情况**:
1. ✅ **异构数据处理**: 使用 AnalogBits 编码将离散特征映射到连续空间 (第 901-922 行)
2. ✅ **正交双重注意力**: `OrthogonalDualAttentionBlock` 在时间维度和特征维度解耦 (第 27-78 行)
3. ✅ **因果约束梯度引导**: `LSTMDiscriminator` 评估生成轨迹与环境暴露的一致性 (第 80-94 行)
4. ✅ **反事实推理**: `predict_counterfactual_risk_gap()` 计算不同干预下的风险差 (第 190-197 行)
5. ✅ **风险轨迹预测**: `predict_cumulative_risk()` 输出纵向累计风险 (第 184-188 行)

**关键公式实现**:
- 能量函数 (第 106 行): `U(x_t) = ||f_φ(x_t) - α_target||²`
- 梯度引导采样 (第 110 行): `x_{t-1} = μ_θ(x_t) - s·σ_t·∇U(x_t) + σ_t·z`
- 累计风险 (第 186 行): `R = 1 - ∏(1 - h_t)`

---

## 二、当前状态

### 已完成工作
- ✅ Baseline 5-seed 实验完成并封版 (`BASELINE_COMPARISON_REPORT.md`)
- ✅ 训练入口修复并验证通过 (`run_experiment_landmark.py`)
- ✅ 数据模块对齐 (`src/data/data_module_landmark.py`)
- ✅ 仓库整理完成 (根目录文件从 104 减至 33)
- ✅ Smoke test 通过 (`smoke_test_b1_2.py`)

### 数据集信息
- **数据源**: `data/landmark_tables/unified_person_landmark_table.pkl`
- **总样本**: 2957 (person-landmark pairs)
- **正类比例**: 1.12% (2-year lung cancer risk)
- **数据划分**: 60% train / 20% val / 20% test (pid-level stratified)
- **特征维度**: 15 (4 baseline + 11 temporal per time)
- **时间步**: 3 (T0, T1, T2)
- **轨迹长度**: 7 (7-year risk trajectory)

### Baseline 最佳性能 (参考目标)
| 模型 | AUROC | AUPRC | F1 |
|------|-------|-------|-----|
| CausalForest | 0.5856 | 0.0256 | 0.0335 |
| iTransformer | 0.5234 | 0.0590 | 0.0136 |
| TSDiff | 0.5555 | 0.0184 | 0.0314 |

**主要指标**: AUPRC (在 1.12% 正类比例下比 AUROC 更可靠)

---

## 三、训练计划

### 阶段 1: 单 Seed 验证 (预计 2-3 小时)

**目标**: 验证完整训练流程，确认日志、checkpoint、指标正常

**命令**:
```bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff
conda activate causal_tabdiff

# 单 seed 完整训练
nohup python -u run_experiment_landmark.py \
  --epochs 100 \
  --batch_size 512 \
  --seed 42 \
  > logs/main_model_seed42.log 2>&1 &

# 实时监控
tail -f logs/main_model_seed42.log
```

**验收标准**:
- [ ] 训练完成无报错
- [ ] 日志包含每 epoch 的 loss (格式: `Epoch X/100 | Loss: Y.ZZZZ`)
- [ ] Loss 呈下降趋势
- [ ] 训练耗时合理 (参考: TSDiff 14s, iTransformer 169s)

---

### 阶段 2: 5-Seed 正式实验 (预计 10-15 小时)

**目标**: 按 baseline 相同协议运行 5-seed 实验

**命令** (串行执行，单 GPU):
```bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff
conda activate causal_tabdiff

# 依次运行 5 个 seed
for seed in 42 52 62 72 82; do
  echo "=== Starting seed $seed ==="
  nohup python -u run_experiment_landmark.py \
    --epochs 100 \
    --batch_size 512 \
    --seed $seed \
    > logs/main_model_seed${seed}.log 2>&1 &
  
  # 等待当前 seed 完成
  wait
  echo "=== Seed $seed completed ==="
done
```

**或使用 screen 会话** (推荐):
```bash
# 创建 5 个 screen 会话
for seed in 42 52 62 72 82; do
  screen -dmS causal_seed${seed} bash -c "
    cd /home/UserData/ljx/Project_2/Causal-TabDiff && \
    conda activate causal_tabdiff && \
    python -u run_experiment_landmark.py \
      --epochs 100 \
      --batch_size 512 \
      --seed $seed \
      > logs/main_model_seed${seed}.log 2>&1
  "
done

# 查看运行状态
screen -ls

# 进入某个会话查看
screen -r causal_seed42

# 退出会话 (不终止): Ctrl+A, D
```

**验收标准**:
- [ ] 5 个 seed 全部完成
- [ ] 每个 seed 生成日志文件 (`logs/main_model_seed*.log`)
- [ ] 5 个 seed 的最终 loss 有合理方差 (不完全相同)

---

### 阶段 3: 评估与汇总 (预计 1 小时)

**目标**: 计算 5-seed 平均指标，生成对比表

**当前问题**: `run_experiment_landmark.py` 仅输出训练 loss，未保存预测文件和评估指标

**需要补充的工作**:
1. 在训练脚本中添加预测保存逻辑
2. 在验证集上选择最优阈值 (F1 最大化)
3. 在测试集上应用固定阈值并计算指标
4. 保存预测文件 (`outputs/main_model/seed{X}_predictions.npz`)
5. 保存指标文件 (`outputs/main_model/seed{X}_metrics.json`)

**临时方案** (如评估脚本未实现):
```bash
# 手动运行评估 (需先实现评估逻辑)
python evaluate_model.py \
  --model_name CausalTabDiff \
  --predictions_dir outputs/main_model/ \
  --output_file outputs/main_model_summary.csv
```

**验收标准**:
- [ ] 生成 `outputs/main_model_summary.csv`
- [ ] 包含 AUROC、AUPRC、F1、Precision、Recall 等指标
- [ ] 格式与 `baseline_layer1_direct.csv` 一致

---

### 阶段 4: 结果分析 (预计 1 小时)

**对比维度**:
1. **AUPRC** (主要指标): 是否 > 0.0590 (iTransformer)
2. **AUROC** (辅助指标): 是否 > 0.5856 (CausalForest)
3. **F1**: 是否 > 0.0335 (CausalForest)

**决策树**:
- ✅ AUPRC 显著优于 baseline → 进入论文写作
- ⚠️ AUPRC 与 baseline 持平 → 分析原因，考虑超参数调优
- ❌ AUPRC 劣于 baseline → 深度调试，检查模型实现

---

## 四、关键注意事项

### 1. 日志规范
当前 `run_experiment_landmark.py` 已符合规范:
- 每 epoch 输出: `Epoch X/100 | Loss: Y.ZZZZ`
- 训练开始输出: `Training started | train_batches=X | val_batches=Y`
- 训练结束输出: `Training complete`

### 2. 阈值协议
**必须遵守**:
- 验证集 F1 最大化选阈值
- 测试集固定使用该阈值
- 禁止在测试集重新调阈值

### 3. 预测文件格式
需保存:
- `val_y_pred`: 验证集预测概率
- `val_y_true`: 验证集真实标签
- `test_y_pred`: 测试集预测概率
- `test_y_true`: 测试集真实标签

### 4. 潜在阻塞点
- ⚠️ 当前训练脚本未实现评估逻辑
- ⚠️ 需补充预测保存和指标计算代码
- ⚠️ 汇总脚本可能需要新建

---

## 五、时间估算

| 阶段 | 耗时 | 备注 |
|------|------|------|
| 单 seed 验证 | 2-3 小时 | 串行 |
| 5-seed 训练 | 10-15 小时 | 串行 (单 GPU) |
| 评估汇总 | 1 小时 | 需补充代码 |
| 结果分析 | 1 小时 | 对比 baseline |
| **总计** | **14-20 小时** | 约 1 天 |

---

## 六、快速启动命令 (复制即用)

```bash
# 进入项目目录
cd /home/UserData/ljx/Project_2/Causal-TabDiff

# 激活环境
conda activate causal_tabdiff

# 单 seed 验证
nohup python -u run_experiment_landmark.py \
  --epochs 100 \
  --batch_size 512 \
  --seed 42 \
  > logs/main_model_seed42.log 2>&1 &

# 查看日志
tail -f logs/main_model_seed42.log

# 查看进程
ps aux | grep run_experiment_landmark

# 终止进程 (如需要)
pkill -f run_experiment_landmark
```

---

## 七、下一步行动

**立即执行**:
1. 启动单 seed 验证 (seed 42)
2. 监控日志确认训练正常
3. 等待训练完成后检查 loss 下降趋势

**后续补充** (如需要):
1. 在训练脚本中添加评估逻辑
2. 实现预测保存和指标计算
3. 编写汇总脚本生成对比表

---

**准备就绪，可以开始训练。**
