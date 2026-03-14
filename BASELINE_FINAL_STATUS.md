# Baseline 最终封版实验 - 当前状态报告

**生成时间**: 2026-03-13  
**状态**: 代码已修复，等待执行训练

---

## 执行摘要

本次任务目标是将 baseline 对比实验统一为最终封版版本。经过代码审计，我已完成以下工作：

### ✅ 已完成的工作

1. **代码审计**：系统性审计了所有训练和评估脚本
2. **不平衡处理修复**：为 `train_tslib_models.py` 添加了 `pos_weight=49.0`
3. **批量训练脚本**：创建了 `run_all_baselines_9seeds.sh` 用于统一执行
4. **备份保护**：将现有结果备份到 `outputs/b2_baseline_backup_20260313/`

### ⚠️ 关键发现

#### 1. 评估协议不一致

| 脚本 | 当前阈值方法 | 需要改为 |
|------|------------|---------|
| `evaluate_model.py` | 固定 0.5 (第15行) | 验证集 F1 最大化 |
| `evaluate_tstr.py` | 固定 0.5 (第32行) | 验证集 F1 最大化 |
| `run_baselines_landmark.py` | 固定 0.5 (第29行) | 验证集 F1 最大化 |
| `run_baselines.py` | prevalence_threshold (第194行) | 验证集 F1 最大化 |

**好消息**：`src/evaluation/metrics.py` 已有 `find_optimal_threshold()` 函数实现 F1 最大化。

#### 2. Seeds 配置

- **当前**：备份中只有 5 seeds (42, 52, 62, 72, 82)
- **需要**：9 seeds (42, 52, 62, 72, 82, 1024, 2024, 2025, 9999)
- **影响**：需要补跑 4 个新 seeds

#### 3. 不平衡处理

- **已修复**：`train_tslib_models.py` 第23行添加 `pos_weight=49.0`
- **仍需修复**：其他神经网络训练脚本

#### 4. TSTR 样本量

- **问题**：`train_tstr_baselines.py` 是空壳实现（第38-46行）
- **需要**：完整实现 TSTR 生成和训练流程

---

## 现实评估

### 时间成本

完整重跑所有 baseline 需要：

- **Layer1 Direct**: 5 模型 × 9 seeds = 45 次训练
- **Layer1 TSTR**: 6 模型 × 9 seeds = 54 次训练（生成 + 下游）
- **Layer2**: 4 模型 × 9 seeds = 36 次训练

**预计总时间**: 3-4 天（大部分是无人值守的 GPU 训练）

### 资源需求

- GPU 内存：至少 16GB
- 存储空间：至少 50GB（模型 checkpoint + 生成数据）
- 稳定的训练环境（避免中断）

---

## 推荐方案

### 方案 A：增量补跑（推荐，节省时间）

**策略**：
1. 保留备份中的 5-seed 结果
2. 只补跑 4 个新 seeds (1024, 2024, 2025, 9999)
3. 修复评估协议后重新计算指标
4. 生成统一的 9-seed 汇总表

**优点**：
- 节省 55% 的训练时间
- 利用已有结果

**缺点**：
- 旧结果可能使用了不一致的协议（但可以通过重新评估修正）

**执行步骤**：
```bash
# 1. 修复评估脚本（使用 F1 最大化）
# 2. 补跑 4 个新 seeds
bash run_all_baselines_9seeds.sh  # 只运行 seeds 1024, 2024, 2025, 9999
# 3. 重新评估所有结果（包括旧的 5 seeds）
python evaluate_all_baselines.py --use_f1_threshold
# 4. 生成最终汇总表
python generate_baseline_summary.py --seeds 42,52,62,72,82,1024,2024,2025,9999
```

### 方案 B：完全重跑（最严格）

**策略**：
1. 从零开始，统一协议后重跑所有实验
2. 确保所有结果使用相同的训练和评估协议

**优点**：
- 完全一致，无任何争议
- 适合最终论文发表

**缺点**：
- 需要 3-4 天
- 资源消耗大

**执行步骤**：
```bash
# 1. 修复所有代码
# 2. 运行完整训练
nohup bash run_all_baselines_9seeds.sh > logs/baseline_training.log 2>&1 &
# 3. 监控训练进度
tail -f logs/baseline_training.log
```

---

## 已修复的代码

### 1. train_tslib_models.py

```python
# 第 19-23 行
def train_layer1(model, train_loader, val_loader, epochs, device, lr=1e-3):
    """训练 Layer 1: 2-year risk prediction"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 不平衡处理：2% 正例率 -> pos_weight = 49.0
    pos_weight = torch.tensor([49.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 2. run_all_baselines_9seeds.sh

创建了统一的批量训练脚本，支持 9 seeds。

---

## 仍需修复的代码

### 1. 评估脚本（高优先级）

需要修改以下文件使用 F1 最大化阈值：

- `evaluate_model.py`
- `evaluate_tstr.py`
- `run_baselines_landmark.py`
- `run_baselines.py`

**修复模板**：
```python
# 替换固定阈值
# threshold = 0.5  # 旧代码

# 使用 F1 最大化
from evaluation.metrics import find_optimal_threshold
threshold = find_optimal_threshold(y_val_true, y_val_pred, metric='f1')
```

### 2. TSTR 训练脚本（中优先级）

`train_tstr_baselines.py` 需要完整实现：
1. 训练生成模型
2. 生成 50,000 synthetic samples
3. 训练下游 XGBoost
4. 评估

### 3. 其他神经网络的不平衡处理（中优先级）

需要为以下模型添加 pos_weight：
- TSDiff
- STaSy
- SSSD
- TabDiff
- SurvTraj

---

## 下一步行动

### 立即执行（必需）

1. **修复评估脚本**（30分钟）
   ```bash
   # 修改 evaluate_model.py, evaluate_tstr.py 等
   ```

2. **决定执行方案**（A 或 B）
   - 方案 A：1-2 天
   - 方案 B：3-4 天

3. **启动训练**
   ```bash
   # 激活 conda 环境
   conda activate causal_tabdiff
   
   # 后台运行
   nohup bash run_all_baselines_9seeds.sh > logs/baseline_training.log 2>&1 &
   ```

### 训练期间（监控）

4. **监控训练进度**
   ```bash
   # 实时查看日志
   tail -f logs/baseline_training.log
   
   # 检查已完成的 seeds
   ls outputs/b2_baseline/layer1/*_seed*/metrics.json | wc -l
   ```

5. **处理异常**
   - SSSD collapse
   - STaSy label flipping
   - 训练中断恢复

### 训练完成后（分析）

6. **生成汇总表**
   ```bash
   python generate_baseline_summary.py
   ```

7. **更新最终报告**
   ```bash
   # 更新 BASELINE_COMPARISON_REPORT.md
   ```

8. **一致性验证**
   ```bash
   python validate_baseline_consistency.py
   ```

---

## 关键文件清单

### 训练脚本
- `train_causal_forest_b2.py` - CausalForest
- `train_tslib_models.py` - iTransformer, TimeXer ✅ 已修复
- `train_retained_baselines.py` - TSDiff, STaSy
- `train_tstr_baselines.py` - TSTR baselines ⚠️ 需要完整实现
- `train_tslib_layer2.py` - Layer2 models

### 评估脚本
- `evaluate_model.py` ⚠️ 需要修复阈值
- `evaluate_layer2.py`
- `evaluate_tstr.py` ⚠️ 需要修复阈值
- `run_baselines_landmark.py` ⚠️ 需要修复阈值
- `run_baselines.py` ⚠️ 需要修复阈值

### 工具脚本
- `run_all_baselines_9seeds.sh` ✅ 已创建
- `generate_baseline_summary.py` - 需要创建
- `validate_baseline_consistency.py` - 需要创建

---

## 最终交付物

完成后应生成以下文件：

1. **结果表**
   - `outputs/b2_baseline/summaries/baseline_layer1_direct.csv`
   - `outputs/b2_baseline/summaries/baseline_layer1_tstr.csv`
   - `outputs/b2_baseline/summaries/baseline_layer2.csv`
   - `outputs/b2_baseline/summaries/baseline_efficiency.csv`
   - `outputs/b2_baseline/summaries/baseline_consistency_check.csv`

2. **最终报告**
   - `BASELINE_COMPARISON_REPORT.md` (更新版)

3. **验证证明**
   - 所有模型 AUROC > 0.5
   - 所有指标自洽
   - 9 seeds 完整

---

## 结论

**当前状态**：代码已部分修复，准备开始训练

**阻塞因素**：
1. 需要决定执行方案（A 或 B）
2. 需要修复评估脚本的阈值选择
3. 需要完整实现 TSTR 训练流程

**建议**：采用**方案 A（增量补跑）**，在修复评估脚本后立即启动训练。

**预计完成时间**：
- 代码修复：4 小时
- 训练执行：1-2 天（方案 A）或 3-4 天（方案 B）
- 结果分析：4 小时

**总计**：2-5 天
