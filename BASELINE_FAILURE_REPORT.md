# B2 Baseline 失败分析与重新训练计划

## 执行摘要

**总体状态**: 部分失败，需要有针对性的修复
- ✓ 成功: 6个模型 (Layer1 Direct)
- ⚠️ 部分失败: 3个模型 (Layer2 指标缺失)
- ✗ 完全失败: 2个模型 (TSTR Pipeline)
- ✗ 文件缺失: 1个模型 (CausalForest 预测)

---

## 1. 关键失败：TSTR Pipeline XGBoost 依赖

### 问题描述
TSDiff 和 STaSy 在训练下游分类器时失败

### 错误信息
```
ImportError: `cupy` is required for handling CUDA buffer.
  File: train_tstr_pipeline.py:67
  Location: src/baselines/tstr_pipeline.py:77 (classifier.fit)
```

### 根本原因
XGBoost 配置中设置了 `device='cpu'`，但 XGBoost 仍尝试导入 cupy（CUDA库）。这是 XGBoost 的已知问题。

### 受影响的模型
- TSDiff (TSTR) - 所有 seeds (42, 52, 62, 72, 82)
- STaSy (TSTR) - 所有 seeds (42, 52, 62, 72, 82)

### 日志位置
```
/home/UserData/ljx/Project_2/Causal-TabDiff/logs/b2_baseline/tsdiff_stasy_fix/
  - tsdiff_final.log
  - stasy_final.log
```

### 修复方案
在 `src/baselines/tstr_pipeline.py` 第 51-58 行修改 XGBoost 初始化：

**当前代码**:
```python
self.classifier = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    tree_method='hist',
    device='cpu'
)
```

**修复方案 A** (推荐 - 使用 sklearn 替代):
```python
from sklearn.ensemble import GradientBoostingClassifier
self.classifier = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

**修复方案 B** (保留 XGBoost - 移除 device 参数):
```python
self.classifier = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    tree_method='hist'
    # 移除 device='cpu' 参数
)
```

---

## 2. Layer2 指标缺失

### 问题描述
Layer2 模型的预测文件存在，但指标计算失败

### 受影响的模型和 Seeds
| 模型 | Seeds | 状态 |
|------|-------|------|
| iTransformer | 42, 52, 62, 72, 82 | pred=✓ metrics=✗ |
| SSSD | 42, 52, 62, 72, 82 | pred=✓ metrics=✗ |
| SurvTraj | 42, 52, 62, 72, 82 | pred=✓ metrics=✗ |

### 文件位置
```
输出: /home/UserData/ljx/Project_2/Causal-TabDiff/outputs/b2_baseline/layer2/
日志: /home/UserData/ljx/Project_2/Causal-TabDiff/logs/b2_baseline/layer2/
```

### 原因分析
从 `final_reconciliation.log` 可见，这些模型的预测文件已生成，但指标计算脚本未能正确处理。可能原因：
1. 指标计算脚本未被调用
2. 指标计算脚本出错但未记录
3. 输出路径不匹配

### 修复方案
重新运行指标计算脚本：
```bash
python train_tslib_layer2.py --model itransformer --compute-metrics-only
python train_tslib_layer2.py --model sssd --compute-metrics-only
python train_tslib_layer2.py --model survtraj --compute-metrics-only
```

---

## 3. CausalForest 预测文件缺失

### 问题描述
CausalForest 的指标和图表已生成，但预测文件缺失

### 受影响的 Seeds
42, 52, 62, 72, 82 (所有 5 个)

### 文件位置
```
预期位置: /home/UserData/ljx/Project_2/Causal-TabDiff/outputs/b2_baseline/layer1/
日志: /home/UserData/ljx/Project_2/Causal-TabDiff/logs/b2_baseline/layer1/
```

### 原因分析
CausalForest 的预测生成逻辑可能与其他模型不同，或者预测文件保存路径不同。

### 修复方案
```bash
python train_causal_forest_b2.py --generate-predictions --seed 42
python train_causal_forest_b2.py --generate-predictions --seed 52
python train_causal_forest_b2.py --generate-predictions --seed 62
python train_causal_forest_b2.py --generate-predictions --seed 72
python train_causal_forest_b2.py --generate-predictions --seed 82
```

---

## 4. 成功的 Baseline

### Layer1 (Direct) - 完全成功 ✓
| 模型 | Seeds | 状态 |
|------|-------|------|
| CausalForest | 42, 52, 62, 72, 82 | metrics=✓ plots=✓ |
| iTransformer | 42, 52, 62, 72, 82 | pred=✓ metrics=✓ plots=✓ |
| TabSyn | 42, 52, 62, 72, 82 | pred=✓ metrics=✓ plots=✓ |
| TabDiff | 42, 52, 62, 72, 82 | pred=✓ metrics=✓ plots=✓ |
| SurvTraj | 42, 52, 62, 72, 82 | pred=✓ metrics=✓ plots=✓ |
| SSSD | 42, 52, 62, 72, 82 | pred=✓ metrics=✓ plots=✓ |

### Layer2 (TSLib) - 部分成功
| 模型 | Seeds | 状态 |
|------|-------|------|
| TimeXer | 42, 52, 62, 72, 82 | pred=✓ metrics=✓ plots=✓ |
| iTransformer | 42, 52, 62, 72, 82 | pred=✓ metrics=✗ |
| SSSD | 42, 52, 62, 72, 82 | pred=✓ metrics=✗ |
| SurvTraj | 42, 52, 62, 72, 82 | pred=✓ metrics=✗ |

---

## 5. 重新训练执行计划

### 优先级 1: 修复 TSTR XGBoost 依赖 (立即执行)

**步骤 1**: 修改 `src/baselines/tstr_pipeline.py`
```bash
# 编辑文件，应用修复方案 A 或 B
vim src/baselines/tstr_pipeline.py
```

**步骤 2**: 重新训练 TSDiff (所有 seeds)
```bash
for seed in 42 52 62 72 82; do
  python train_tstr_pipeline.py --model tsdiff --seed $seed
done
```

**步骤 3**: 重新训练 STaSy (所有 seeds)
```bash
for seed in 42 52 62 72 82; do
  python train_tstr_pipeline.py --model stasy --seed $seed
done
```

### 优先级 2: 修复 Layer2 指标 (次日执行)

```bash
# 重新计算 iTransformer Layer2 指标
python train_tslib_layer2.py --model itransformer --compute-metrics-only

# 重新计算 SSSD Layer2 指标
python train_tslib_layer2.py --model sssd --compute-metrics-only

# 重新计算 SurvTraj Layer2 指标
python train_tslib_layer2.py --model survtraj --compute-metrics-only
```

### 优先级 3: 生成 CausalForest 预测 (次日执行)

```bash
for seed in 42 52 62 72 82; do
  python train_causal_forest_b2.py --generate-predictions --seed $seed
done
```

---

## 6. 验证清单

完成重新训练后，使用以下命令验证：

```bash
# 运行最终对账
python scripts/reconciliation_check.py

# 检查所有预测文件
find outputs/b2_baseline -name "*predictions*" | wc -l

# 检查所有指标文件
find outputs/b2_baseline -name "*metrics*" | wc -l

# 生成最终汇总表格
python scripts/generate_baseline_summary.py
```

---

## 7. 关键文件位置

### 训练脚本
- TSTR Pipeline: `/home/UserData/ljx/Project_2/Causal-TabDiff/train_tstr_pipeline.py`
- Layer2 训练: `/home/UserData/ljx/Project_2/Causal-TabDiff/train_tslib_layer2.py`
- CausalForest: `/home/UserData/ljx/Project_2/Causal-TabDiff/train_causal_forest_b2.py`

### 源代码
- TSTR Pipeline 实现: `/home/UserData/ljx/Project_2/Causal-TabDiff/src/baselines/tstr_pipeline.py`
- 模型 Wrappers: `/home/UserData/ljx/Project_2/Causal-TabDiff/src/baselines/wrappers.py`

### 日志和输出
- 日志根目录: `/home/UserData/ljx/Project_2/Causal-TabDiff/logs/b2_baseline/`
- 输出根目录: `/home/UserData/ljx/Project_2/Causal-TabDiff/outputs/b2_baseline/`
- 汇总表格: `/home/UserData/ljx/Project_2/Causal-TabDiff/outputs/b2_baseline/summaries/`

---

## 8. 预期时间表

| 任务 | 预计耗时 | 优先级 |
|------|---------|--------|
| 修复 XGBoost 依赖 | 5 分钟 | P1 |
| 重新训练 TSDiff (5 seeds) | 30 分钟 | P1 |
| 重新训练 STaSy (5 seeds) | 30 分钟 | P1 |
| 重新计算 Layer2 指标 | 15 分钟 | P2 |
| 生成 CausalForest 预测 | 20 分钟 | P3 |
| **总计** | **~100 分钟** | - |

---

## 9. 注意事项

1. **环境激活**: 所有命令需在 `causal_tabdiff` conda 环境中执行
   ```bash
   conda activate causal_tabdiff
   ```

2. **GPU 可用性**: 确保 GPU 可用（CUDA 11.8+）

3. **日志监控**: 使用以下命令实时监控训练进度
   ```bash
   tail -f logs/b2_baseline/tsdiff_stasy_正式/tsdiff_seed42.log
   ```

4. **备份**: 重新训练前建议备份现有结果
   ```bash
   cp -r outputs/b2_baseline outputs/b2_baseline_backup_$(date +%Y%m%d)
   ```

