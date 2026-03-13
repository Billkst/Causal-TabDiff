# B2 Baseline 失败分析总结

## 核心发现

### 失败模型清单 (6个需要修复)

| # | 模型 | 层级 | 失败类型 | Seeds | 优先级 |
|---|------|------|---------|-------|--------|
| 1 | TSDiff | TSTR | XGBoost 依赖 | 42,52,62,72,82 | P1 |
| 2 | STaSy | TSTR | XGBoost 依赖 | 42,52,62,72,82 | P1 |
| 3 | iTransformer | Layer2 | 指标缺失 | 42,52,62,72,82 | P2 |
| 4 | SSSD | Layer2 | 指标缺失 | 42,52,62,72,82 | P2 |
| 5 | SurvTraj | Layer2 | 指标缺失 | 42,52,62,72,82 | P2 |
| 6 | CausalForest | Layer1 | 预测缺失 | 42,52,62,72,82 | P3 |

### 成功模型 (7个完全成功)

**Layer1 Direct** (6/6 完全成功):
- CausalForest, iTransformer, TabSyn, TabDiff, SurvTraj, SSSD

**Layer2 TSLib** (1/4 完全成功):
- TimeXer ✓

---

## 问题详解

### P1: TSTR XGBoost 依赖缺失

**错误**:
```
ImportError: `cupy` is required for handling CUDA buffer.
```

**位置**: `src/baselines/tstr_pipeline.py:51-58`

**原因**: XGBoost 配置中 `device='cpu'` 与 `tree_method='hist'` 冲突

**修复**: 移除 `device='cpu'` 参数

---

### P2: Layer2 指标计算失败

**症状**: 预测文件存在但指标文件缺失

**模型**: iTransformer, SSSD, SurvTraj (各5个seeds)

**原因**: 指标计算脚本未完成或出错

---

### P3: CausalForest 预测文件缺失

**症状**: 指标和图表存在但预测文件缺失

**原因**: 预测生成逻辑与其他模型不同

---

## 快速修复步骤

### 步骤1: 修复XGBoost (5分钟)
```bash
# 编辑 src/baselines/tstr_pipeline.py 第51-58行
# 移除 device='cpu' 参数
```

### 步骤2: 重新训练TSTR (60分钟)
```bash
for seed in 42 52 62 72 82; do
  python train_tstr_pipeline.py --model tsdiff --seed $seed
  python train_tstr_pipeline.py --model stasy --seed $seed
done
```

### 步骤3: 修复Layer2指标 (15分钟)
```bash
python train_tslib_layer2.py --model itransformer --compute-metrics-only
python train_tslib_layer2.py --model sssd --compute-metrics-only
python train_tslib_layer2.py --model survtraj --compute-metrics-only
```

### 步骤4: 生成CausalForest预测 (20分钟)
```bash
for seed in 42 52 62 72 82; do
  python train_causal_forest_b2.py --generate-predictions --seed $seed
done
```

---

## 文件位置

| 类型 | 路径 |
|------|------|
| 失败日志 | `logs/b2_baseline/tsdiff_stasy_fix/` |
| 对账报告 | `logs/b2_baseline/final_reconciliation.log` |
| 源代码 | `src/baselines/tstr_pipeline.py` |
| 训练脚本 | `train_tstr_pipeline.py`, `train_tslib_layer2.py` |
| 输出结果 | `outputs/b2_baseline/` |

---

## 验证命令

```bash
# 检查预测文件
find outputs/b2_baseline -name "*predictions*" | wc -l

# 检查指标文件
find outputs/b2_baseline -name "*metrics*" | wc -l

# 生成最终汇总
python scripts/generate_baseline_summary.py
```

