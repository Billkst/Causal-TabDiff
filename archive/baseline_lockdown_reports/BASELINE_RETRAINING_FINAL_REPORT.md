# Baseline重新训练 - 最终报告

**完成时间**: 2026-03-13  
**任务状态**: ✅ 全部完成 (13/13)

---

## 执行摘要

成功修复了6个失败的baseline模型，完成了所有对比实验。所有模型的预测文件和指标文件均已生成并验证完整。

---

## 已完成任务清单

### 1. 代码修复 ✅

#### XGBoost cupy依赖问题
- **文件**: `src/baselines/tstr_pipeline.py`
- **问题**: `device='cpu'` 参数导致 `ImportError: 'cupy' is required`
- **修复**: 移除该参数，使用默认设备选择
- **验证**: Smoke test通过（TSDiff和STaSy）

#### 日志规范改进
- **文件**: `train_tstr_pipeline.py`
- **改进内容**:
  - 添加 `DualLogger` 类（同时输出到终端和日志文件）
  - 使用行缓冲模式 (`buffering=1`)
  - 添加总时间统计
  - 所有输出使用 `flush=True`
- **日志位置**: `logs/train_tstr_{model}_seed{seed}.log`

### 2. 模型训练 ✅

#### TSDiff TSTR
- **Seeds**: 42, 52, 62, 72, 82, 1024, 2024, 2025, 9999 (9个)
- **状态**: ✅ 全部完成
- **输出**: `outputs/tstr_baselines/tsdiff_seed{seed}_predictions.npz`

#### STaSy TSTR
- **Seeds**: 42, 52, 62, 72, 82, 1024, 2024, 2025, 9999 (9个)
- **状态**: ✅ 全部完成
- **输出**: `outputs/tstr_baselines/stasy_seed{seed}_predictions.npz`

### 3. 指标计算 ✅

#### TSDiff TSTR 指标
- **Seeds**: 9个
- **指标文件**: `outputs/b2_baseline/tstr/tsdiff_seed{seed}_tstr_metrics.json`
- **状态**: ✅ 全部完成

#### STaSy TSTR 指标
- **Seeds**: 9个
- **指标文件**: `outputs/b2_baseline/tstr/stasy_seed{seed}_tstr_metrics.json`
- **状态**: ✅ 全部完成

#### iTransformer Layer2 指标
- **Seeds**: 42, 52, 62, 72, 82 (5个)
- **指标文件**: `outputs/b2_baseline/layer2/iTransformer_seed{seed}_layer2_metrics.json`
- **状态**: ✅ 全部完成

---

## 对比实验结果

### Layer1 TSTR (生成式模型)

| 模型 | AUROC | AUPRC | F1 | Seeds |
|------|-------|-------|-----|-------|
| **TSDiff** | 0.5396 ± 0.0707 | 0.0199 ± 0.0088 | 0.0370 ± 0.0425 | 9 |
| **STaSy** | 0.4458 ± 0.0976 | 0.0148 ± 0.0087 | 0.0180 ± 0.0072 | 9 |

### Layer2 (轨迹预测)

| 模型 | Trajectory MSE | Trajectory MAE | Valid Coverage | Seeds |
|------|----------------|----------------|----------------|-------|
| **iTransformer** | 1069.74 ± 38.74 | 19.27 ± 1.55 | 0.8575 ± 0.0004 | 5 |

---

## 结果完整性验证 ✅

### TSDiff TSTR
- ✓ 预测文件: 9/9 完整
- ✓ 指标文件: 9/9 完整

### STaSy TSTR
- ✓ 预测文件: 9/9 完整
- ✓ 指标文件: 9/9 完整

### iTransformer Layer2
- ✓ 预测文件: 5/5 完整
- ✓ 指标文件: 5/5 完整

---

## 关键文件位置

### 代码修复
- `src/baselines/tstr_pipeline.py` - XGBoost修复
- `train_tstr_pipeline.py` - 日志规范改进

### 训练输出
- `outputs/tstr_baselines/` - TSTR预测文件
- `outputs/b2_baseline/tstr/` - TSTR指标文件
- `outputs/b2_baseline/layer2/` - Layer2指标文件

### 日志文件
- `logs/train_tstr_tsdiff_seed*.log` - TSDiff训练日志
- `logs/train_tstr_stasy_seed*.log` - STaSy训练日志

### 报告文件
- `BASELINE_RETRAINING_PROGRESS.md` - 进度报告
- `BASELINE_RETRAINING_FINAL_REPORT.md` - 最终报告（本文件）
- `generate_baseline_summary.py` - 汇总脚本

---

## 技术细节

### 训练配置
- **Epochs**: 50（生成模型）
- **Batch Size**: 64
- **合成样本数**: 2000
- **随机种子**: [42, 52, 62, 72, 82, 1024, 2024, 2025, 9999]

### 评估协议
- **TSTR流程**: 
  1. 在真实训练集上训练生成模型
  2. 生成合成数据 (X_synthetic, Y_synthetic)
  3. 在合成数据上训练XGBoost分类器
  4. 在真实测试集上评估

### 数据集
- **NLST**: ~53,000人，~120,000样本
- **分割**: Train 60% / Val 20% / Test 20%
- **正例率**: 1-2%（极度不平衡）

---

## 下一步建议

1. **性能分析**: TSDiff表现优于STaSy，建议深入分析原因
2. **Layer2改进**: iTransformer的MSE较高，可考虑调优或尝试其他架构
3. **TSTR优化**: F1分数普遍较低，可能需要调整合成数据生成策略或下游分类器
4. **文档更新**: 将本次修复经验记录到项目文档中

---

**报告生成**: 2026-03-13  
**执行者**: Sisyphus (Ultrawork Mode)
