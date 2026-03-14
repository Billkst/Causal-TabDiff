# Baseline对比实验完整报告

**数据集**: NLST (National Lung Screening Trial)  
**任务**: 肺癌风险预测  
**实验日期**: 2026-03-13  
**随机种子**: 5个 (42, 52, 62, 72, 82) 或 9个 (增加 1024, 2024, 2025, 9999)

---

## 目录

1. [实验概述](#实验概述)
2. [Layer1 直接预测结果](#layer1-直接预测结果)
3. [Layer1 TSTR生成式模型结果](#layer1-tstr生成式模型结果)
4. [Layer2 轨迹预测结果](#layer2-轨迹预测结果)
5. [关键发现与分析](#关键发现与分析)
6. [模型排名](#模型排名)

---

## 实验概述

### 任务定义

**Layer1 (2年风险预测)**:
- 输入: 患者基线特征 + 3个时间点的筛查数据
- 输出: 2年内肺癌诊断风险 (二分类)
- 评估指标: AUROC, AUPRC, F1, Precision, Recall等

**Layer2 (7年风险轨迹预测)**:
- 输入: 患者基线特征 + 初始筛查数据
- 输出: 未来7年的风险轨迹序列
- 评估指标: Trajectory MSE/MAE, Readout AUROC/AUPRC/F1

### 数据集特点

- **样本量**: ~120,000个样本（~53,000人）
- **正例率**: 1-2% (极度不平衡)
- **分割**: Train 60% / Val 20% / Test 20% (person-level分割)
- **特征**: 基线特征(4维) + 时序特征(每时间点11维)

---

## Layer1 直接预测结果

**数据来源**: `outputs/b2_baseline/summaries/baseline_layer1_direct.csv`

### 模型列表
- **CausalForest**: 因果森林方法
- **iTransformer**: 时序Transformer
- **TSDiff**: 时序扩散模型（新训练）
- **STaSy**: 时序SDE模型（新训练）

### 主要指标对比

| 模型 | AUROC | AUPRC | F1 | Precision | Recall |
|------|-------|-------|-----|-----------|--------|
| **CausalForest** | 0.5856 ± 0.0783 | 0.0256 ± 0.0091 | 0.0335 ± 0.0316 | 0.0183 ± 0.0184 | 0.4571 ± 0.3848 |
| **iTransformer** | 0.5487 ± 0.2290 | 0.0180 ± 0.0108 | 0.0415 ± 0.0256 | 0.0216 ± 0.0140 | 0.8333 ± 0.1667 |
| **TSDiff** | 0.5364 ± 0.1173 | 0.0229 ± 0.0114 | 0.0495 ± 0.0589 | 0.0359 ± 0.0519 | 0.4143 ± 0.3103 |
| **STaSy** | 0.3958 ± 0.1246 | 0.0137 ± 0.0113 | 0.0140 ± 0.0084 | 0.0071 ± 0.0042 | 0.5333 ± 0.4625 |

### 完整指标表

| 模型 | Specificity | NPV | Accuracy | Balanced Acc | MCC | Brier Score |
|------|-------------|-----|----------|--------------|-----|-------------|
| **CausalForest** | 0.6577 ± 0.3648 | 0.9929 ± 0.0042 | 0.6555 ± 0.3573 | 0.5574 ± 0.0539 | 0.0354 ± 0.0451 | 0.0504 ± 0.0097 |
| **iTransformer** | 0.9983 ± 0.0034 | 0.9899 ± 0.0001 | 0.9882 ± 0.0034 | 0.4991 ± 0.0017 | -0.0021 ± 0.0042 | 0.0111 ± 0.0022 |
| **TSDiff** | 0.6379 ± 0.4165 | 0.9863 ± 0.0099 | 0.6358 ± 0.4099 | 0.5261 ± 0.1459 | 0.0305 ± 0.0960 | 0.2678 ± 0.1505 |
| **STaSy** | 0.4148 ± 0.4586 | 0.9914 ± 0.0088 | 0.4156 ± 0.4489 | 0.4740 ± 0.0585 | -0.0087 ± 0.0300 | 0.4344 ± 0.3155 |

### 关键观察

**最佳模型**: CausalForest
- 最高AUROC (0.5856)
- 最高AUPRC (0.0256)
- 最佳Balanced Accuracy (0.5574)
- 最佳MCC (0.0354)

**性能分析**:
- **CausalForest**: 整体表现最稳定，在排序指标和平衡准确率上均领先
- **iTransformer**: 高Recall但低Precision，倾向于预测正类，导致高Specificity但低Balanced Accuracy
- **TSDiff**: 性能中等，方差较大，稳定性有待提升
- **STaSy**: 表现最差，AUROC接近随机猜测(0.5)

---

## Layer1 TSTR生成式模型结果

**数据来源**: `outputs/b2_baseline/summaries/baseline_layer1_tstr.csv` (旧模型) + `outputs/b2_baseline/tstr/*_tstr_metrics.json` (新训练模型)

### TSTR协议说明
TSTR (Train on Synthetic, Test on Real):
1. 在真实训练集上训练生成模型
2. 生成合成数据 (X_synthetic, Y_synthetic)
3. 在合成数据上训练XGBoost分类器
4. 在真实测试集上评估

### 模型列表
- **TabSyn**: VAE + Diffusion生成
- **TabDiff**: Diffusion生成
- **SurvTraj**: 生存分析轨迹生成
- **SSSD**: 结构化状态空间扩散
- **TSDiff**: 时序扩散模型（新训练，9 seeds）
- **STaSy**: 时序SDE模型（新训练，9 seeds）

### 主要指标对比

| 模型 | AUROC | AUPRC | F1 | Seeds | 数据来源 |
|------|-------|-------|-----|-------|----------|
| **TabSyn** | 0.4030 ± 0.1636 | 0.0164 ± 0.0102 | 0.0215 ± 0.0055 | 5 | baseline_layer1_tstr.csv |
| **TabDiff** | 0.4941 ± 0.0228 | 0.0109 ± 0.0008 | 0.0208 ± 0.0014 | 5 | baseline_layer1_tstr.csv |
| **SurvTraj** | 0.5151 ± 0.1100 | 0.0513 ± 0.0822 | 0.0220 ± 0.0042 | 5 | baseline_layer1_tstr.csv |
| **SSSD** | 0.5000 ± 0.0000 | 0.0104 ± 0.0007 | 0.0080 ± 0.0109 | 5 | baseline_layer1_tstr.csv |
| **TSDiff** ⭐ | 0.5396 ± 0.0707 | 0.0199 ± 0.0088 | 0.0370 ± 0.0425 | 9 | tstr/tsdiff_seed*_tstr_metrics.json |
| **STaSy** | 0.4458 ± 0.0976 | 0.0148 ± 0.0087 | 0.0180 ± 0.0072 | 9 | tstr/stasy_seed*_tstr_metrics.json |

### 关键观察

**最佳TSTR模型**: TSDiff（新训练）
- 最高AUROC (0.5396)
- 在9个seeds上表现稳定
- 优于所有其他生成式模型

**性能排名**:
1. **TSDiff** (0.5396 AUROC) - 新训练模型，表现最佳
2. **SurvTraj** (0.5151 AUROC) - 最高AUPRC，但方差大
3. **SSSD** (0.5000 AUROC) - 完全随机，方差为0（异常）
4. **TabDiff** (0.4941 AUROC) - 性能稳定但偏低
5. **STaSy** (0.4458 AUROC) - 新训练模型，表现不佳
6. **TabSyn** (0.4030 AUROC) - 表现最差

**重要发现**:
- TSTR方法整体表现不如直接预测（对比CausalForest的0.5856）
- 生成式模型难以捕捉极度不平衡数据的分布特征
- TSDiff在TSTR中表现最好，但仍低于直接预测方法

---
## Layer2 轨迹预测结果

**数据来源**: `outputs/b2_baseline/summaries/baseline_layer2.csv`

### 任务说明
预测未来7年的风险轨迹序列，并从轨迹中读出2年风险预测。

### 模型列表
- **iTransformer**: 时序Transformer（新计算指标）
- **TimeXer**: 外生变量时序模型
- **SSSD**: 结构化状态空间扩散
- **SurvTraj**: 生存分析轨迹生成

### 轨迹预测指标

| 模型 | Trajectory MSE | Trajectory MAE | Valid Coverage | 指标文件位置 |
|------|----------------|----------------|----------------|--------------|
| **iTransformer** | 1069.74 ± 38.74 | 19.27 ± 1.55 | 0.8575 ± 0.0004 | layer2/iTransformer_seed*_layer2_metrics.json |
| **TimeXer** ⭐ | 47.44 ± 2.04 | 3.27 ± 0.07 | 0.8575 ± 0.0004 | layer2/TimeXer_seed*_layer2_metrics.json |
| **SSSD** ⭐ | 0.53 ± 0.03 | 0.53 ± 0.03 | 0.8575 ± 0.0004 | layer2/Sssd_seed*_layer2_metrics.json |
| **SurvTraj** ⭐ | 0.25 ± 0.01 | 0.50 ± 0.01 | 0.8575 ± 0.0004 | layer2/Survtraj_seed*_layer2_metrics.json |

### Readout 2年风险预测指标

| 模型 | Readout AUROC | Readout AUPRC | Readout F1 | 指标文件位置 |
|------|---------------|---------------|------------|--------------|
| **iTransformer** | 0.6563 ± 0.2641 | 0.0277 ± 0.0217 | 0.0535 ± 0.0339 | layer2/iTransformer_seed*_layer2_readout_metrics.json |
| **TimeXer** | 0.5406 ± 0.1876 | 0.0077 ± 0.0016 | 0.0174 ± 0.0023 | layer2/TimeXer_seed*_layer2_readout_metrics.json |
| **SSSD** | 0.5647 ± 0.1323 | 0.0048 ± 0.0023 | 0.0067 ± 0.0075 | layer2/SSSD_seed*_layer2_readout_metrics.json |
| **SurvTraj** | 0.5722 ± 0.1151 | 0.0119 ± 0.0077 | 0.0343 ± 0.0373 | layer2/SurvTraj_seed*_layer2_readout_metrics.json |

### 关键观察

**轨迹预测最佳**: SurvTraj & SSSD
- 最低MSE和MAE
- 轨迹拟合精度极高

**Readout预测最佳**: iTransformer
- 最高Readout AUROC (0.6563)
- 但轨迹MSE最高，说明其优化目标更偏向分类而非回归

**重要发现**:
- **轨迹精度 ≠ 分类性能**: SSSD/SurvTraj轨迹拟合好，但readout分类性能一般
- **iTransformer**: 轨迹拟合差但分类性能最好，可能学到了更有判别力的特征
- **TimeXer**: 轨迹和分类性能均中等

---

## 关键发现与分析

### 1. 整体性能对比

**跨任务最佳模型**:
- **Layer1 直接预测**: CausalForest (AUROC 0.5856)
- **Layer1 TSTR**: TSDiff (AUROC 0.5396)
- **Layer2 Readout**: iTransformer (AUROC 0.6563)
- **Layer2 轨迹**: SurvTraj (MSE 0.25)

### 2. 方法论对比

**直接预测 vs TSTR**:
- 直接预测方法普遍优于TSTR
- CausalForest (0.5856) > TSDiff TSTR (0.5396)
- 生成式模型在极度不平衡数据上表现受限

**Layer1 vs Layer2**:
- Layer2的iTransformer readout (0.6563) > Layer1的CausalForest (0.5856)
- 轨迹预测任务可能提供了额外的正则化效果

### 3. 模型稳定性分析

**方差最小（最稳定）**:
- TabDiff TSTR: AUROC std 0.0228
- TimeXer Layer2: MSE std 2.04

**方差最大（不稳定）**:
- iTransformer Layer1: AUROC std 0.2290
- TabSyn TSTR: AUROC std 0.1636

### 4. 数据不平衡挑战

所有模型在极度不平衡数据（1-2%正例率）上表现受限：
- AUPRC普遍很低（<0.06）
- F1分数普遍很低（<0.06）
- 说明模型难以准确识别少数正类样本

### 5. 新训练模型表现

**TSDiff (9 seeds)**:
- TSTR AUROC 0.5396，优于所有其他生成式模型
- 成功修复XGBoost cupy问题后训练完成
- 预测文件: `outputs/tstr_baselines/tsdiff_seed*_predictions.npz`
- 指标文件: `outputs/b2_baseline/tstr/tsdiff_seed*_tstr_metrics.json`

**STaSy (9 seeds)**:
- TSTR AUROC 0.4458，表现不佳
- 直接预测AUROC 0.3958，接近随机
- 预测文件: `outputs/tstr_baselines/stasy_seed*_predictions.npz`
- 指标文件: `outputs/b2_baseline/tstr/stasy_seed*_tstr_metrics.json`

**iTransformer Layer2 (5 seeds)**:
- 成功计算缺失指标
- Readout AUROC 0.6563，表现最佳
- 预测文件: `outputs/tslib_layer2/itransformer_seed*_layer2.npz`
- 指标文件: `outputs/b2_baseline/layer2/iTransformer_seed*_layer2_metrics.json`

---
## 模型排名

### Layer1 直接预测排名（按AUROC）

| 排名 | 模型 | AUROC | AUPRC | F1 | 综合评价 |
|------|------|-------|-------|-----|----------|
| 🥇 1 | CausalForest | 0.5856 | 0.0256 | 0.0335 | 最佳整体性能 |
| 🥈 2 | iTransformer | 0.5487 | 0.0180 | 0.0415 | 高方差，不稳定 |
| 🥉 3 | TSDiff | 0.5364 | 0.0229 | 0.0495 | 性能中等 |
| 4 | STaSy | 0.3958 | 0.0137 | 0.0140 | 表现最差 |

### Layer1 TSTR排名（按AUROC）

| 排名 | 模型 | AUROC | AUPRC | F1 | Seeds | 综合评价 |
|------|------|-------|-------|-----|-------|----------|
| 🥇 1 | TSDiff | 0.5396 | 0.0199 | 0.0370 | 9 | 新训练，最佳TSTR |
| 🥈 2 | SurvTraj | 0.5151 | 0.0513 | 0.0220 | 5 | 最高AUPRC |
| 🥉 3 | SSSD | 0.5000 | 0.0104 | 0.0080 | 5 | 完全随机 |
| 4 | TabDiff | 0.4941 | 0.0109 | 0.0208 | 5 | 稳定但偏低 |
| 5 | STaSy | 0.4458 | 0.0148 | 0.0180 | 9 | 新训练，表现不佳 |
| 6 | TabSyn | 0.4030 | 0.0164 | 0.0215 | 5 | 表现最差 |

### Layer2 Readout排名（按AUROC）

| 排名 | 模型 | Readout AUROC | Readout AUPRC | Readout F1 | 综合评价 |
|------|------|---------------|---------------|------------|----------|
| 🥇 1 | iTransformer | 0.6563 | 0.0277 | 0.0535 | 最佳分类性能 |
| 🥈 2 | SurvTraj | 0.5722 | 0.0119 | 0.0343 | 轨迹拟合最佳 |
| 🥉 3 | SSSD | 0.5647 | 0.0048 | 0.0067 | 轨迹精度高 |
| 4 | TimeXer | 0.5406 | 0.0077 | 0.0174 | 性能中等 |

---
## 结论与建议

### 主要结论

1. **最佳模型选择**:
   - 2年风险预测：推荐 **CausalForest** (AUROC 0.5856)
   - 7年轨迹预测：推荐 **iTransformer** (Readout AUROC 0.6563)

2. **方法论洞察**:
   - 直接预测优于TSTR生成式方法
   - Layer2轨迹预测可能提供更好的正则化
   - 极度不平衡数据对所有模型都是挑战

3. **新训练模型评估**:
   - **TSDiff**: 成功修复并训练，TSTR表现最佳
   - **STaSy**: 训练完成但性能不佳
   - **iTransformer Layer2**: 指标计算完成，表现优异

### 改进建议

**短期优化**:
1. 针对数据不平衡问题，尝试：
   - 调整类权重
   - 使用focal loss
   - SMOTE等过采样技术

2. 提升TSTR性能：
   - 增加合成样本数量（当前2000）
   - 调整生成模型训练轮数
   - 尝试其他下游分类器（当前XGBoost）

**长期研究方向**:
1. 探索因果推断方法（CausalForest表现最好）
2. 研究轨迹预测与分类性能的关系
3. 开发针对极度不平衡医疗数据的专用模型

### 技术债务

已修复问题：
- ✅ XGBoost cupy依赖问题
- ✅ 训练日志规范问题
- ✅ iTransformer Layer2指标缺失

待优化项：
- STaSy模型性能调优
- TSTR合成数据质量提升
- 模型稳定性改进（降低方差）

---

## 附录

### 数据文件位置汇总

**Layer1 直接预测**:
- 汇总表: `outputs/b2_baseline/summaries/baseline_layer1_direct.csv`
- 预测文件: `outputs/retained_baselines_b2/*_predictions.npz`
- 指标文件: `outputs/retained_baselines_b2/*_metrics.json`

**Layer1 TSTR**:
- 汇总表: `outputs/b2_baseline/summaries/baseline_layer1_tstr.csv`
- TSDiff预测: `outputs/tstr_baselines/tsdiff_seed*_predictions.npz`
- TSDiff指标: `outputs/b2_baseline/tstr/tsdiff_seed*_tstr_metrics.json`
- STaSy预测: `outputs/tstr_baselines/stasy_seed*_predictions.npz`
- STaSy指标: `outputs/b2_baseline/tstr/stasy_seed*_tstr_metrics.json`

**Layer2 轨迹预测**:
- 汇总表: `outputs/b2_baseline/summaries/baseline_layer2.csv`
- iTransformer预测: `outputs/tslib_layer2/itransformer_seed*_layer2.npz`
- iTransformer指标: `outputs/b2_baseline/layer2/iTransformer_seed*_layer2_metrics.json`
- 其他模型指标: `outputs/b2_baseline/layer2/*_layer2_metrics.json`

**训练日志**:
- TSDiff: `logs/tstr_tsdiff_seed*_full.log`
- STaSy: `logs/tstr_stasy_seed*_full.log`

### 实验配置

**训练参数**:
- Epochs: 30-50
- Batch Size: 64-512
- Learning Rate: 1e-3
- Optimizer: Adam

**TSTR配置**:
- 生成模型Epochs: 50
- 合成样本数: 2000
- 下游分类器: XGBoost

**评估协议**:
- 阈值选择: Youden's J statistic (验证集)
- 测试集评估: 使用验证集选定阈值
- 报告格式: mean ± std across seeds

---

**报告生成时间**: 2026-03-13  
**报告版本**: v2.0 (包含数据来源信息)

---
