# B2 Baseline 表格说明文档

## 表格结构说明

### 1. baseline_layer1_direct.csv - Layer1 直接预测表
**作用**: 评估模型在 Layer1 任务（2年风险预测）上的直接预测性能

**包含模型**:
- CausalForest: 传统因果森林方法
- iTransformer: 时序 Transformer 模型
- TSDiff: 时序扩散模型
- STaSy: 时序 SDE 模型

**指标**: AUROC, AUPRC, F1, Precision, Recall, Specificity, NPV, Accuracy, Balanced Accuracy, MCC, Brier Score, Calibration Intercept, Calibration Slope

---

### 2. baseline_layer1_tstr.csv - Layer1 TSTR 表（生成式模型）
**作用**: 评估生成式模型通过 TSTR（Train on Synthetic, Test on Real）范式在 Layer1 任务上的性能

**TSTR 流程**:
1. 在真实训练数据上训练生成模型
2. 生成合成数据
3. 在合成数据上训练下游分类器
4. 在真实测试数据上评估分类器

**包含模型**:
- TabSyn: VAE + Diffusion 生成模型
- TabDiff: Diffusion 生成模型
- SurvTraj: 生存分析轨迹生成模型
- SSSD: 结构化状态空间扩散模型

**指标**: 与 Layer1 直接预测表相同

**与主表的关系**: 
- 主表（Layer1 直接预测表）: 模型直接在真实数据上训练和预测
- TSTR 表: 生成式模型先生成合成数据，再在合成数据上训练分类器

---

### 3. baseline_layer2.csv - Layer2 轨迹预测表
**作用**: 评估模型在 Layer2 任务（未来7年风险轨迹预测）上的性能

**包含模型**:
- iTransformer: 时序 Transformer（轨迹预测）
- TimeXer: 外生变量时序模型
- SSSD: 结构化状态空间扩散模型
- SurvTraj: 生存分析轨迹生成模型

**指标**:
- **Trajectory 指标**: MSE, MAE, Valid Coverage（评估7年轨迹预测质量）
- **2-year Readout 指标**: AUROC, AUPRC, F1（评估轨迹第0年的2年风险预测）

---

### 4. baseline_efficiency.csv - 效率表
**作用**: 对比各模型的训练效率和资源消耗

**指标**:
- total_training_wall_clock_sec: 总训练时间（秒）
- peak_gpu_memory_mb: 峰值 GPU 内存（MB）
- total_params: 总参数量
- trainable_params: 可训练参数量

**格式**: 每个模型显示 mean ± std（5 seeds）

---

## 关键问题修复记录

### 问题1: iTransformer 指标异常
- **原因**: 预测值极低（max<0.9），固定阈值0.5导致全部预测为负类
- **修复**: 使用 Youden's J statistic 动态选择最优阈值
- **结果**: F1 从 0.0000 提升到 0.0415 ± 0.0256

### 问题2: Layer2 表指标不足
- **原因**: 只有 trajectory 指标，缺少分类指标
- **修复**: 补充 2-year readout 的 AUROC, AUPRC, F1 指标
- **结果**: Layer2 表现在包含完整的轨迹和分类指标

### 问题3: TSTR 表定位不清
- **原因**: TSTR 表与主表模型重复，作用不明确
- **修复**: 
  - 主表改名为 `baseline_layer1_direct.csv`
  - TSTR 表改名为 `baseline_layer1_tstr.csv`
  - 添加详细说明文档

### 问题4: 效率表格式混乱
- **原因**: 未计算 mean±std，缺少大部分模型数据
- **修复**: 
  - 计算每个模型的 mean ± std
  - 补充缺失的效率数据
  - 统一格式

---

## 最终 Baseline 名单

### Layer1 直接预测（4个模型）
1. CausalForest
2. iTransformer
3. TSDiff
4. STaSy

### Layer1 TSTR（4个生成式模型）
1. TabSyn
2. TabDiff
3. SurvTraj
4. SSSD

### Layer2 轨迹预测（4个模型）
1. iTransformer
2. TimeXer
3. SSSD
4. SurvTraj

**总计**: 8 个 Layer1 模型（4直接+4TSTR），4 个 Layer2 模型
